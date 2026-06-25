"""Substrate-agnostic graph store protocol for the concept knowledge graph.

Application code (extractors, retrievers, lifecycle jobs) depends on
``GraphStore``, never on Kuzu, Neo4j, or any other persistence driver.
Cypher and vendor-specific query strings live inside concrete store
implementations only (see ``CONCEPT_GRAPH.md`` §"Compatibility shim").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from open_webui.retrieval.concepts.schema import (
    Artifact,
    Concept,
    ConceptKind,
    Edge,
    EdgeType,
)

# Shared filter for subgraph-scoped traversals and analytics.
EdgeFilter = Sequence[EdgeType] | None


@runtime_checkable
class GraphTransaction(Protocol):
    """Unit-of-work boundary for batched graph writes.

    **Canonical usage:** explicit ``commit()`` / ``rollback()`` after
    ``begin_transaction()``. The context manager (``__enter__`` /
    ``__exit__``) is a safety net: ``__exit__`` calls ``rollback()``
    when the block exits with an exception and the transaction has not
    yet been committed. Successful blocks do **not** auto-commit —
    callers must call ``commit()`` explicitly so partial writes never
    persist accidentally.

    Production stores (Kuzu, Neo4j) will map this to native transactions;
    ``InMemoryGraphStore`` uses snapshot-and-restore.
    """

    def commit(self) -> None:
        """Persist all writes performed inside this transaction."""
        ...

    def rollback(self) -> None:
        """Discard all writes performed inside this transaction."""
        ...

    def __enter__(self) -> GraphTransaction:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        ...


# GraphStore is intentionally NOT @runtime_checkable: contract tests and
# static type checkers validate implementations; isinstance() checks against
# large Protocols are slow and brittle (see typing docs on runtime_checkable).
class GraphStore(Protocol):
    """Substrate-agnostic graph operations.

    Implementations: ``InMemoryGraphStore``, ``KuzuGraphStore`` (step 6),
    ``Neo4jGraphStore`` (future). All application code depends on this
    protocol, never on a vendor driver.
    """

    def upsert_concept(self, concept: Concept) -> int:
        """Insert or merge a concept keyed by ``(name, kind)``.

        Answers the extractor's "have I seen this token before?" pattern
        (``CONCEPT_GRAPH.md`` decision 1). Idempotent on ``(name, kind)``:
        re-upsert updates ``last_seen_at`` and unions ``original_tokens``
        while preserving the stable integer id required by edges and
        centrality scores.
        """
        ...

    def upsert_concepts_batch(self, concepts: Sequence[Concept]) -> list[int]:
        """Upsert many concepts in one transaction-equivalent call.

        Returns ids in the same positional order as ``concepts``. Idempotency
        semantics MUST match ``upsert_concept`` per-element (lookup by
        ``(name, kind)``; existing concepts get their id returned and any
        mutable fields updated). Implementations SHOULD use this path to
        amortize round-trip overhead on bulk ingest; in-memory backends
        may simply iterate.

        Empty input returns ``[]``. Returns the SAME id twice if the caller
        submits the same ``(name, kind)`` twice in one batch (idempotent on
        duplicates within the batch too).
        """
        ...

    def upsert_edge(self, edge: Edge) -> None:
        """Insert or merge an edge keyed by ``(type, src_id, dst_id)``.

        Answers incremental ingest: the same chunk re-processed must not
        duplicate relationship rows. Property merge semantics are
        edge-type-specific (counts summed, positions unioned, etc.).
        """
        ...

    def upsert_edges_batch(self, edges: Sequence[Edge]) -> None:
        """Upsert many edges. Edges of mixed type are accepted; the
        implementation may internally group by type for efficiency.
        Property-merge semantics MUST match ``upsert_edge``
        (``merge_edge_properties`` for additive/min/max/concat fields).

        Endpoint ids must already exist in the store — empty/missing
        endpoints raise ``ValueError`` (memory store) or whatever native
        error the backend produces (kuzu). Callers must
        ``upsert_concepts_batch`` + ``upsert_artifacts_batch`` BEFORE this
        call.
        """
        ...

    def get_concept(self, concept_id: int) -> Concept | None:
        """Point lookup by stable integer id.

        Used by retrievers and alias resolution to hydrate a single node
        after id-based traversals return only identifiers.
        """
        ...

    def find_concept_by_name(
        self,
        name: str,
        kind: ConceptKind | None = None,
    ) -> int | None:
        """Find a concept's graph id by exact name match. If ``kind`` is provided,
        restricts the match to that kind. Returns None if no match.

        This is a precise-lookup primitive (case-sensitive, exact-equality). It
        is NOT a search; for fuzzy/vector retrieval, use ``vector_search``.

        Idiomatic use: the router's seed-resolution step calls this to convert
        free-text query tokens into concept ids before delegating to a graph-
        walking retriever."""
        ...

    def upsert_artifact(self, artifact: Artifact) -> int:
        """Upsert an artifact and return its graph id.

        Idempotency key is ``(path, chunk_index)`` for ``kind=CHUNK``, or
        ``(path,)`` for ``kind=SOURCE_FILE`` / ``DOC_FILE``. The
        ``(path, chunk_index)`` key lets the builder re-run on a changed
        file without orphaning prior artifacts: the same chunk slot merges
        in place while mutable fields refresh.

        Re-upserting an existing artifact:

        - returns the existing id (does NOT mint a new one);
        - updates ``last_modified_at`` to the new value;
        - updates ``byte_start`` / ``byte_end`` / ``language`` to the new
          values (these can change as the chunker re-runs);
        - does NOT touch ``kind`` — kind is immutable; upserting with a
          different kind on an existing key raises ``ValueError``.

        First insert assigns a monotonic id, mirroring ``upsert_concept``.
        """
        ...

    def upsert_artifacts_batch(self, artifacts: Sequence[Artifact]) -> list[int]:
        """Same contract as ``upsert_concepts_batch`` but for ``Artifact``.

        Idempotency key matches ``upsert_artifact`` per-element
        (``(path, chunk_index)`` for ``CHUNK``; ``(path,)`` for
        ``SOURCE_FILE`` / ``DOC_FILE``). ``ValueError`` on kind mismatch
        for any element.
        """
        ...

    def get_artifact(self, artifact_id: int) -> Artifact | None:
        """Fetch an artifact by graph id.

        Used by the builder and edge writers to hydrate artifact endpoints
        after id-based lookups. Returns ``None`` when absent.
        """
        ...

    def neighborhood(
        self,
        anchor_id: int,
        *,
        radius: int = 1,
        edge_types: EdgeFilter = None,
        limit: int = 100,
    ) -> Sequence[Concept]:
        """Bounded BFS from an anchor concept.

        Answers the neighborhood-walk retrieval primitive
        (``retrieve/neighborhood.py``). Resolves ``anchor_id`` through
        ``resolve_alias`` first so alias nodes transparently inherit
        their canonical's neighborhood (decision 3). Respects ``limit``
        as a hard budget on returned concepts.

        **Determinism contract (Phase 1.5):** Returns results in a deterministic
        order: hop distance from anchor ascending, edge weight on the
        discovering edge descending (within a ring), concept id ascending
        (tiebreaker). Two calls with identical arguments against an unchanged
        store return identical results. Implementations MUST honor this contract —
        adjacency-set iteration order is not a substitute. Enforced by
        ``test_store_contract.py`` against both ``InMemoryGraphStore`` and
        ``KuzuGraphStore``.
        """
        ...

    def shortest_path(
        self,
        src_id: int,
        dst_id: int,
        *,
        edge_types: EdgeFilter = None,
        max_hops: int = 6,
    ) -> Sequence[Concept]:
        """Shortest path between two concepts within ``max_hops``.

        Answers "how are X and Y related?" queries and path-aware
        re-ranking. Returns the ordered list of concepts including
        endpoints, or an empty sequence when no path exists within the
        hop budget.
        """
        ...

    def pagerank(
        self,
        *,
        edge_types: EdgeFilter = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> Mapping[int, float]:
        """Power-iteration PageRank over a filtered edge subgraph.

        Answers nightly centrality precompute (``lifecycle/centrality.py``).
        Default subgraph is ``CO_OCCURS_WITH`` when ``edge_types`` is
        ``None`` — co-occurrence density drives retrieval boosting
        (decision 2, v1 scope item 3).

        **Determinism contract (Phase 1.5):** Returns scores stable to
        ±1e-6 (``_PAGERANK_TOLERANCE``) across runs. Two calls with
        identical arguments against an unchanged store return mappings whose
        values are stable to ±1e-6. Enforced by ``test_store_contract.py``
        against both ``InMemoryGraphStore`` and ``KuzuGraphStore``.
        """
        ...

    def personalized_pagerank(
        self,
        seed_ids: Sequence[int],
        *,
        edge_types: EdgeFilter = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> Mapping[int, float]:
        """Personalized PageRank with teleport mass concentrated on ``seed_ids``.

        Unlike ``pagerank``, the random-walk teleport vector is non-uniform:
        instead of teleporting to every node with probability ``1/n``, the walker
        teleports back to one of the seed nodes (uniformly distributed across the
        seeds — equal mass on each seed). The resulting stationary distribution
        scores nodes by their proximity to the seed set, NOT by global popularity.
        This is the principled ranking signal for query-driven retrieval: a node
        that co-occurs heavily with seeds gets a high score even if it has low
        global PageRank, and vice versa.

        Empty ``seed_ids`` returns ``{}`` (no walk to perform). Seeds that don't
        exist in the graph are silently dropped from the teleport set; if all
        seeds drop, returns ``{}``.

        ``edge_types``: same semantics as ``pagerank`` (None defaults to
        ``CO_OCCURS_WITH``).

        **Determinism contract:** Returns scores stable to +/- 1e-6 across runs
        given identical arguments and unchanged store state, matching
        ``pagerank``'s determinism guarantee.
        """
        ...

    def vector_search(
        self,
        embedding: Sequence[float],
        *,
        kind: ConceptKind | None = None,
        limit: int = 20,
    ) -> Sequence[tuple[Concept, float]]:
        """Brute-force cosine similarity over stored concept embeddings.

        Answers hybrid retrieval's candidate generation when no graph
        anchor is detected. Filtered by ``kind`` so phrase-concept
        queries do not pollute atomic results (decision 1).

        **Determinism contract (Phase 1.5):** Returns results in a deterministic
        order: cosine score descending, concept id ascending (tiebreaker).
        Two calls with identical arguments against an unchanged store return
        identical results. Enforced by ``test_store_contract.py`` against both
        ``InMemoryGraphStore`` and ``KuzuGraphStore``.
        """
        ...

    def resolve_alias(self, concept_id: int) -> int:
        """Follow ``IS_CANONICAL_ALIAS_OF`` edges to the canonical concept.

        Transparent alias resolution (decision 3): every read path calls
        this before traversal so rename migrations never strand queries
        on stale names. Raises ``RuntimeError`` if the alias chain
        contains a cycle (cardinality invariant: alias edges form a tree).
        """
        ...

    def begin_transaction(self) -> GraphTransaction:
        """Start a transactional batch of writes.

        Answers delta-apply and rebuild orchestrators that need all-or-
        nothing writes across many concepts and edges. Implementations
        map to native transactions; the in-memory store snapshots state.
        """
        ...
