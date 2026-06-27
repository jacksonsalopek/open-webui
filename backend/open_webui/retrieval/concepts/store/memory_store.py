"""In-memory ``GraphStore`` for unit tests, REPL dev, and CI without Kuzu."""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType
from typing import Any

from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    Edge,
    EdgeType,
)
from open_webui.retrieval.concepts.store.protocol import EdgeFilter, GraphTransaction

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a project dependency
    _HAS_NUMPY = False

try:
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

# Stop power iteration when the L1 delta falls below this threshold.
_PAGERANK_TOLERANCE = 1e-6


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def merge_edge_properties(
    edge_type: EdgeType,
    existing: Mapping[str, object],
    incoming: Mapping[str, object],
) -> dict[str, object]:
    """Merge edge properties on idempotent re-upsert.

    Semantics per v1 edge type:
    - ``DEFINES``: sum ``count``.
    - ``REFERENCES``: sum ``count``; union ``positions`` (sorted, deduped).
    - ``CO_OCCURS_WITH``: take latest ``weight`` from incoming; sum ``chunk_count``.
    - ``IS_NAMED_IN``: keep earliest ``first_seen_at``.
    - ``IS_CANONICAL_ALIAS_OF``: take incoming timestamps (latest governance wins).
    """
    if edge_type == EdgeType.DEFINES:
        return {
            'count': int(existing['count']) + int(incoming['count']),
        }

    if edge_type == EdgeType.REFERENCES:
        pos_a = existing.get('positions')
        pos_b = incoming.get('positions')
        merged: list[int] = []
        if pos_a is not None:
            merged.extend(int(x) for x in pos_a)  # type: ignore[union-attr]
        if pos_b is not None:
            merged.extend(int(x) for x in pos_b)  # type: ignore[union-attr]
        result: dict[str, object] = {
            'count': int(existing['count']) + int(incoming['count']),
        }
        if merged:
            result['positions'] = sorted(set(merged))
        return result

    if edge_type == EdgeType.CO_OCCURS_WITH:
        return {
            'weight': float(incoming['weight']),
            'chunk_count': int(existing['chunk_count']) + int(incoming['chunk_count']),
        }

    if edge_type == EdgeType.IS_NAMED_IN:
        existing_dt = _parse_dt(existing['first_seen_at'])
        incoming_dt = _parse_dt(incoming['first_seen_at'])
        return {
            'first_seen_at': (
                existing_dt if existing_dt <= incoming_dt else incoming_dt
            ).isoformat(),
        }

    if edge_type == EdgeType.IS_CANONICAL_ALIAS_OF:
        merged_alias: dict[str, object] = {}
        for key in ('introduced_at', 'planned_removal_at', 'removed_at'):
            if key in incoming:
                value = incoming[key]
                merged_alias[key] = (
                    value.isoformat() if isinstance(value, datetime) else value
                )
            elif key in existing:
                value = existing[key]
                merged_alias[key] = (
                    value if isinstance(value, str) else _parse_dt(value).isoformat()
                )
        return merged_alias

    raise ValueError(f'unsupported edge type for merge: {edge_type!r}')


def _artifact_idempotency_key(artifact: Artifact) -> tuple[str, int | None]:
    """Return the lookup key for artifact upsert idempotency.

    CHUNK rows key on ``(path, chunk_index)``; file-level kinds
    (``SOURCE_FILE``, ``DOC_FILE``, …) key on ``(path,)`` only — stored
    as ``(path, None)`` so re-upserts ignore a spurious ``chunk_index``.
    """
    if artifact.kind == ArtifactKind.CHUNK:
        return (artifact.path, artifact.chunk_index)
    return (artifact.path, None)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    if _HAS_NUMPY:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class _MemoryTransaction:
    """Snapshot-and-restore transaction for ``InMemoryGraphStore``.

    Production stores will use native engine transactions; this double
    deep-copies the entire store state on ``__enter__`` and restores it
    on ``rollback()`` (or on ``__exit__`` when an exception propagates).
    """

    def __init__(self, store: InMemoryGraphStore) -> None:
        self._store = store
        self._snapshot: dict[str, Any] | None = None
        self._committed = False

    def commit(self) -> None:
        self._committed = True
        self._snapshot = None

    def rollback(self) -> None:
        if self._snapshot is not None:
            self._store._restore_state(self._snapshot)
        self._snapshot = None
        self._committed = True

    def __enter__(self) -> _MemoryTransaction:
        self._snapshot = self._store._snapshot_state()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is not None and not self._committed:
            self.rollback()


class InMemoryGraphStore:
    """Dict-backed ``GraphStore`` for tests and local development."""

    def __init__(self) -> None:
        self._concepts: dict[int, Concept] = {}
        self._by_name_kind: dict[tuple[str, ConceptKind], int] = {}
        self._artifacts: dict[int, Artifact] = {}
        self._by_path_chunk: dict[tuple[str, int | None], int] = {}
        self._edges: dict[tuple[EdgeType, int, int], Edge] = {}
        self._adjacency: dict[int, set[tuple[int, EdgeType]]] = {}
        self._next_id = 1

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            'concepts': copy.deepcopy(self._concepts),
            'by_name_kind': copy.deepcopy(self._by_name_kind),
            'artifacts': copy.deepcopy(self._artifacts),
            'by_path_chunk': copy.deepcopy(self._by_path_chunk),
            'edges': copy.deepcopy(self._edges),
            'adjacency': copy.deepcopy(self._adjacency),
            'next_id': self._next_id,
        }

    def _restore_state(self, snapshot: dict[str, Any]) -> None:
        self._concepts = snapshot['concepts']
        self._by_name_kind = snapshot['by_name_kind']
        self._artifacts = snapshot['artifacts']
        self._by_path_chunk = snapshot['by_path_chunk']
        self._edges = snapshot['edges']
        self._adjacency = snapshot['adjacency']
        self._next_id = snapshot['next_id']

    def upsert_concept(self, concept: Concept) -> int:
        key = (concept.name, concept.kind)
        existing_id = self._by_name_kind.get(key)
        if existing_id is not None:
            existing = self._concepts[existing_id]
            merged_tokens = tuple(
                dict.fromkeys((*existing.original_tokens, *concept.original_tokens)),
            )
            updated = Concept(
                id=existing_id,
                name=existing.name,
                kind=existing.kind,
                first_seen_at=existing.first_seen_at,
                last_seen_at=concept.last_seen_at,
                centrality_score=existing.centrality_score,
                embedding=existing.embedding,
                definition=existing.definition,
                language_hint=existing.language_hint,
                original_tokens=merged_tokens,
            )
            self._concepts[existing_id] = updated
            return existing_id

        new_id = self._next_id
        self._next_id += 1
        stored = Concept(
            id=new_id,
            name=concept.name,
            kind=concept.kind,
            first_seen_at=concept.first_seen_at,
            last_seen_at=concept.last_seen_at,
            centrality_score=concept.centrality_score,
            embedding=concept.embedding,
            definition=concept.definition,
            language_hint=concept.language_hint,
            original_tokens=concept.original_tokens,
        )
        self._concepts[new_id] = stored
        self._by_name_kind[key] = new_id
        return new_id

    def set_concept_embedding(
        self,
        concept_id: int,
        embedding: tuple[float, ...] | None,
    ) -> None:
        if embedding == ():
            raise ValueError(
                f'set_concept_embedding({concept_id}): empty-tuple embedding is '
                f'rejected; pass None to clear instead.',
            )
        existing = self._concepts.get(concept_id)
        if existing is None:
            raise KeyError(concept_id)
        self._concepts[concept_id] = replace(existing, embedding=embedding)

    def upsert_concepts_batch(self, concepts: Sequence[Concept]) -> list[int]:
        if not concepts:
            return []
        return [self.upsert_concept(c) for c in concepts]

    def upsert_edge(self, edge: Edge) -> None:
        key = (edge.type, edge.src_id, edge.dst_id)
        existing = self._edges.get(key)
        if existing is not None:
            merged = merge_edge_properties(
                edge.type,
                existing.properties,
                edge.properties,
            )
            self._edges[key] = Edge(
                type=edge.type,
                src_id=edge.src_id,
                dst_id=edge.dst_id,
                properties=MappingProxyType(merged),
            )
            return

        self._edges[key] = edge
        self._adjacency.setdefault(edge.src_id, set()).add((edge.dst_id, edge.type))
        self._adjacency.setdefault(edge.dst_id, set())

    def upsert_edges_batch(self, edges: Sequence[Edge]) -> None:
        if not edges:
            return
        for edge in edges:
            self._validate_edge_endpoints(edge)
            self.upsert_edge(edge)

    def _validate_edge_endpoints(self, edge: Edge) -> None:
        meta = {
            EdgeType.DEFINES: ('Artifact', 'Concept'),
            EdgeType.REFERENCES: ('Artifact', 'Concept'),
            EdgeType.CO_OCCURS_WITH: ('Concept', 'Concept'),
            EdgeType.IS_NAMED_IN: ('Concept', 'Artifact'),
            EdgeType.IS_CANONICAL_ALIAS_OF: ('Concept', 'Concept'),
        }
        from_label, to_label = meta[edge.type]
        if from_label == 'Concept' and edge.src_id not in self._concepts:
            raise ValueError(f'missing concept endpoint src_id={edge.src_id}')
        if from_label == 'Artifact' and edge.src_id not in self._artifacts:
            raise ValueError(f'missing artifact endpoint src_id={edge.src_id}')
        if to_label == 'Concept' and edge.dst_id not in self._concepts:
            raise ValueError(f'missing concept endpoint dst_id={edge.dst_id}')
        if to_label == 'Artifact' and edge.dst_id not in self._artifacts:
            raise ValueError(f'missing artifact endpoint dst_id={edge.dst_id}')

    def get_concept(self, concept_id: int) -> Concept | None:
        return self._concepts.get(concept_id)

    def list_concepts(self) -> Iterable[Concept]:
        for cid in sorted(self._concepts):
            yield self._concepts[cid]

    def find_concept_by_name(
        self,
        name: str,
        kind: ConceptKind | None = None,
    ) -> int | None:
        if kind is not None:
            result = self._by_name_kind.get((name, kind))
            return int(result) if result is not None else None
        for (concept_name, _concept_kind), concept_id in self._by_name_kind.items():
            if concept_name == name:
                return int(concept_id)
        return None

    def upsert_artifact(self, artifact: Artifact) -> int:
        key = _artifact_idempotency_key(artifact)
        existing_id = self._by_path_chunk.get(key)
        if existing_id is not None:
            existing = self._artifacts[existing_id]
            if existing.kind != artifact.kind:
                raise ValueError(
                    f'artifact kind mismatch on idempotency key {key!r}: '
                    f'existing={existing.kind!r}, incoming={artifact.kind!r}',
                )
            updated = replace(
                existing,
                last_modified_at=artifact.last_modified_at,
                byte_start=artifact.byte_start,
                byte_end=artifact.byte_end,
                language=artifact.language,
            )
            self._artifacts[existing_id] = updated
            return existing_id

        new_id = self._next_id
        self._next_id += 1
        stored = replace(artifact, id=new_id)
        self._artifacts[new_id] = stored
        self._by_path_chunk[key] = new_id
        return new_id

    def upsert_artifacts_batch(self, artifacts: Sequence[Artifact]) -> list[int]:
        if not artifacts:
            return []
        return [self.upsert_artifact(a) for a in artifacts]

    def get_artifact(self, artifact_id: int) -> Artifact | None:
        return self._artifacts.get(artifact_id)

    def _normalize_edge_types(self, edge_types: EdgeFilter) -> set[EdgeType]:
        if edge_types is None:
            return {EdgeType.CO_OCCURS_WITH}
        return set(edge_types)

    def _edge_weight(
        self,
        edge_type: EdgeType,
        src_id: int,
        dst_id: int,
    ) -> float:
        edge = self._edges.get((edge_type, src_id, dst_id))
        if edge is None:
            return 1.0
        weight = edge.properties.get('weight')
        if weight is not None:
            return float(weight)
        return 1.0

    def _neighbors(
        self,
        concept_id: int,
        allowed: set[EdgeType],
    ) -> list[tuple[int, float]]:
        weight_by_neighbor: dict[int, float] = {}
        for neighbor_id, edge_type in self._adjacency.get(concept_id, ()):
            if edge_type not in allowed:
                continue
            weight = self._edge_weight(edge_type, concept_id, neighbor_id)
            existing = weight_by_neighbor.get(neighbor_id)
            if existing is None or weight > existing:
                weight_by_neighbor[neighbor_id] = weight
        result = list(weight_by_neighbor.items())
        result.sort(key=lambda item: (-item[1], item[0]))
        return result

    def neighborhood(
        self,
        anchor_id: int,
        *,
        radius: int = 1,
        edge_types: EdgeFilter = None,
        limit: int = 100,
    ) -> list[Concept]:
        resolved = self.resolve_alias(anchor_id)
        allowed = self._normalize_edge_types(edge_types)
        if resolved not in self._concepts or radius < 1 or limit < 1:
            return []

        seen: set[int] = {resolved}
        frontier: list[int] = [resolved]
        collected: list[Concept] = []

        for _ in range(radius):
            next_frontier: list[int] = []
            for node_id in frontier:
                for neighbor_id, _weight in self._neighbors(node_id, allowed):
                    if neighbor_id in seen:
                        continue
                    seen.add(neighbor_id)
                    next_frontier.append(neighbor_id)
                    concept = self._concepts.get(neighbor_id)
                    if concept is not None:
                        collected.append(concept)
                        if len(collected) >= limit:
                            return collected[:limit]
            frontier = next_frontier
            if not frontier:
                break

        return collected[:limit]

    def shortest_path(
        self,
        src_id: int,
        dst_id: int,
        *,
        edge_types: EdgeFilter = None,
        max_hops: int = 6,
    ) -> list[Concept]:
        if src_id == dst_id:
            concept = self._concepts.get(src_id)
            return [concept] if concept is not None else []

        allowed = self._normalize_edge_types(edge_types)
        if src_id not in self._concepts or dst_id not in self._concepts:
            return []

        parent: dict[int, int | None] = {src_id: None}
        queue: deque[tuple[int, int]] = deque([(src_id, 0)])
        found = False

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor_id, _weight in self._neighbors(current, allowed):
                if neighbor_id in parent:
                    continue
                parent[neighbor_id] = current
                if neighbor_id == dst_id:
                    found = True
                    queue.clear()
                    break
                queue.append((neighbor_id, depth + 1))

        if not found:
            return []

        path_ids: list[int] = []
        cursor: int | None = dst_id
        while cursor is not None:
            path_ids.append(cursor)
            cursor = parent[cursor]
        path_ids.reverse()
        return [self._concepts[cid] for cid in path_ids if cid in self._concepts]

    def pagerank(
        self,
        *,
        edge_types: EdgeFilter = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[int, float]:
        allowed = self._normalize_edge_types(edge_types)
        nodes = sorted(self._concepts)
        if not nodes:
            return {}

        index = {node_id: i for i, node_id in enumerate(nodes)}
        n = len(nodes)
        out_neighbors: list[list[int]] = [[] for _ in range(n)]
        in_neighbors: list[list[int]] = [[] for _ in range(n)]

        for edge in self._edges.values():
            if edge.type not in allowed:
                continue
            if edge.src_id not in index or edge.dst_id not in index:
                continue
            src_i = index[edge.src_id]
            dst_i = index[edge.dst_id]
            out_neighbors[src_i].append(dst_i)
            in_neighbors[dst_i].append(src_i)

        scores = [1.0 / n] * n
        teleport = (1.0 - damping) / n

        for _ in range(iterations):
            new_scores = [teleport] * n
            dangling_mass = 0.0
            for i in range(n):
                out_deg = len(out_neighbors[i])
                if out_deg == 0:
                    dangling_mass += scores[i]
                else:
                    share = damping * scores[i] / out_deg
                    for j in out_neighbors[i]:
                        new_scores[j] += share
            if dangling_mass:
                spread = damping * dangling_mass / n
                for j in range(n):
                    new_scores[j] += spread
            delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if delta < _PAGERANK_TOLERANCE:
                break

        return {nodes[i]: scores[i] for i in range(n)}

    def personalized_pagerank(
        self,
        seed_ids: Sequence[int],
        *,
        edge_types: EdgeFilter = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[int, float]:
        if not seed_ids:
            return {}

        allowed = self._normalize_edge_types(edge_types)
        nodes = sorted(self._concepts)
        if not nodes:
            return {}

        valid_seeds = [s for s in seed_ids if s in self._concepts]
        if not valid_seeds:
            return {}

        index = {node_id: i for i, node_id in enumerate(nodes)}
        seed_set = set(valid_seeds)
        n = len(nodes)
        out_neighbors: list[list[int]] = [[] for _ in range(n)]

        for edge in self._edges.values():
            if edge.type not in allowed:
                continue
            if edge.src_id not in index or edge.dst_id not in index:
                continue
            src_i = index[edge.src_id]
            dst_i = index[edge.dst_id]
            out_neighbors[src_i].append(dst_i)

        personalization = [0.0] * n
        seed_mass = 1.0 / len(seed_set)
        for s in seed_set:
            if s in index:
                personalization[index[s]] = seed_mass

        scores = [1.0 / n] * n

        for _ in range(iterations):
            new_scores = [(1.0 - damping) * personalization[j] for j in range(n)]
            dangling_mass = 0.0
            for i in range(n):
                out_deg = len(out_neighbors[i])
                if out_deg == 0:
                    dangling_mass += scores[i]
                else:
                    share = damping * scores[i] / out_deg
                    for j in out_neighbors[i]:
                        new_scores[j] += share
            if dangling_mass:
                for j in range(n):
                    new_scores[j] += damping * dangling_mass * personalization[j]
            delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if delta < _PAGERANK_TOLERANCE:
                break

        return {nodes[i]: scores[i] for i in range(n)}

    def vector_search(
        self,
        embedding: Sequence[float],
        *,
        kind: ConceptKind | None = None,
        limit: int = 20,
    ) -> list[tuple[Concept, float]]:
        if limit < 1:
            return []

        scored: list[tuple[Concept, float]] = []
        for concept in self._concepts.values():
            if kind is not None and concept.kind != kind:
                continue
            if concept.embedding is None:
                continue
            score = _cosine_similarity(embedding, concept.embedding)
            scored.append((concept, score))

        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:limit]

    def resolve_alias(self, concept_id: int) -> int:
        if concept_id not in self._concepts:
            return concept_id

        visited: list[int] = []
        current = concept_id
        while True:
            if current in visited:
                cycle = ' -> '.join(str(x) for x in (*visited, current))
                raise RuntimeError(
                    f'alias cycle detected involving concept ids: {cycle}',
                )
            visited.append(current)
            next_id: int | None = None
            for edge in self._edges.values():
                if (
                    edge.type == EdgeType.IS_CANONICAL_ALIAS_OF
                    and edge.src_id == current
                ):
                    next_id = edge.dst_id
                    break
            if next_id is None:
                return current
            current = next_id

    def begin_transaction(self) -> GraphTransaction:
        return _MemoryTransaction(self)

    def pagerank_networkx_reference(
        self,
        *,
        edge_types: EdgeFilter = None,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> dict[int, float]:
        """Optional cross-validation helper when NetworkX is installed."""
        if not _HAS_NETWORKX:
            raise ImportError('networkx is not installed')
        allowed = self._normalize_edge_types(edge_types)
        graph = nx.DiGraph()
        for concept_id in self._concepts:
            graph.add_node(concept_id)
        for edge in self._edges.values():
            if edge.type in allowed:
                graph.add_edge(edge.src_id, edge.dst_id)
        scores = nx.pagerank(
            graph,
            alpha=damping,
            max_iter=iterations,
            tol=_PAGERANK_TOLERANCE,
        )
        return {int(k): float(v) for k, v in scores.items()}
