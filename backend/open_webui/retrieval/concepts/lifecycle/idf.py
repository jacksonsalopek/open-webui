"""Inverse-document-frequency precompute for the concept knowledge graph.

The neighborhood retriever's tiebreaker historically reached for PageRank
centrality, which correlates strongly with raw frequency: globally popular
concepts like ``user``, ``model``, ``view`` win every tie even when they
add no discriminative signal to the answer. IDF inverts that bias — rare
concepts (in few artifacts) score higher than common ones.

The score is keyed by concept id and cached per-store via a module-level
``WeakKeyDictionary`` (mirroring ``lifecycle.centrality``). Computation is
read-only: it walks artifact-touching edges (``DEFINES``, ``REFERENCES``,
``IS_NAMED_IN``) and counts the distinct artifacts each concept appears in.

IDF is computed with the standard ``log((N + 1) / (df + 1)) + 1`` smoothing
so concepts that appear in *every* artifact still receive a small positive
weight, and unseen concepts receive a maximal weight rather than zero.
"""

from __future__ import annotations

import logging
import math
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from open_webui.retrieval.concepts.schema import EdgeType
from open_webui.retrieval.concepts.store.protocol import GraphStore

log = logging.getLogger(__name__)

_IDF_CACHE: weakref.WeakKeyDictionary[GraphStore, IdfScores] = (
    weakref.WeakKeyDictionary()
)

# Edge types whose endpoints are (artifact -> concept) or (concept ->
# artifact). These are the only edges that contribute to document-frequency
# counting; CO_OCCURS_WITH and IS_CANONICAL_ALIAS_OF are intra-concept.
_ARTIFACT_TOUCHING_EDGES: frozenset[EdgeType] = frozenset(
    {EdgeType.DEFINES, EdgeType.REFERENCES, EdgeType.IS_NAMED_IN},
)


@dataclass(frozen=True, slots=True)
class IdfScores:
    """Per-concept inverse-document-frequency scores plus the corpus size
    used to compute them. Higher means more discriminative (fewer
    artifacts mention the concept)."""

    scores: Mapping[int, float]
    total_artifacts: int
    computed_at: datetime

    def get(self, concept_id: int, default: float = 0.0) -> float:
        return self.scores.get(concept_id, default)


def _iter_artifact_concept_pairs(
    store: GraphStore,
) -> Iterable[tuple[int, int]]:
    """Yield ``(artifact_id, concept_id)`` pairs for every artifact-touching
    edge in ``store``.

    Tries a public protocol-shaped method first; falls back to duck-typing
    the in-memory store's ``_edges`` dict. The duck-type path is the Phase 1
    expedient — adding ``iter_artifact_concept_edges`` to the protocol is
    queued for Phase 2 once the kuzu backend exits debug.
    """
    public = getattr(store, 'iter_artifact_concept_edges', None)
    if callable(public):
        yield from public(_ARTIFACT_TOUCHING_EDGES)
        return

    edges = getattr(store, '_edges', None)
    artifacts = getattr(store, '_artifacts', None)
    concepts = getattr(store, '_concepts', None)
    if edges is None or artifacts is None or concepts is None:
        log.warning(
            'idf.compute: store %r exposes no edge iteration; returning empty',
            type(store).__name__,
        )
        return

    for edge in edges.values():
        if edge.type not in _ARTIFACT_TOUCHING_EDGES:
            continue
        artifact_id, concept_id = _normalize_endpoints(
            edge.type,
            edge.src_id,
            edge.dst_id,
            artifacts=artifacts,
            concepts=concepts,
        )
        if artifact_id is None or concept_id is None:
            continue
        yield artifact_id, concept_id


def _normalize_endpoints(
    edge_type: EdgeType,
    src_id: int,
    dst_id: int,
    *,
    artifacts: Mapping[int, object],
    concepts: Mapping[int, object],
) -> tuple[int | None, int | None]:
    """Identify which endpoint is the artifact and which is the concept,
    using the existence of the id in each table as the discriminator.

    DEFINES/REFERENCES are artifact→concept; IS_NAMED_IN is concept→artifact.
    Phase 1's contract tests occasionally use concept ids as artifact
    endpoints (a documented test-only quirk), so prefer membership checks
    over relying on edge-type direction alone.
    """
    src_is_artifact = src_id in artifacts
    dst_is_artifact = dst_id in artifacts
    src_is_concept = src_id in concepts
    dst_is_concept = dst_id in concepts

    if edge_type == EdgeType.IS_NAMED_IN:
        if src_is_concept and dst_is_artifact:
            return dst_id, src_id
        if dst_is_concept and src_is_artifact:
            return src_id, dst_id
        return None, None

    # DEFINES / REFERENCES: src is artifact, dst is concept.
    if src_is_artifact and dst_is_concept:
        return src_id, dst_id
    if dst_is_artifact and src_is_concept:
        return dst_id, src_id
    return None, None


def compute(store: GraphStore) -> IdfScores:
    """Compute IDF scores from artifact-touching edges currently in the store.

    Uses smoothed IDF: ``log((N + 1) / (df + 1)) + 1`` so that concepts that
    appear in every artifact still get a small (non-zero) score and so an
    empty corpus is well-defined. The ``+ 1`` smoothing matches scikit-learn's
    convention and avoids the divide-by-zero / log-of-zero edge cases.
    """
    document_frequency: dict[int, set[int]] = {}
    artifact_ids: set[int] = set()

    for artifact_id, concept_id in _iter_artifact_concept_pairs(store):
        artifact_ids.add(artifact_id)
        document_frequency.setdefault(concept_id, set()).add(artifact_id)

    n_artifacts = len(artifact_ids)
    smoothing_n = n_artifacts + 1
    scores: dict[int, float] = {}
    for concept_id, artifact_set in document_frequency.items():
        df = len(artifact_set)
        scores[concept_id] = math.log(smoothing_n / (df + 1)) + 1.0

    return IdfScores(
        scores=scores,
        total_artifacts=n_artifacts,
        computed_at=datetime.now(timezone.utc),
    )


def compute_and_persist(store: GraphStore) -> datetime:
    """Compute IDF scores and store them in the module-level sidecar cache.

    Parallels ``centrality.compute_and_persist``; called by ``builder.build``
    once per rebuild after artifacts and edges are persisted. Returns the
    computed-at timestamp so the builder can record it in its
    ``BuildResult``.
    """
    scores = compute(store)
    _IDF_CACHE[store] = scores
    log.info(
        'idf computed: concepts=%d total_artifacts=%d',
        len(scores.scores),
        scores.total_artifacts,
    )
    return scores.computed_at


def get_cached(store: GraphStore) -> IdfScores | None:
    """Return cached IDF scores for ``store``, or ``None`` if absent."""
    return _IDF_CACHE.get(store)


def clear_cache(store: GraphStore | None = None) -> None:
    """Evict cached scores. ``store=None`` clears the entire cache."""
    if store is None:
        _IDF_CACHE.clear()
        return
    _IDF_CACHE.pop(store, None)
