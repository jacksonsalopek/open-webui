"""PageRank centrality precompute for the concept knowledge graph.

Computes two named centrality variants over filtered edge subgraphs and
caches results in a module-level sidecar (Phase 1 v1). The graph store
Protocol exposes ``pagerank(edge_types=...)`` as the substrate seam; this
module owns score persistence until Phase 2 adds dedicated schema fields.

``semantic`` PageRank walks ``CO_OCCURS_WITH`` only — surfaces concepts
densely co-mentioned in code ("what is this codebase about").

``structural`` PageRank walks ``DEFINES`` ∪ ``REFERENCES`` — surfaces
concepts that are heavily defined or referenced ("load-bearing types").
Note: current ``InMemoryGraphStore.pagerank`` only traverses concept→concept
edges; artifact→concept DEFINES/REFERENCES inbound signal is uniform on
that backend until a store-level projection lands. Kuzu loads the same
adjacency shape. Retrieval (step 8) should prefer ``semantic`` for ranking
boosts; ``structural`` is still useful on graphs with concept-concept
paths in those edge types.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from open_webui.retrieval.concepts.schema import EdgeType
from open_webui.retrieval.concepts.store.protocol import GraphStore

log = logging.getLogger(__name__)

_CENTRALITY_CACHE: weakref.WeakKeyDictionary[GraphStore, CentralityScores] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True, slots=True)
class CentralityScores:
    """Two named centrality scores per concept.

    ``semantic`` is PageRank over ``CO_OCCURS_WITH`` only — surfaces concepts
    that are densely co-mentioned in code (the "what is this codebase
    about" axis).

    ``structural`` is PageRank over ``DEFINES`` ∪ ``REFERENCES`` — surfaces
    concepts that are heavily-defined or heavily-referenced (the "what
    are the load-bearing types" axis).

    Both are normalized to sum to 1.0 within their respective edge-type
    subgraph. They are NOT directly comparable across types — use them
    independently for ranking.
    """

    semantic: Mapping[int, float]
    structural: Mapping[int, float]
    computed_at: datetime
    damping: float
    iterations: int


def _normalize(scores: Mapping[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    total = sum(scores.values())
    if total <= 0.0:
        n = len(scores)
        return {node_id: 1.0 / n for node_id in scores}
    return {node_id: value / total for node_id, value in scores.items()}


def compute(
    store: GraphStore,
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> CentralityScores:
    """Compute both centrality variants by calling ``store.pagerank`` twice
    with different ``edge_types`` filters. Pure read; no store mutation."""
    semantic_raw = store.pagerank(
        edge_types=[EdgeType.CO_OCCURS_WITH],
        damping=damping,
        iterations=iterations,
    )
    structural_raw = store.pagerank(
        edge_types=[EdgeType.DEFINES, EdgeType.REFERENCES],
        damping=damping,
        iterations=iterations,
    )
    computed_at = datetime.now(timezone.utc)
    return CentralityScores(
        semantic=_normalize(semantic_raw),
        structural=_normalize(structural_raw),
        computed_at=computed_at,
        damping=damping,
        iterations=iterations,
    )


def compute_and_persist(
    store: GraphStore,
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> datetime:
    """Compute centrality and persist scores in the module sidecar cache.

    Phase 1 v1 does NOT write back to ``Concept.centrality_score`` — the
    sidecar ``WeakKeyDictionary`` decouples centrality from schema/store
    columns. Step 8 retrievers call ``get_cached(store)`` for ranking
    boosts. Phase 2 TODO: add ``semantic_centrality`` /
    ``structural_centrality`` fields on ``Concept`` and persist via store.

    Returns the ``computed_at`` timestamp.
    """
    scores = compute(store, damping=damping, iterations=iterations)
    _CENTRALITY_CACHE[store] = scores
    log.info(
        'centrality computed: semantic_nodes=%d structural_nodes=%d damping=%.2f iters=%d',
        len(scores.semantic),
        len(scores.structural),
        damping,
        iterations,
    )
    return scores.computed_at


def get_cached(store: GraphStore) -> CentralityScores | None:
    """Return cached centrality scores for ``store``, or ``None`` if absent."""
    return _CENTRALITY_CACHE.get(store)


def clear_cache(store: GraphStore | None = None) -> None:
    """Evict cached scores. ``store=None`` clears the entire cache."""
    if store is None:
        _CENTRALITY_CACHE.clear()
        return
    _CENTRALITY_CACHE.pop(store, None)
