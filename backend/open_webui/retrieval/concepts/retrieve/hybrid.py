"""Centrality-boosted hybrid retrieval: vector search + neighborhood + PageRank."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from open_webui.retrieval.concepts.retrieve.base import RetrievalHit, RetrievalQuery
from open_webui.retrieval.concepts.schema import Concept, ConceptKind, EdgeType
from open_webui.retrieval.concepts.store.protocol import GraphStore

log = logging.getLogger(__name__)

_centrality_warned = False


@dataclass(frozen=True, slots=True)
class HybridRetrieverConfig:
    vector_top_k: int = 20
    """How many candidates to fetch from vector_search before re-ranking."""

    neighborhood_radius: int = 1
    """Expand each top-vector hit by 1-hop neighbors as additional candidates."""

    vector_weight: float = 0.6
    """Linear blend weight for the vector_search score component."""

    neighborhood_weight: float = 0.25
    """Linear blend weight for the neighborhood-expansion score component."""

    centrality_weight: float = 0.15
    """Linear blend weight for the centrality-boost component. Set to 0
    to disable centrality coupling (useful if lifecycle.centrality isn't
    available or hasn't been computed)."""

    centrality_kind: Literal['semantic', 'structural'] = 'semantic'
    """Which centrality variant to use as the boost. Per CONCEPT_GRAPH_PHASE1.md
    carry-forward risk #3: semantic centrality (CO_OCCURS_WITH-only) is the
    right ranking lever for free-text 'explain this' queries; structural
    is right for 'show me the load-bearing types' style queries."""

    edge_types_for_neighborhood: tuple[EdgeType, ...] = (EdgeType.CO_OCCURS_WITH,)


class HybridRetriever:
    name = 'hybrid'

    def __init__(self, config: HybridRetrieverConfig | None = None) -> None:
        self.config = config or HybridRetrieverConfig()

    def retrieve(self, query: RetrievalQuery, store: GraphStore) -> list[RetrievalHit]:
        """Three-component ranking:
        1. vector_search produces top-K candidates with cosine scores.
        2. For each top-K hit, 1-hop neighborhood expansion adds graph-adjacent
           candidates with score = 0.5 * source_vector_score.
        3. Optionally boost final scores by centrality[centrality_kind].

        Final score = vector_weight * normalized_vector_score
                    + neighborhood_weight * normalized_neighborhood_score
                    + centrality_weight * normalized_centrality_score

        Normalization: min-max scale to [0, 1] within each component
        across the candidate set, then blend. (Avoids dominance by
        wildly different score ranges.)
        """
        if query.embedding is None:
            raise ValueError(
                'HybridRetriever requires query.embedding; caller must precompute',
            )

        vector_hits = _vector_search(query, store, self.config.vector_top_k)
        vector_scores: dict[int, float] = {
            concept.id: score for concept, score in vector_hits
        }

        neighborhood_scores: dict[int, float] = {}
        for concept, vector_score in vector_hits:
            propagated = 0.5 * vector_score
            neighbors = store.neighborhood(
                concept.id,
                radius=self.config.neighborhood_radius,
                edge_types=self.config.edge_types_for_neighborhood,
                limit=self.config.vector_top_k,
            )
            for neighbor in neighbors:
                existing = neighborhood_scores.get(neighbor.id)
                if existing is None or propagated > existing:
                    neighborhood_scores[neighbor.id] = propagated

        candidate_ids = set(vector_scores) | set(neighborhood_scores)

        centrality_raw: dict[int, float] = {}
        if self.config.centrality_weight > 0:
            centrality_map = _load_centrality_scores_for_config(
                store,
                self.config.centrality_kind,
            )
            if centrality_map is not None:
                for concept_id in candidate_ids:
                    centrality_raw[concept_id] = centrality_map.get(concept_id, 0.0)

        norm_vector = _min_max_normalize(
            {cid: vector_scores.get(cid, 0.0) for cid in candidate_ids},
        )
        norm_neighborhood = _min_max_normalize(
            {cid: neighborhood_scores.get(cid, 0.0) for cid in candidate_ids},
        )
        norm_centrality = _min_max_normalize(
            {cid: centrality_raw.get(cid, 0.0) for cid in candidate_ids},
        )

        ranked: list[tuple[int, float, dict[str, object]]] = []
        for concept_id in sorted(candidate_ids):
            final_score = (
                self.config.vector_weight * norm_vector[concept_id]
                + self.config.neighborhood_weight * norm_neighborhood[concept_id]
                + self.config.centrality_weight * norm_centrality[concept_id]
            )
            sources: list[str] = []
            if concept_id in vector_scores:
                sources.append('vector')
            if concept_id in neighborhood_scores:
                sources.append('neighborhood')

            provenance: dict[str, object] = {
                'retriever': self.name,
                'vector_score': vector_scores.get(concept_id, 0.0),
                'neighborhood_score': neighborhood_scores.get(concept_id, 0.0),
                'centrality_score': centrality_raw.get(concept_id, 0.0),
                'final_score': final_score,
                'sources': sources,
            }
            ranked.append((concept_id, final_score, provenance))

        ranked.sort(key=lambda item: (-item[1], item[0]))

        hits: list[RetrievalHit] = []
        seen: set[int] = set()
        for concept_id, final_score, provenance in ranked:
            if concept_id in seen:
                continue
            seen.add(concept_id)
            concept = store.get_concept(concept_id)
            if concept is None:
                continue
            if query.kind_filter is not None and concept.kind not in query.kind_filter:
                continue
            hits.append(
                RetrievalHit(
                    concept=concept,
                    artifact=None,
                    score=final_score,
                    provenance=provenance,
                ),
            )
            if len(hits) >= query.top_k:
                break

        return hits


def _vector_search(
    query: RetrievalQuery,
    store: GraphStore,
    limit: int,
) -> list[tuple[Concept, float]]:
    if query.kind_filter and len(query.kind_filter) > 1:
        best: dict[int, tuple[Concept, float]] = {}
        for kind in query.kind_filter:
            for concept, score in store.vector_search(
                query.embedding,  # type: ignore[arg-type]
                kind=kind,
                limit=limit,
            ):
                existing = best.get(concept.id)
                if existing is None or score > existing[1]:
                    best[concept.id] = (concept, score)
        merged = sorted(best.values(), key=lambda item: item[1], reverse=True)
        return merged[:limit]

    kind = query.kind_filter[0] if query.kind_filter else None
    return list(
        store.vector_search(
            query.embedding,  # type: ignore[arg-type]
            kind=kind,
            limit=limit,
        ),
    )


def _load_centrality_scores_for_config(
    store: GraphStore,
    centrality_kind: Literal['semantic', 'structural'],
) -> dict[int, float] | None:
    global _centrality_warned
    try:
        from open_webui.retrieval.concepts.lifecycle.centrality import get_cached
    except ImportError:
        if not _centrality_warned:
            log.warning(
                'lifecycle.centrality unavailable; hybrid centrality boost disabled. '
                'Run lifecycle.centrality.compute_and_persist after graph rebuild.',
            )
            _centrality_warned = True
        return None

    cached = get_cached(store)
    if cached is None:
        if not _centrality_warned:
            log.warning(
                'No cached centrality scores for store; hybrid centrality boost '
                'disabled. Run lifecycle.centrality.compute_and_persist first.',
            )
            _centrality_warned = True
        return None

    if centrality_kind == 'semantic':
        return dict(cached.semantic)
    return dict(cached.structural)


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {key: 1.0 if maximum > 0.0 else 0.0 for key in scores}

    span = maximum - minimum
    return {key: (value - minimum) / span for key, value in scores.items()}
