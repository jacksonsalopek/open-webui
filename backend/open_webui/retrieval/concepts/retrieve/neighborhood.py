"""Graph-walk neighborhood retrieval from seed concepts."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from open_webui.retrieval.concepts.extraction.glossary import Glossary
from open_webui.retrieval.concepts.extraction.identifiers import (
    CSHARP_DEFAULT_RULES,
    tokenize_text,
)
from open_webui.retrieval.concepts.extraction.stopwords import (
    StopwordClass,
    classify,
)
from open_webui.retrieval.concepts.retrieve.base import RetrievalHit, RetrievalQuery
from open_webui.retrieval.concepts.schema import Concept, ConceptKind, EdgeType
from open_webui.retrieval.concepts.store.protocol import GraphStore

log = logging.getLogger(__name__)

_centrality_fallback_warned = False
_idf_fallback_warned = False


class SeedFilter(str, Enum):
    NONE = 'none'
    """No filter; all resolved seeds are used (the wave-5 behavior)."""

    NON_STOPWORD = 'non_stopword'
    """Drop seeds whose underlying name classifies as ENGLISH | CODE | LANGUAGE
    via stopwords.classify(). DEFAULT."""

    CENTRALITY_THRESHOLD = 'centrality_threshold'
    """In addition to NON_STOPWORD, also require the seed's semantic centrality
    to be in the top (1 - seed_centrality_percentile) fraction. Requires
    lifecycle.centrality.get_cached(store) to return non-None; gracefully
    falls back to NON_STOPWORD if centrality is unavailable (with a warning
    logged once per process)."""


@dataclass(frozen=True, slots=True)
class NeighborhoodRetrieverConfig:
    radius: int = 2
    """Hops to walk from each seed."""

    edge_types: tuple[EdgeType, ...] | None = (
        EdgeType.CO_OCCURS_WITH,
        EdgeType.DEFINES,
        EdgeType.REFERENCES,
    )
    """Edge types to follow. Default covers structural + semantic."""

    max_neighbors_per_seed: int = 200
    """Cap on neighbors returned by each ``store.neighborhood()`` call.

    Per the Phase 1.5 determinism contract on ``GraphStore.neighborhood``,
    the cap now drops the lowest-weight neighbors deterministically — not
    arbitrary set-iteration leftovers. 200 remains the default because it's
    the smallest cap that empirically retains the answer concepts on the
    Phase 1 acceptance set; lower caps work fine on sparser graphs."""

    dedupe_by_concept_id: bool = True

    include_seeds_as_hits: bool = True
    """When True, resolved seeds are returned as score=2.0 hits (above any
    1-hop neighbor). The user asked ABOUT these concepts; for find_symbol /
    where_used / find_concept intents the seeds ARE part of the answer, not
    noise to filter. Set False to recover the "show me what's RELATED to X"
    semantic (seeds excluded from results)."""

    seed_filter: SeedFilter = SeedFilter.NON_STOPWORD
    """Determines which resolved seed candidates from query.text are actually used.
    See SeedFilter for semantics."""

    seed_centrality_percentile: float = 0.5
    """When seed_filter=CENTRALITY_THRESHOLD: keep seeds whose semantic centrality
    is in the top (1 - this) fraction of all atomic concepts in the store.
    e.g., 0.5 keeps seeds in the top 50% by centrality. Ignored for other modes.
    Set to 0.0 to keep all seeds (equivalent to no filter on the centrality axis)."""

    seed_language: str | None = None
    """Language hint passed to stopwords.classify() for the NON_STOPWORD filter.
    None means classify with no language context (English+Code only)."""

    filter_stopword_hits: bool = True
    """When True, drop non-seed atomic hits whose name classifies as a stopword
    via ``stopwords.classify()``. Atomic concepts like ``up``, ``called``,
    ``after``, ``without`` survive extraction because PascalCase identifiers
    (``MoveUp``, ``AfterCommit``) emit them — but they're useless as retrieval
    answers and crowd out real concepts when the IDF tiebreaker promotes any
    moderately-rare 1-hop neighbor. Seeds and phrase concepts are never
    dropped by this filter (the caller asked about those by name)."""

    tiebreaker: str = 'ppr'
    """Secondary sort signal for hits with identical walk-distance scores.

    - ``'ppr'`` (DEFAULT, Phase 1.5 P0-3): tiebreak by Personalized PageRank
      computed at retrieval time from the resolved seed set, via
      ``store.personalized_pagerank(seeds, edge_types=...)``. PPR scores
      random-walk proximity to the seeds — a node strongly co-occurring with
      the seed neighborhood beats a globally-popular node that happens to be
      weakly connected. This replaces the wave-7 three-knob ``cent_mult_idf``
      product with one principled signal. See ``CONCEPT_GRAPH_PHASE1.md`` §
      "Phase 1.5 — P0 accuracy closure" P0-3 for the rationale.
    - ``'cent_mult_idf'`` (legacy default, kept for ablation): tiebreak by
      ``semantic_centrality × query_multiplicity × IDF``. The three factors
      pull in complementary directions:

      * ``query_multiplicity`` (how many query seeds reached the hit) is the
        query-relevance signal — concepts touched by many seeds beat concepts
        touched by one.
      * ``IDF`` is the rarity signal — concepts that appear in few artifacts
        beat concepts that appear in every artifact (this is the user-requested
        ``user`` / ``model`` / ``view`` demotion).
      * ``semantic_centrality`` is the topical-importance signal — high-degree
        concepts that connect many subgraphs beat low-degree leaves
        (recovers cases where the answer happens to be a popular concept,
        e.g. ``toolbar`` for the floating-popup question).

      Multiplying all three balances the user's anti-popularity ask
      (``IDF`` term) against the regression risk of pure ``IDF × mult``
      (which structurally favors rare-but-irrelevant nodes over the
      central-and-relevant answer).

    - ``'idf_multiplicity'``: pure ``IDF × query_multiplicity`` (no centrality
      term). Faithfully implements the original user request; preserved for
      experiments and for fixtures where centrality is uninformative.
    - ``'centrality'``: legacy semantic-centrality only (pre-Phase-2). Kept
      for regression comparisons.
    - ``'none'``: no tiebreaker — concept id ordering only."""


class NeighborhoodRetriever:
    name = 'neighborhood'

    def __init__(self, config: NeighborhoodRetrieverConfig | None = None) -> None:
        self.config = config or NeighborhoodRetrieverConfig()

    def retrieve(self, query: RetrievalQuery, store: GraphStore) -> list[RetrievalHit]:
        """Walk neighborhood from each seed; return concept hits ranked
        by walk distance (closer = higher score). If ``query.seed_concept_ids``
        is empty, derive seeds from ``query.text`` via identifier
        tokenization → store concept lookup. If no seeds resolve,
        return ``[]``.

        Explicit ``query.seed_concept_ids`` bypass seed filtering entirely —
        the caller is asserting those exact seeds.

        Hop distance is computed by calling ``store.neighborhood`` iteratively
        for ``radius=1..R`` and taking set differences per ring so score
        ``= 1 / hop_distance`` (Phase 1 approximation; store calls do not
        expose per-node distances directly).
        """
        seeds, explicit_override = _resolve_seeds(query, store)
        centrality_threshold_used: float | None = None

        if not explicit_override and self.config.seed_filter != SeedFilter.NONE:
            seeds, centrality_threshold_used = _filter_seeds(
                seeds,
                store,
                self.config,
            )

        if not seeds:
            if not explicit_override:
                log.debug(
                    'No seeds remain after seed_filter=%s; returning empty hits',
                    self.config.seed_filter.value,
                )
            return []

        edge_types = self.config.edge_types or query.edge_types_filter
        seed_set = set(seeds)
        filter_mode = (
            SeedFilter.NONE.value
            if explicit_override
            else self.config.seed_filter.value
        )

        best_score: dict[int, float] = {}
        best_provenance: dict[int, dict[str, object]] = {}
        query_multiplicity: dict[int, int] = {}

        if self.config.include_seeds_as_hits:
            for seed_id in seeds:
                best_score[seed_id] = 2.0
                query_multiplicity[seed_id] = 1
                best_provenance[seed_id] = {
                    'retriever': self.name,
                    'seed_id': seed_id,
                    'hop_distance': 0,
                    'seed_filter_mode': filter_mode,
                    'centrality_threshold_used': centrality_threshold_used,
                }

        for seed_id in seeds:
            prev_ids: set[int] = set()
            for hop in range(1, self.config.radius + 1):
                ring_concepts = store.neighborhood(
                    seed_id,
                    radius=hop,
                    edge_types=edge_types,
                    limit=self.config.max_neighbors_per_seed,
                )
                ring_ids = {concept.id for concept in ring_concepts}
                new_ids = ring_ids - prev_ids
                prev_ids = ring_ids

                hop_score = 1.0 / hop
                for concept in ring_concepts:
                    if concept.id not in new_ids:
                        continue
                    if concept.id in seed_set:
                        continue
                    query_multiplicity[concept.id] = (
                        query_multiplicity.get(concept.id, 0) + 1
                    )
                    existing = best_score.get(concept.id)
                    if existing is not None and existing >= 2.0:
                        continue
                    if existing is None or hop_score > existing:
                        best_score[concept.id] = hop_score
                        best_provenance[concept.id] = {
                            'retriever': self.name,
                            'seed_id': seed_id,
                            'hop_distance': hop,
                            'seed_filter_mode': filter_mode,
                            'centrality_threshold_used': centrality_threshold_used,
                        }

        concept_by_id: dict[int, Concept] = {}
        for concept_id in best_score:
            concept = store.get_concept(concept_id)
            if concept is not None:
                concept_by_id[concept_id] = concept

        hits: list[RetrievalHit] = []
        for concept_id, score in best_score.items():
            concept = concept_by_id.get(concept_id)
            if concept is None:
                continue
            if query.kind_filter is not None and concept.kind not in query.kind_filter:
                continue
            if (
                self.config.filter_stopword_hits
                and concept_id not in seed_set
                and concept.kind == ConceptKind.ATOMIC
                and classify(concept.name, language=self.config.seed_language)
                != StopwordClass.NOT_STOPWORD
            ):
                continue
            hits.append(
                RetrievalHit(
                    concept=concept,
                    artifact=None,
                    score=score,
                    provenance=best_provenance[concept_id],
                ),
            )

        if self.config.dedupe_by_concept_id:
            # Scores already deduped by concept_id via best_score dict.
            pass

        # Tiebreak when walk-distance scores collide. See
        # ``NeighborhoodRetrieverConfig.tiebreaker`` for mode semantics.
        wants_idf = self.config.tiebreaker in ('idf_multiplicity', 'cent_mult_idf')
        wants_centrality = self.config.tiebreaker in (
            'idf_multiplicity',
            'cent_mult_idf',
            'centrality',
        )
        idf_map = _load_idf_scores(store) if wants_idf else None
        centrality_map = (
            _load_semantic_centrality(store) if wants_centrality else None
        ) or {}
        ppr_map: Mapping[int, float] = (
            _load_ppr_scores(seeds, store, edge_types)
            if self.config.tiebreaker == 'ppr'
            else {}
        )

        # Always thread ``query_multiplicity`` through provenance — even when
        # the tiebreaker doesn't use it, downstream auditors (and the
        # acceptance harness) want to see how many seeds reached each hit.
        for cid in best_score:
            provenance = best_provenance[cid]
            provenance['query_multiplicity'] = query_multiplicity.get(cid, 0)
            if idf_map is not None:
                provenance['idf'] = idf_map.get(cid, 0.0)
            if centrality_map:
                provenance['centrality'] = centrality_map.get(cid, 0.0)
            if ppr_map:
                provenance['ppr'] = ppr_map.get(cid, 0.0)

        def _sort_key(hit: RetrievalHit) -> tuple[float, float, float, int]:
            cid = hit.concept.id if hit.concept else 0
            mult = max(query_multiplicity.get(cid, 0), 1)
            cent = centrality_map.get(cid, 0.0)
            idf = idf_map.get(cid, 0.0) if idf_map is not None else 0.0

            if self.config.tiebreaker == 'ppr':
                primary_tiebreak = -ppr_map.get(cid, 0.0)
                secondary_tiebreak = 0.0
            elif self.config.tiebreaker == 'cent_mult_idf':
                primary_tiebreak = -cent * mult * max(idf, 1e-6)
                secondary_tiebreak = -idf * mult
            elif self.config.tiebreaker == 'idf_multiplicity':
                primary_tiebreak = -idf * mult
                secondary_tiebreak = -cent
            elif self.config.tiebreaker == 'centrality':
                primary_tiebreak = -cent
                secondary_tiebreak = 0.0
            else:  # 'none'
                primary_tiebreak = 0.0
                secondary_tiebreak = 0.0

            return (-hit.score, primary_tiebreak, secondary_tiebreak, cid)

        hits.sort(key=_sort_key)
        return hits[: query.top_k]


def _resolve_seeds(
    query: RetrievalQuery,
    store: GraphStore,
) -> tuple[list[int], bool]:
    """Return ``(seed_ids, explicit_override)``.

    When ``explicit_override`` is True, seed filtering is skipped.
    """
    if query.seed_concept_ids:
        return list(query.seed_concept_ids), True

    seeds: set[int] = set()

    # TODO(step 9): router should pass language so TokenRules match the KB.
    rules = CSHARP_DEFAULT_RULES
    for token in tokenize_text(query.text, rules=rules):
        concept_id = _lookup_concept_id(store, token, ConceptKind.ATOMIC)
        if concept_id is not None:
            seeds.add(concept_id)

    for hit in Glossary.default().match(query.text):
        concept_id = _lookup_concept_id(store, hit.phrase.name, ConceptKind.PHRASE)
        if concept_id is not None:
            seeds.add(concept_id)

    return sorted(seeds), False


def _filter_seeds(
    seed_ids: list[int],
    store: GraphStore,
    config: NeighborhoodRetrieverConfig,
) -> tuple[list[int], float | None]:
    """Apply ``config.seed_filter`` to resolved seed candidates."""
    filtered = _filter_non_stopword(seed_ids, store, config.seed_language)

    if config.seed_filter != SeedFilter.CENTRALITY_THRESHOLD:
        return filtered, None

    if config.seed_centrality_percentile <= 0.0:
        return filtered, None

    centrality_map = _load_semantic_centrality(store)
    if centrality_map is None:
        return filtered, None

    values = sorted(centrality_map.values(), reverse=True)
    if not values:
        return filtered, None

    boundary_index = int(len(values) * config.seed_centrality_percentile)
    if boundary_index >= len(values):
        boundary_index = len(values) - 1
    threshold = values[boundary_index]

    centrality_filtered = [
        seed_id
        for seed_id in filtered
        if centrality_map.get(seed_id, 0.0) >= threshold
    ]
    return centrality_filtered, threshold


def _filter_non_stopword(
    seed_ids: list[int],
    store: GraphStore,
    language: str | None,
) -> list[int]:
    kept: list[int] = []
    for seed_id in seed_ids:
        concept = store.get_concept(seed_id)
        if concept is None:
            continue
        if concept.kind == ConceptKind.PHRASE:
            kept.append(seed_id)
            continue
        if classify(concept.name, language=language) == StopwordClass.NOT_STOPWORD:
            kept.append(seed_id)
    return kept


def _load_semantic_centrality(store: GraphStore) -> dict[int, float] | None:
    global _centrality_fallback_warned
    try:
        from open_webui.retrieval.concepts.lifecycle.centrality import get_cached
    except ImportError:
        if not _centrality_fallback_warned:
            log.warning(
                'lifecycle.centrality unavailable; CENTRALITY_THRESHOLD seed filter '
                'falls back to NON_STOPWORD only.',
            )
            _centrality_fallback_warned = True
        return None

    cached = get_cached(store)
    if cached is None:
        if not _centrality_fallback_warned:
            log.warning(
                'No cached centrality scores for store; CENTRALITY_THRESHOLD seed '
                'filter falls back to NON_STOPWORD only. Run '
                'lifecycle.centrality.compute_and_persist first.',
            )
            _centrality_fallback_warned = True
        return None

    return dict(cached.semantic)


def _load_idf_scores(store: GraphStore) -> dict[int, float] | None:
    """Return cached IDF scores for the store, or compute-on-demand when
    no cache is present. On-demand compute keeps the acceptance harness
    working when the test fixture forgot to call ``compute_and_persist``;
    production usage runs through ``lifecycle.builder`` which always
    populates the cache."""
    global _idf_fallback_warned
    try:
        from open_webui.retrieval.concepts.lifecycle import idf as idf_module
    except ImportError:
        if not _idf_fallback_warned:
            log.warning(
                'lifecycle.idf unavailable; tiebreaker falls back to centrality.',
            )
            _idf_fallback_warned = True
        return None

    cached = idf_module.get_cached(store)
    if cached is not None:
        return dict(cached.scores)

    try:
        cached = idf_module.compute(store)
    except Exception:  # pragma: no cover - defensive
        log.exception('lifecycle.idf.compute failed; falling back to centrality')
        return None

    if not cached.scores:
        return None
    return dict(cached.scores)


def _load_ppr_scores(
    seeds: Sequence[int],
    store: GraphStore,
    edge_types: Sequence[EdgeType] | None,
) -> Mapping[int, float]:
    """Compute PPR scores for the resolved seed set via the store primitive.

    Returns the full PPR distribution keyed by concept_id. Empty seeds or
    a store that can't compute PPR (e.g., contract-mismatch fallback)
    returns an empty mapping; the retriever then falls back to id-asc
    ordering for tiebreaks (equivalent to ``tiebreaker='none'``).
    """
    if not seeds:
        return {}
    try:
        return store.personalized_pagerank(
            seeds,
            edge_types=tuple(edge_types) if edge_types is not None else None,
        )
    except (AttributeError, NotImplementedError) as exc:
        log.warning(
            'personalized_pagerank unavailable on store %r; PPR tiebreaker '
            'falls back to none. Reason: %s',
            type(store).__name__,
            exc,
        )
        return {}


def _lookup_concept_id(
    store: GraphStore,
    name: str,
    kind: ConceptKind,
) -> int | None:
    """Resolve ``(name, kind)`` to a concept id without mutating the store.

    Uses ``GraphStore.find_concept_by_name`` (added in step 9, wave 6) as the
    backend-agnostic primitive. Falls back to a duck-typed ``_by_name_kind``
    index for stores that don't yet implement it (defensive — should not trigger
    on current memory or kuzu backends).
    """
    find = getattr(store, 'find_concept_by_name', None)
    if callable(find):
        result = find(name, kind)
        return int(result) if result is not None else None
    by_name_kind = getattr(store, '_by_name_kind', None)
    if by_name_kind is not None:
        result = by_name_kind.get((name, kind))
        return int(result) if result is not None else None
    return None
