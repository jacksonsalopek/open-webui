"""Tests for ``open_webui.retrieval.concepts.retrieve.neighborhood``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from open_webui.retrieval.concepts.lifecycle.centrality import clear_cache
from open_webui.retrieval.concepts.retrieve import neighborhood as neighborhood_module
from open_webui.retrieval.concepts.retrieve.base import RetrievalQuery
from open_webui.retrieval.concepts.retrieve.neighborhood import (
    NeighborhoodRetriever,
    NeighborhoodRetrieverConfig,
    SeedFilter,
)
from open_webui.retrieval.concepts.schema import (
    CoOccursWithProps,
    Concept,
    ConceptKind,
    EdgeType,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
from open_webui.retrieval.concepts.store.protocol import GraphStore

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _concept(
    name: str,
    *,
    kind: ConceptKind = ConceptKind.ATOMIC,
) -> Concept:
    return Concept(
        id=0,
        name=name,
        kind=kind,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=(
            'A curated phrase concept.'
            if kind == ConceptKind.PHRASE
            else None
        ),
        language_hint=None,
        original_tokens=(name,),
    )


def _upsert(store: GraphStore, concept: Concept) -> int:
    return store.upsert_concept(concept)


def _link(
    store: GraphStore,
    src_id: int,
    dst_id: int,
    *,
    weight: float = 1.0,
) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=src_id,
            dst_id=dst_id,
            props=CoOccursWithProps(weight=weight, chunk_count=1),
        ),
    )


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def retriever() -> NeighborhoodRetriever:
    return NeighborhoodRetriever()


def test_seeds_from_query_text(store: InMemoryGraphStore, retriever: NeighborhoodRetriever) -> None:
    view_id = _upsert(store, _concept('view'))
    model_id = _upsert(store, _concept('model'))
    toolbar_id = _upsert(store, _concept('toolbar'))
    widget_id = _upsert(store, _concept('widget'))
    _link(store, view_id, model_id)
    _link(store, model_id, toolbar_id)
    _link(store, toolbar_id, widget_id)

    query = RetrievalQuery(text='the view model in the toolbar', top_k=10)
    hits = retriever.retrieve(query, store)

    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert widget_id in hit_ids
    assert view_id in hit_ids
    assert model_id in hit_ids
    assert toolbar_id in hit_ids
    seed_scores = [hit.score for hit in hits if hit.concept and hit.concept.id in {view_id, model_id, toolbar_id}]
    assert all(score == 2.0 for score in seed_scores)


def test_explicit_seeds_used_when_provided(
    store: InMemoryGraphStore,
    retriever: NeighborhoodRetriever,
) -> None:
    alpha = _upsert(store, _concept('alpha'))
    beta = _upsert(store, _concept('beta'))
    gamma = _upsert(store, _concept('gamma'))
    _link(store, alpha, beta)
    _link(store, gamma, beta)

    query = RetrievalQuery(text='gamma unrelated tokens', seed_concept_ids=(alpha,), top_k=10)
    hits = retriever.retrieve(query, store)

    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert beta in hit_ids
    assert gamma not in hit_ids


def test_returns_empty_when_no_seeds_resolve(
    store: InMemoryGraphStore,
    retriever: NeighborhoodRetriever,
) -> None:
    query = RetrievalQuery(text='abc xyz nonexistent', top_k=10)
    assert retriever.retrieve(query, store) == []


def test_excludes_self_from_results(store: InMemoryGraphStore) -> None:
    """With include_seeds_as_hits=False (opt-in), seeds are excluded — the
    "show me what's RELATED to X" semantic."""
    anchor = _upsert(store, _concept('anchor'))
    neighbor = _upsert(store, _concept('neighbor'))
    _link(store, anchor, neighbor)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(radius=1, include_seeds_as_hits=False),
    )
    query = RetrievalQuery(text='', seed_concept_ids=(anchor,), top_k=10)
    hits = retriever.retrieve(query, store)

    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert anchor not in hit_ids
    assert neighbor in hit_ids


def test_includes_seeds_as_hits_by_default(store: InMemoryGraphStore) -> None:
    """Default behavior (post-acceptance-fix): seeds appear in hits with
    score=2.0, above any 1-hop neighbor. The user asked ABOUT these concepts;
    they ARE part of the answer."""
    anchor = _upsert(store, _concept('anchor'))
    neighbor = _upsert(store, _concept('neighbor'))
    _link(store, anchor, neighbor)

    retriever = NeighborhoodRetriever(NeighborhoodRetrieverConfig(radius=1))
    query = RetrievalQuery(text='', seed_concept_ids=(anchor,), top_k=10)
    hits = retriever.retrieve(query, store)

    by_id = {hit.concept.id: hit for hit in hits if hit.concept is not None}
    assert anchor in by_id
    assert neighbor in by_id
    assert by_id[anchor].score == 2.0
    assert by_id[anchor].provenance['hop_distance'] == 0
    assert by_id[neighbor].score == 1.0


def test_score_decreases_with_hop_distance(store: InMemoryGraphStore) -> None:
    a = _upsert(store, _concept('a'))
    b = _upsert(store, _concept('b'))
    c = _upsert(store, _concept('c'))
    _link(store, a, b)
    _link(store, b, c)

    retriever = NeighborhoodRetriever(NeighborhoodRetrieverConfig(radius=2))
    query = RetrievalQuery(text='', seed_concept_ids=(a,), top_k=10)
    hits = retriever.retrieve(query, store)

    score_by_id = {hit.concept.id: hit.score for hit in hits if hit.concept is not None}
    assert score_by_id[a] > score_by_id[b]
    assert score_by_id[b] > score_by_id[c]
    assert hits[0].provenance['hop_distance'] == 0


def test_max_neighbors_per_seed_respected(store: InMemoryGraphStore) -> None:
    seed = _upsert(store, _concept('seed'))
    neighbor_ids: list[int] = []
    for index in range(10):
        neighbor_ids.append(_upsert(store, _concept(f'n{index}')))
        _link(store, seed, neighbor_ids[-1])

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1, max_neighbors_per_seed=3, include_seeds_as_hits=False,
        ),
    )
    query = RetrievalQuery(text='', seed_concept_ids=(seed,), top_k=20)
    hits = retriever.retrieve(query, store)

    assert len(hits) <= 3


def test_kind_filter_post_walk(store: InMemoryGraphStore) -> None:
    seed = _upsert(store, _concept('seed'))
    atomic = _upsert(store, _concept('atomic-neighbor'))
    phrase = _upsert(
        store,
        _concept('race-condition', kind=ConceptKind.PHRASE),
    )
    _link(store, seed, atomic)
    _link(store, seed, phrase)

    retriever = NeighborhoodRetriever(NeighborhoodRetrieverConfig(radius=1))
    query = RetrievalQuery(
        text='',
        seed_concept_ids=(seed,),
        top_k=10,
        kind_filter=(ConceptKind.PHRASE,),
    )
    hits = retriever.retrieve(query, store)

    assert hits
    assert all(hit.concept is not None and hit.concept.kind == ConceptKind.PHRASE for hit in hits)
    assert {hit.concept.id for hit in hits} == {phrase}


def test_dedupe_by_concept_id(store: InMemoryGraphStore) -> None:
    seed_a = _upsert(store, _concept('seed-a'))
    seed_b = _upsert(store, _concept('seed-b'))
    shared = _upsert(store, _concept('shared'))
    _link(store, seed_a, shared)
    _link(store, seed_b, shared)

    retriever = NeighborhoodRetriever(NeighborhoodRetrieverConfig(radius=1))
    query = RetrievalQuery(text='', seed_concept_ids=(seed_a, seed_b), top_k=10)
    hits = retriever.retrieve(query, store)

    shared_hits = [hit for hit in hits if hit.concept and hit.concept.id == shared]
    assert len(shared_hits) == 1
    assert shared_hits[0].score == pytest.approx(1.0)


def test_top_k_truncation(store: InMemoryGraphStore) -> None:
    seed = _upsert(store, _concept('seed'))
    for index in range(8):
        neighbor_id = _upsert(store, _concept(f'neighbor-{index}'))
        _link(store, seed, neighbor_id)

    retriever = NeighborhoodRetriever(NeighborhoodRetrieverConfig(radius=1))
    query = RetrievalQuery(text='', seed_concept_ids=(seed,), top_k=3)
    hits = retriever.retrieve(query, store)

    assert len(hits) == 3


def test_seed_filter_default_is_non_stopword() -> None:
    assert NeighborhoodRetrieverConfig().seed_filter == SeedFilter.NON_STOPWORD


def test_seed_filter_non_stopword_drops_stopword_seeds(
    store: InMemoryGraphStore,
) -> None:
    the_id = _upsert(store, _concept('the'))
    toolbar_id = _upsert(store, _concept('toolbar'))
    the_neighbor = _upsert(store, _concept('the-only-neighbor'))
    toolbar_neighbor = _upsert(store, _concept('toolbar-neighbor'))
    _link(store, the_id, the_neighbor)
    _link(store, toolbar_id, toolbar_neighbor)

    retriever = NeighborhoodRetriever(NeighborhoodRetrieverConfig(radius=1))
    query = RetrievalQuery(text='the toolbar', top_k=10)
    hits = retriever.retrieve(query, store)

    seed_ids = {hit.provenance['seed_id'] for hit in hits}
    assert seed_ids == {toolbar_id}
    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert toolbar_neighbor in hit_ids
    assert the_neighbor not in hit_ids


def test_seed_filter_none_keeps_all(store: InMemoryGraphStore) -> None:
    the_id = _upsert(store, _concept('the'))
    toolbar_id = _upsert(store, _concept('toolbar'))
    the_neighbor = _upsert(store, _concept('the-only-neighbor'))
    toolbar_neighbor = _upsert(store, _concept('toolbar-neighbor'))
    _link(store, the_id, the_neighbor)
    _link(store, toolbar_id, toolbar_neighbor)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(radius=1, seed_filter=SeedFilter.NONE),
    )
    query = RetrievalQuery(text='the toolbar', top_k=10)
    hits = retriever.retrieve(query, store)

    seed_ids = {hit.provenance['seed_id'] for hit in hits}
    assert seed_ids == {the_id, toolbar_id}
    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert the_neighbor in hit_ids
    assert toolbar_neighbor in hit_ids


def test_seed_filter_centrality_threshold_uses_cached_scores(
    store: InMemoryGraphStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha_id = _upsert(store, _concept('alpha'))
    beta_id = _upsert(store, _concept('beta'))
    gamma_id = _upsert(store, _concept('gamma'))
    alpha_neighbor = _upsert(store, _concept('alpha-neighbor'))
    beta_neighbor = _upsert(store, _concept('beta-neighbor'))
    gamma_neighbor = _upsert(store, _concept('gamma-neighbor'))
    _link(store, alpha_id, alpha_neighbor)
    _link(store, beta_id, beta_neighbor)
    _link(store, gamma_id, gamma_neighbor)

    clear_cache(store)
    injected = {
        alpha_id: 0.9,
        beta_id: 0.8,
        gamma_id: 0.1,
    }
    monkeypatch.setattr(
        neighborhood_module,
        '_load_semantic_centrality',
        lambda graph_store: injected if graph_store is store else None,
    )

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1,
            seed_filter=SeedFilter.CENTRALITY_THRESHOLD,
            seed_centrality_percentile=0.5,
        ),
    )
    query = RetrievalQuery(text='alpha beta gamma', top_k=20)
    hits = retriever.retrieve(query, store)

    seed_ids = {hit.provenance['seed_id'] for hit in hits}
    assert seed_ids == {alpha_id, beta_id}
    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert alpha_neighbor in hit_ids
    assert beta_neighbor in hit_ids
    assert gamma_neighbor not in hit_ids
    assert hits[0].provenance['centrality_threshold_used'] == pytest.approx(0.8)


def test_seed_filter_centrality_threshold_falls_back_when_no_cache(
    store: InMemoryGraphStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    the_id = _upsert(store, _concept('the'))
    toolbar_id = _upsert(store, _concept('toolbar'))
    toolbar_neighbor = _upsert(store, _concept('toolbar-neighbor'))
    _link(store, toolbar_id, toolbar_neighbor)

    clear_cache(store)
    neighborhood_module._centrality_fallback_warned = False

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1,
            seed_filter=SeedFilter.CENTRALITY_THRESHOLD,
            seed_centrality_percentile=0.5,
        ),
    )
    with caplog.at_level('WARNING'):
        query = RetrievalQuery(text='the toolbar', top_k=10)
        hits = retriever.retrieve(query, store)

    assert any('falls back to NON_STOPWORD' in record.message for record in caplog.records)
    seed_ids = {hit.provenance['seed_id'] for hit in hits}
    assert seed_ids == {toolbar_id}


def test_explicit_seed_concept_ids_bypass_filter(store: InMemoryGraphStore) -> None:
    the_id = _upsert(store, _concept('the'))
    the_neighbor = _upsert(store, _concept('the-neighbor'))
    _link(store, the_id, the_neighbor)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(radius=1, seed_filter=SeedFilter.NON_STOPWORD),
    )
    query = RetrievalQuery(
        text='ignored',
        seed_concept_ids=(the_id,),
        top_k=10,
    )
    hits = retriever.retrieve(query, store)

    assert hits
    assert hits[0].provenance['seed_id'] == the_id
    assert hits[0].provenance['seed_filter_mode'] == SeedFilter.NONE.value
    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert the_neighbor in hit_ids


def test_provenance_records_filter_mode(store: InMemoryGraphStore) -> None:
    seed = _upsert(store, _concept('seed'))
    neighbor = _upsert(store, _concept('neighbor'))
    _link(store, seed, neighbor)

    retriever = NeighborhoodRetriever(NeighborhoodRetrieverConfig(radius=1))
    query = RetrievalQuery(text='seed', top_k=10)
    hits = retriever.retrieve(query, store)

    assert hits
    assert hits[0].provenance['seed_filter_mode'] == SeedFilter.NON_STOPWORD.value


# ---- Wave-7 tiebreaker behavior ----


def test_query_multiplicity_recorded_in_provenance(
    store: InMemoryGraphStore,
) -> None:
    """The number of seeds that reached a hit should be tracked so the
    cent*mult*idf tiebreaker can use it (and so callers can audit the
    'why this rank?' signal)."""
    seed_a = _upsert(store, _concept('alpha'))
    seed_b = _upsert(store, _concept('beta'))
    common = _upsert(store, _concept('common'))
    only_a = _upsert(store, _concept('only-alpha'))
    _link(store, seed_a, common)
    _link(store, seed_b, common)
    _link(store, seed_a, only_a)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(radius=1, seed_filter=SeedFilter.NONE),
    )
    query = RetrievalQuery(
        text='ignored',
        seed_concept_ids=(seed_a, seed_b),
        top_k=10,
    )
    hits = retriever.retrieve(query, store)
    by_id = {hit.concept.id: hit for hit in hits if hit.concept is not None}

    assert by_id[common].provenance['query_multiplicity'] == 2
    assert by_id[only_a].provenance['query_multiplicity'] == 1


def test_tiebreaker_idf_multiplicity_demotes_popular_concepts(
    store: InMemoryGraphStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure IDF*mult tiebreaker should put a rare-but-multiply-touched
    concept ahead of a common one with the same multiplicity. Pins the
    user's original request shape."""
    seed_a = _upsert(store, _concept('alpha'))
    seed_b = _upsert(store, _concept('beta'))
    rare = _upsert(store, _concept('rare'))
    common = _upsert(store, _concept('common'))
    _link(store, seed_a, rare)
    _link(store, seed_b, rare)
    _link(store, seed_a, common)
    _link(store, seed_b, common)

    monkeypatch.setattr(
        neighborhood_module,
        '_load_idf_scores',
        lambda graph_store: {rare: 6.0, common: 1.0},
    )
    monkeypatch.setattr(
        neighborhood_module,
        '_load_semantic_centrality',
        lambda graph_store: {rare: 0.001, common: 0.5},
    )

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1,
            seed_filter=SeedFilter.NONE,
            include_seeds_as_hits=False,
            tiebreaker='idf_multiplicity',
        ),
    )
    query = RetrievalQuery(
        text='ignored',
        seed_concept_ids=(seed_a, seed_b),
        top_k=10,
    )
    hits = retriever.retrieve(query, store)
    hit_order = [hit.concept.id for hit in hits if hit.concept is not None]
    assert hit_order.index(rare) < hit_order.index(common)


def test_tiebreaker_cent_mult_idf_keeps_central_answer(
    store: InMemoryGraphStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``cent*mult*idf`` should keep a high-centrality answer
    ahead of an extremely-rare but low-centrality noise hit when both
    have the same multiplicity. Pins the q09 regression guard."""
    seed_a = _upsert(store, _concept('alpha'))
    seed_b = _upsert(store, _concept('beta'))
    central = _upsert(store, _concept('central'))
    rare_noise = _upsert(store, _concept('rare-noise'))
    _link(store, seed_a, central)
    _link(store, seed_b, central)
    _link(store, seed_a, rare_noise)
    _link(store, seed_b, rare_noise)

    monkeypatch.setattr(
        neighborhood_module,
        '_load_idf_scores',
        lambda graph_store: {central: 2.0, rare_noise: 6.0},
    )
    monkeypatch.setattr(
        neighborhood_module,
        '_load_semantic_centrality',
        lambda graph_store: {central: 0.05, rare_noise: 0.0001},
    )

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1,
            seed_filter=SeedFilter.NONE,
            include_seeds_as_hits=False,
            tiebreaker='cent_mult_idf',
        ),
    )
    query = RetrievalQuery(
        text='ignored',
        seed_concept_ids=(seed_a, seed_b),
        top_k=10,
    )
    hits = retriever.retrieve(query, store)
    hit_order = [hit.concept.id for hit in hits if hit.concept is not None]
    assert hit_order.index(central) < hit_order.index(rare_noise)


def test_filter_stopword_hits_drops_english_particle_neighbors(
    store: InMemoryGraphStore,
) -> None:
    """Atomic hits that classify as English stopwords (``before``, ``up``,
    ``one``) should be dropped post-walk so the IDF tiebreaker doesn't
    promote them into the answer ranking. Seeds are exempt — even a
    stopword named seed survives because the caller asked for it."""
    seed = _upsert(store, _concept('seed'))
    noise = _upsert(store, _concept('before'))
    answer = _upsert(store, _concept('toolbar'))
    _link(store, seed, noise)
    _link(store, seed, answer)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1,
            seed_filter=SeedFilter.NONE,
            filter_stopword_hits=True,
        ),
    )
    query = RetrievalQuery(
        text='ignored',
        seed_concept_ids=(seed,),
        top_k=10,
    )
    hits = retriever.retrieve(query, store)
    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert noise not in hit_ids
    assert answer in hit_ids


def test_filter_stopword_hits_can_be_disabled(
    store: InMemoryGraphStore,
) -> None:
    seed = _upsert(store, _concept('seed'))
    noise = _upsert(store, _concept('before'))
    _link(store, seed, noise)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1,
            seed_filter=SeedFilter.NONE,
            filter_stopword_hits=False,
        ),
    )
    query = RetrievalQuery(
        text='ignored',
        seed_concept_ids=(seed,),
        top_k=10,
    )
    hits = retriever.retrieve(query, store)
    hit_ids = {hit.concept.id for hit in hits if hit.concept is not None}
    assert noise in hit_ids


# ---- Wave-9 PPR tiebreaker (Phase 1.5 P0-3) ----


def test_ppr_is_default_tiebreaker() -> None:
    assert NeighborhoodRetrieverConfig().tiebreaker == 'ppr'


def test_ppr_tiebreaker_calls_personalized_pagerank() -> None:
    store = InMemoryGraphStore()
    anchor = _upsert(store, _concept('anchor'))
    neighbor = _upsert(store, _concept('neighbor'))
    _link(store, anchor, neighbor)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(radius=1, seed_filter=SeedFilter.NONE),
    )
    query = RetrievalQuery(text='', seed_concept_ids=(anchor,), top_k=10)
    hits = retriever.retrieve(query, store)

    assert hits
    assert all('ppr' in hit.provenance for hit in hits)


def _build_hub_vs_local_graph(store: InMemoryGraphStore) -> dict[str, int]:
    """Graph with a seed-local cluster and a globally-popular hub."""
    ids: dict[str, int] = {}
    for name in ('anchor', 'local_a', 'local_b', 'global_hub', 'unrelated'):
        ids[name] = _upsert(store, _concept(name))
    _link(store, ids['anchor'], ids['local_a'], weight=5.0)
    _link(store, ids['anchor'], ids['global_hub'], weight=1.0)
    _link(store, ids['local_a'], ids['local_b'], weight=5.0)
    _link(store, ids['global_hub'], ids['local_a'])
    _link(store, ids['global_hub'], ids['local_b'])
    _link(store, ids['global_hub'], ids['unrelated'])
    _link(store, ids['global_hub'], ids['anchor'])
    return ids


def test_ppr_tiebreaker_prefers_seed_local_over_global_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryGraphStore()
    ids = _build_hub_vs_local_graph(store)
    query = RetrievalQuery(
        text='',
        seed_concept_ids=(ids['anchor'],),
        top_k=10,
    )
    base_config = dict(
        radius=2,
        seed_filter=SeedFilter.NONE,
        include_seeds_as_hits=False,
    )

    monkeypatch.setattr(
        neighborhood_module,
        '_load_semantic_centrality',
        lambda graph_store: {
            ids['global_hub']: 0.9,
            ids['local_a']: 0.1,
            ids['local_b']: 0.05,
        },
    )

    ppr_hits = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(tiebreaker='ppr', **base_config),
    ).retrieve(query, store)
    cent_hits = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(tiebreaker='centrality', **base_config),
    ).retrieve(query, store)

    def rank_of(hits: list, concept_id: int) -> int:
        return [h.concept.id for h in hits if h.concept].index(concept_id)

    local_a = ids['local_a']
    hub = ids['global_hub']

    ppr_order = [h.concept.id for h in ppr_hits if h.concept]
    assert local_a in ppr_order and hub in ppr_order
    assert rank_of(ppr_hits, local_a) < rank_of(ppr_hits, hub)

    assert rank_of(cent_hits, hub) < rank_of(cent_hits, local_a)


def test_ppr_falls_back_gracefully_when_unsupported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PprUnsupportedStore(InMemoryGraphStore):
        def personalized_pagerank(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AttributeError('personalized_pagerank not implemented')

    store = PprUnsupportedStore()
    low = _upsert(store, _concept('aaa'))
    high = _upsert(store, _concept('zzz'))
    _link(store, low, high)

    retriever = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(
            radius=1,
            seed_filter=SeedFilter.NONE,
            include_seeds_as_hits=False,
            tiebreaker='ppr',
        ),
    )
    query = RetrievalQuery(text='', seed_concept_ids=(low,), top_k=10)
    with caplog.at_level('WARNING'):
        hits = retriever.retrieve(query, store)

    assert any('personalized_pagerank unavailable' in r.message for r in caplog.records)
    tie_hits = [h for h in hits if h.concept and h.score == pytest.approx(1.0)]
    assert [h.concept.id for h in tie_hits] == sorted(h.concept.id for h in tie_hits)


def test_legacy_tiebreakers_still_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryGraphStore()
    seed_a = _upsert(store, _concept('alpha'))
    seed_b = _upsert(store, _concept('beta'))
    central = _upsert(store, _concept('central'))
    rare = _upsert(store, _concept('rare'))
    _link(store, seed_a, central)
    _link(store, seed_b, central)
    _link(store, seed_a, rare)
    _link(store, seed_b, rare)

    monkeypatch.setattr(
        neighborhood_module,
        '_load_idf_scores',
        lambda graph_store: {central: 2.0, rare: 6.0},
    )
    monkeypatch.setattr(
        neighborhood_module,
        '_load_semantic_centrality',
        lambda graph_store: {central: 0.05, rare: 0.0001},
    )

    query = RetrievalQuery(
        text='ignored',
        seed_concept_ids=(seed_a, seed_b),
        top_k=10,
    )
    base = dict(radius=1, seed_filter=SeedFilter.NONE, include_seeds_as_hits=False)

    cent_mult_hits = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(tiebreaker='cent_mult_idf', **base),
    ).retrieve(query, store)
    assert cent_mult_hits
    order = [h.concept.id for h in cent_mult_hits if h.concept]
    assert order.index(central) < order.index(rare)

    idf_mult_hits = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(tiebreaker='idf_multiplicity', **base),
    ).retrieve(query, store)
    assert idf_mult_hits
    order = [h.concept.id for h in idf_mult_hits if h.concept]
    assert order.index(rare) < order.index(central)

    cent_hits = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(tiebreaker='centrality', **base),
    ).retrieve(query, store)
    assert cent_hits
    order = [h.concept.id for h in cent_hits if h.concept]
    assert order.index(central) < order.index(rare)

    none_hits = NeighborhoodRetriever(
        NeighborhoodRetrieverConfig(tiebreaker='none', **base),
    ).retrieve(query, store)
    assert none_hits
    tie_order = [h.concept.id for h in none_hits if h.concept]
    assert tie_order == sorted(tie_order)


def test_ppr_skipped_when_seeds_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def _fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError('_load_ppr_scores should not be called')

    monkeypatch.setattr(neighborhood_module, '_load_ppr_scores', _fail_if_called)

    store = InMemoryGraphStore()
    retriever = NeighborhoodRetriever()
    query = RetrievalQuery(text='nonexistent tokens xyz', top_k=10)
    assert retriever.retrieve(query, store) == []
    assert not called
