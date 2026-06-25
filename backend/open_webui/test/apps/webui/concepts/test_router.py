"""Tests for ``open_webui.retrieval.concepts.retrieve.router``."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from open_webui.retrieval.concepts.retrieve.base import RetrievalQuery
from open_webui.retrieval.concepts.retrieve.neighborhood import (
    NeighborhoodRetriever,
    NeighborhoodRetrieverConfig,
    SeedFilter,
)
from open_webui.retrieval.concepts.retrieve.router import (
    Intent,
    RouterConfig,
    classify_intent,
    route,
)
from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    CoOccursWithProps,
    Concept,
    ConceptKind,
    DefinesProps,
    EdgeType,
    ReferencesProps,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
from open_webui.retrieval.concepts.store.protocol import GraphStore

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _concept(
    name: str,
    *,
    kind: ConceptKind = ConceptKind.ATOMIC,
    embedding: tuple[float, ...] | None = None,
) -> Concept:
    return Concept(
        id=0,
        name=name,
        kind=kind,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=embedding,
        definition=(
            'A curated phrase concept.'
            if kind == ConceptKind.PHRASE
            else None
        ),
        language_hint=None,
        original_tokens=(name,),
    )


def _artifact(path: str = '/src/Foo.cs') -> Artifact:
    return Artifact(
        id=0,
        kind=ArtifactKind.CHUNK,
        path=path,
        chunk_index=0,
        language='csharp',
        byte_start=0,
        byte_end=100,
        last_modified_at=_TS,
    )


def _upsert(store: GraphStore, concept: Concept) -> int:
    return store.upsert_concept(concept)


def test_classify_find_symbol_via_where_is() -> None:
    result = classify_intent('where is ToolbarViewModel defined?')
    assert result.intent == Intent.FIND_SYMBOL
    assert {'toolbar', 'view', 'model'}.issubset(set(result.extracted_symbols))


def test_classify_where_used_via_callers_of() -> None:
    result = classify_intent('callers of ClipboardManager')
    assert result.intent == Intent.WHERE_USED


def test_classify_explain_region_via_how_does() -> None:
    result = classify_intent('how does the toolbar handle extensions?')
    assert result.intent == Intent.EXPLAIN_REGION


def test_classify_find_concept_via_pattern() -> None:
    result = classify_intent('what is the view-model pattern in this codebase?')
    assert result.intent == Intent.FIND_CONCEPT
    assert 'view-model' in result.extracted_phrases


def test_classify_generate_code_via_write_a() -> None:
    result = classify_intent('write a class that monitors clipboard changes')
    assert result.intent == Intent.GENERATE_CODE


def test_classify_default_to_explain_region() -> None:
    result = classify_intent('tell me stuff')
    assert result.intent == Intent.EXPLAIN_REGION
    assert result.classifier_provenance['rule'] == 'default'


def test_route_find_symbol_dispatches_to_neighborhood_radius_1_with_defines_filter() -> None:
    store = InMemoryGraphStore()
    symbol_id = _upsert(store, _concept('toolbar'))
    neighbor_id = _upsert(store, _concept('clipboard'))
    artifact_id = store.upsert_artifact(_artifact('/src/ToolbarViewModel.cs'))
    store.upsert_edge(
        edge_with_props(
            src_id=artifact_id,
            dst_id=symbol_id,
            props=DefinesProps(count=1),
        ),
    )
    store.upsert_edge(
        edge_with_props(
            src_id=symbol_id,
            dst_id=neighbor_id,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )

    configs: list[NeighborhoodRetrieverConfig] = []
    queries: list[RetrievalQuery] = []
    original_init = NeighborhoodRetriever.__init__
    original_retrieve = NeighborhoodRetriever.retrieve

    def tracking_init(self, config=None):
        configs.append(config or NeighborhoodRetrieverConfig())
        original_init(self, config)

    def tracking_retrieve(self, query, graph_store):
        queries.append(query)
        return original_retrieve(self, query, graph_store)

    with (
        patch.object(NeighborhoodRetriever, '__init__', tracking_init),
        patch.object(NeighborhoodRetriever, 'retrieve', tracking_retrieve),
    ):
        result = route(
            RetrievalQuery(text='where is ToolbarViewModel defined?', top_k=5),
            store,
        )

    assert result.retriever_used == 'neighborhood'
    assert configs[0].radius == 1
    assert configs[0].seed_filter == SeedFilter.NONE
    assert configs[0].edge_types == (EdgeType.DEFINES, EdgeType.IS_NAMED_IN)
    assert symbol_id in queries[0].seed_concept_ids


def test_route_where_used_filters_to_references_and_cooccurrence() -> None:
    store = InMemoryGraphStore()
    symbol_id = _upsert(store, _concept('clipboard'))
    caller_id = _upsert(store, _concept('manager'))
    artifact_id = store.upsert_artifact(_artifact('/src/ClipboardManager.cs'))
    store.upsert_edge(
        edge_with_props(
            src_id=artifact_id,
            dst_id=symbol_id,
            props=ReferencesProps(count=1, positions=(10,)),
        ),
    )
    store.upsert_edge(
        edge_with_props(
            src_id=symbol_id,
            dst_id=caller_id,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )

    queries: list[RetrievalQuery] = []
    original_retrieve = NeighborhoodRetriever.retrieve

    def tracking_retrieve(self, query, graph_store):
        queries.append(query)
        return original_retrieve(self, query, graph_store)

    with patch.object(NeighborhoodRetriever, 'retrieve', tracking_retrieve):
        route(
            RetrievalQuery(text='where is ClipboardManager used?', top_k=5),
            store,
        )

    assert queries[0].edge_types_filter == (
        EdgeType.REFERENCES,
        EdgeType.CO_OCCURS_WITH,
    )


def test_route_explain_region_with_embed_fn_uses_hybrid() -> None:
    store = InMemoryGraphStore()
    _upsert(
        store,
        _concept('toolbar', embedding=(1.0, 0.0, 0.0)),
    )

    def stub_embed(_text: str) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0)

    cfg = RouterConfig(embed_fn=stub_embed)
    result = route(
        RetrievalQuery(text='how does the overall architecture work?', top_k=5),
        store,
        config=cfg,
    )
    assert result.retriever_used == 'hybrid'

    result_fallback = route(
        RetrievalQuery(text='how does the overall architecture work?', top_k=5),
        store,
    )
    assert result_fallback.retriever_used == 'neighborhood'


def test_route_find_concept_resolves_phrase_first() -> None:
    store = InMemoryGraphStore()
    phrase_id = _upsert(store, _concept('view-model', kind=ConceptKind.PHRASE))
    neighbor_id = _upsert(store, _concept('binding'))
    store.upsert_edge(
        edge_with_props(
            src_id=phrase_id,
            dst_id=neighbor_id,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )

    queries: list[RetrievalQuery] = []
    original_retrieve = NeighborhoodRetriever.retrieve

    def tracking_retrieve(self, query, graph_store):
        queries.append(query)
        return original_retrieve(self, query, graph_store)

    with patch.object(NeighborhoodRetriever, 'retrieve', tracking_retrieve):
        result = route(
            RetrievalQuery(text='view-model pattern', top_k=5),
            store,
        )

    assert phrase_id in queries[0].seed_concept_ids
    assert result.intent.intent == Intent.FIND_CONCEPT


def test_route_returns_router_result_with_elapsed_ms() -> None:
    store = InMemoryGraphStore()
    result = route(
        RetrievalQuery(text='how does it work?', top_k=5),
        store,
    )
    assert result.elapsed_ms >= 0


def test_route_no_seeds_resolved_returns_empty_hits() -> None:
    store = InMemoryGraphStore()
    result = route(
        RetrievalQuery(text='where is MissingSymbol defined?', top_k=5),
        store,
    )
    assert result.hits == []


def test_route_provenance_includes_intent_and_retriever() -> None:
    store = InMemoryGraphStore()
    _upsert(store, _concept('toolbar'))
    result = route(
        RetrievalQuery(text='where is ToolbarViewModel defined?', top_k=5),
        store,
    )
    assert result.intent.intent == Intent.FIND_SYMBOL
    assert result.intent.classifier_provenance['rule'] == 1
    assert result.retriever_used == 'neighborhood'


# ---- Wave-7 fixes for q07 classifier and unresolved-token decomposition ----


def test_classify_how_does_with_pattern_word_prefers_explain_region() -> None:
    """q07 regression guard: when ``how does`` / ``how is`` co-occurs with
    the generic ``pattern`` keyword (text pattern, regex pattern, design
    pattern), the classifier must pick EXPLAIN_REGION, not FIND_CONCEPT.

    Glossary phrases still beat explain_region — the order is:
    ``where_used > find_symbol > glossary-find_concept > explain_region >
    keyword-find_concept``."""
    result = classify_intent(
        "how does SelectionService get text when UIA doesn't have a text pattern?",
    )
    assert result.intent == Intent.EXPLAIN_REGION


def test_classify_glossary_phrase_still_wins_find_concept_over_explain() -> None:
    """A glossary phrase is a strong concept signal and beats explain_region
    even with 'how does' present."""
    result = classify_intent(
        'how does the view-model pattern work for the toolbar?',
    )
    assert result.intent == Intent.FIND_CONCEPT
    assert result.classifier_provenance.get('matched_via') == 'glossary'


def test_classify_keyword_only_find_concept_still_classified() -> None:
    """When NO explain_region keyword is present, the classifier should
    still fall back to keyword-only find_concept (so q03-style queries
    don't regress)."""
    result = classify_intent('the design concept used by the dispatcher')
    assert result.intent == Intent.FIND_CONCEPT
    assert result.classifier_provenance.get('matched_via') == 'keyword'


def test_router_decomposes_unresolved_compound_token() -> None:
    """A query token that doesn't match any atomic (``sendinput``) should
    be greedy-prefix-decomposed against the atomic index when the parts
    DO exist (``send`` + ``input``). Replays the q07 fix path."""
    store = InMemoryGraphStore()
    send_id = _upsert(store, _concept('send'))
    input_id = _upsert(store, _concept('input'))

    queries: list[RetrievalQuery] = []
    original_retrieve = NeighborhoodRetriever.retrieve

    def tracking_retrieve(self, query, graph_store):
        queries.append(query)
        return original_retrieve(self, query, graph_store)

    with patch.object(NeighborhoodRetriever, 'retrieve', tracking_retrieve):
        route(
            RetrievalQuery(text='where is sendinput used?', top_k=5),
            store,
        )

    seeds = set(queries[0].seed_concept_ids or ())
    assert send_id in seeds
    assert input_id in seeds


def test_router_decomposition_rejects_arbitrary_substring() -> None:
    """Decomposition must NOT pick up a random middle substring: ``pops``
    should not yield ``ops`` as a seed just because ``ops`` happens to
    exist as an atomic."""
    store = InMemoryGraphStore()
    ops_id = _upsert(store, _concept('ops'))

    queries: list[RetrievalQuery] = []
    original_retrieve = NeighborhoodRetriever.retrieve

    def tracking_retrieve(self, query, graph_store):
        queries.append(query)
        return original_retrieve(self, query, graph_store)

    with patch.object(NeighborhoodRetriever, 'retrieve', tracking_retrieve):
        route(RetrievalQuery(text='where is pops used?', top_k=5), store)

    seeds = set(queries[0].seed_concept_ids or ())
    assert ops_id not in seeds


def test_router_decomposition_can_be_disabled() -> None:
    store = InMemoryGraphStore()
    _upsert(store, _concept('send'))
    _upsert(store, _concept('input'))

    queries: list[RetrievalQuery] = []
    original_retrieve = NeighborhoodRetriever.retrieve

    def tracking_retrieve(self, query, graph_store):
        queries.append(query)
        return original_retrieve(self, query, graph_store)

    cfg = RouterConfig(decompose_unresolved_tokens=False)
    with patch.object(NeighborhoodRetriever, 'retrieve', tracking_retrieve):
        route(
            RetrievalQuery(text='where is sendinput defined?', top_k=5),
            store,
            config=cfg,
        )

    assert not queries[0].seed_concept_ids


def test_router_max_neighbors_override_threads_to_retriever() -> None:
    """``RouterConfig.max_neighbors_per_seed`` should flow into the
    NeighborhoodRetrieverConfig used by every route."""
    store = InMemoryGraphStore()
    _upsert(store, _concept('toolbar'))

    configs: list[NeighborhoodRetrieverConfig] = []
    original_init = NeighborhoodRetriever.__init__

    def tracking_init(self, config=None):
        configs.append(config or NeighborhoodRetrieverConfig())
        original_init(self, config)

    cfg = RouterConfig(max_neighbors_per_seed=400)
    with patch.object(NeighborhoodRetriever, '__init__', tracking_init):
        route(
            RetrievalQuery(text='where is Toolbar defined?', top_k=5),
            store,
            config=cfg,
        )

    assert configs[0].max_neighbors_per_seed == 400
