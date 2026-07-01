"""Tests for the concept-graph retrieve hook in query_doc_with_hybrid_search.

Verifies that the optional concept_graph_store + concept_graph_weight params
correctly add the ConceptGraphRetriever as a third EnsembleRetriever member
when enabled, and degrade gracefully when not."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

if not os.environ.get('WEBUI_SECRET_KEY'):
    os.environ['WEBUI_SECRET_KEY'] = 'pytest-concept-graph-integration'

import pytest

from open_webui.config import CONCEPT_GRAPH_ENABLED
from open_webui.retrieval.concepts.integration.retriever_adapter import (
    ConceptGraphRetriever,
)
from open_webui.retrieval.concepts.retrieve.router import (
    ClassifiedIntent,
    Intent,
    RouterResult,
)
from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.utils import (
    query_collection_with_hybrid_search,
    query_doc_with_hybrid_search,
)


def _collection_result() -> SimpleNamespace:
    return SimpleNamespace(
        documents=[['doc1 text', 'doc2 text']],
        metadatas=[[{'source': 'a.cs'}, {'source': 'b.cs'}]],
    )


async def _fake_embedding_function(query, prefix=None, user=None):
    return [0.0] * 8


class _EnsembleCapture:
    """Records EnsembleRetriever construction for assertions."""

    instances: list[_EnsembleCapture] = []

    def __init__(self, *, retrievers=None, weights=None, id_key=None, **kwargs):
        self.retrievers = retrievers or []
        self.weights = weights or []
        self.id_key = id_key
        _EnsembleCapture.instances.append(self)


class _FakeCompressionRetriever:
    def __init__(self, base_compressor, base_retriever):
        self.base_retriever = base_retriever

    async def ainvoke(self, query):
        return []


def _empty_router_result() -> RouterResult:
    return RouterResult(
        intent=ClassifiedIntent(
            intent=Intent.EXPLAIN_REGION,
            extracted_symbols=(),
            extracted_phrases=(),
            raw_text='test',
            classifier_provenance={},
        ),
        hits=[],
        retriever_used='neighborhood',
        elapsed_ms=1,
    )


def _sample_hits() -> list[RetrievalHit]:
    from datetime import datetime, timezone

    from open_webui.retrieval.concepts.schema import Concept, ConceptKind

    ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def _concept(name: str, concept_id: int) -> Concept:
        return Concept(
            id=concept_id,
            name=name,
            kind=ConceptKind.ATOMIC,
            first_seen_at=ts,
            last_seen_at=ts,
            centrality_score=None,
            embedding=(0.1 * concept_id, 0.2 * concept_id),
            definition=None,
            language_hint=None,
            original_tokens=(name,),
        )

    return [
        RetrievalHit(
            concept=_concept('alpha', 1),
            artifact=None,
            score=0.9,
            provenance={},
        ),
        RetrievalHit(
            concept=_concept('beta', 2),
            artifact=None,
            score=0.8,
            provenance={},
        ),
    ]


def _router_result_with_hits() -> RouterResult:
    return RouterResult(
        intent=ClassifiedIntent(
            intent=Intent.EXPLAIN_REGION,
            extracted_symbols=(),
            extracted_phrases=(),
            raw_text='test',
            classifier_provenance={},
        ),
        hits=_sample_hits(),
        retriever_used='neighborhood',
        elapsed_ms=1,
    )


def _run_doc_search(**overrides):
    defaults = {
        'collection_name': 'test-collection',
        'collection_result': _collection_result(),
        'query': 'test query',
        'embedding_function': _fake_embedding_function,
        'k': 5,
        'reranking_function': None,
        'k_reranker': 5,
        'r': 0.0,
        'hybrid_bm25_weight': 0.5,
    }
    defaults.update(overrides)
    return asyncio.run(query_doc_with_hybrid_search(**defaults))


@pytest.fixture
def hybrid_patches():
    """Patch langchain retriever stack so query_doc_with_hybrid_search is unit-testable."""
    _EnsembleCapture.instances.clear()
    bm25_mock = MagicMock(name='bm25_retriever')
    vector_mock = MagicMock(name='vector_retriever')

    with (
        patch(
            'open_webui.retrieval.utils.EnsembleRetriever',
            _EnsembleCapture,
        ),
        patch(
            'open_webui.retrieval.utils.ContextualCompressionRetriever',
            _FakeCompressionRetriever,
        ),
        patch(
            'open_webui.retrieval.utils.BM25Retriever.from_texts',
            return_value=bm25_mock,
        ),
        patch(
            'open_webui.retrieval.utils.VectorSearchRetriever',
            return_value=vector_mock,
        ),
    ):
        yield bm25_mock, vector_mock


def test_no_op_when_store_is_none(hybrid_patches) -> None:
    with patch(
        'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
    ) as mock_cgr:
        result = _run_doc_search(concept_graph_store=None)

    assert result == {'distances': [[]], 'documents': [[]], 'metadatas': [[]]}
    assert len(_EnsembleCapture.instances) == 1
    assert len(_EnsembleCapture.instances[0].retrievers) == 2
    mock_cgr.assert_not_called()


def test_no_op_when_disabled(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = False
    try:
        with patch(
            'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
        ) as mock_cgr:
            result = _run_doc_search(concept_graph_store=MagicMock())

        assert result == {'distances': [[]], 'documents': [[]], 'metadatas': [[]]}
        assert len(_EnsembleCapture.instances) == 1
        assert len(_EnsembleCapture.instances[0].retrievers) == 2
        mock_cgr.assert_not_called()
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_adds_third_member_when_enabled_and_store_present(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    try:
        with patch(
            'open_webui.retrieval.concepts.retrieve.router.route',
            return_value=_empty_router_result(),
        ):
            result = _run_doc_search(concept_graph_store=MagicMock())

        assert result == {'distances': [[]], 'documents': [[]], 'metadatas': [[]]}
        assert len(_EnsembleCapture.instances) == 2
        final = _EnsembleCapture.instances[-1]
        assert len(final.retrievers) == 3
        assert isinstance(final.retrievers[-1], ConceptGraphRetriever)
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_weight_balancing(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    try:
        with patch(
            'open_webui.retrieval.concepts.retrieve.router.route',
            return_value=_empty_router_result(),
        ):
            _run_doc_search(
                concept_graph_store=MagicMock(),
                concept_graph_weight=0.3,
                hybrid_bm25_weight=0.5,
            )

        final = _EnsembleCapture.instances[-1]
        assert len(final.weights) == 3
        assert final.weights[0] == pytest.approx(0.35)
        assert final.weights[1] == pytest.approx(0.35)
        assert final.weights[2] == pytest.approx(0.3)
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_failure_falls_back_to_existing_ensemble(hybrid_patches, caplog) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    try:
        with (
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=RuntimeError('router exploded'),
            ),
            caplog.at_level('ERROR'),
        ):
            result = _run_doc_search(concept_graph_store=MagicMock())

        assert result == {'distances': [[]], 'documents': [[]], 'metadatas': [[]]}
        assert len(_EnsembleCapture.instances) == 1
        assert len(_EnsembleCapture.instances[0].retrievers) == 2
        assert any(
            'failed to add concept_graph member' in record.message
            for record in caplog.records
        )
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_collection_search_passes_through() -> None:
    store = MagicMock()
    weight = 0.3

    async def _run():
        with (
            patch(
                'open_webui.retrieval.utils.query_doc_with_hybrid_search',
                new_callable=AsyncMock,
                return_value={'documents': [[]], 'metadatas': [[]], 'distances': [[]]},
            ) as mock_qd,
            patch(
                'open_webui.retrieval.utils.ASYNC_VECTOR_DB_CLIENT.get',
                new_callable=AsyncMock,
                return_value=_collection_result(),
            ),
        ):
            await query_collection_with_hybrid_search(
                collection_names=['test-collection'],
                queries=['test query'],
                embedding_function=_fake_embedding_function,
                k=5,
                reranking_function=None,
                k_reranker=5,
                r=0.0,
                hybrid_bm25_weight=0.5,
                concept_graph_store=store,
                concept_graph_weight=weight,
            )

            mock_qd.assert_called_once()
            call_kwargs = mock_qd.call_args.kwargs
            assert call_kwargs['concept_graph_store'] is store
            assert call_kwargs['concept_graph_weight'] == weight

    asyncio.run(_run())


def test_concept_graph_embed_fn_threaded_to_router_config(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    captured_kwargs: list[dict] = []

    def _capture_router_config(**kwargs):
        captured_kwargs.append(kwargs)
        return MagicMock()

    def my_fn(text: str) -> tuple[float, ...]:
        return (0.1, 0.2)

    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.RouterConfig',
                side_effect=_capture_router_config,
            ),
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
        ):
            _run_doc_search(
                concept_graph_store=MagicMock(),
                concept_graph_embed_fn=my_fn,
            )

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]['embed_fn'] is my_fn
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_concept_graph_embed_fn_default_none(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    captured_kwargs: list[dict] = []

    def _capture_router_config(**kwargs):
        captured_kwargs.append(kwargs)
        return MagicMock()

    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.RouterConfig',
                side_effect=_capture_router_config,
            ),
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
        ):
            _run_doc_search(concept_graph_store=MagicMock())

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0].get('embed_fn') is None
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_concept_graph_reranker_threads_through(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    reranker_calls: list[tuple[str, list]] = []
    captured_retrieve: list = []

    def _fake_reranker(query: str, hits: list) -> list:
        reranker_calls.append((query, list(hits)))
        return list(reversed(hits))

    def _capture_cgr(**kwargs):
        captured_retrieve.append(kwargs['router_retrieve'])
        return MagicMock()

    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_router_result_with_hits(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture_cgr,
            ),
        ):
            _run_doc_search(
                concept_graph_store=MagicMock(),
                concept_graph_reranker=_fake_reranker,
                query='find selection service',
            )

        assert len(captured_retrieve) == 1
        hits = captured_retrieve[0]('find selection service', 20)
        assert len(reranker_calls) == 1
        assert reranker_calls[0][0] == 'find selection service'
        assert len(reranker_calls[0][1]) == 2
        assert reranker_calls[0][1][0].concept.name == 'alpha'
        assert hits[0].concept.name == 'beta'
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_concept_graph_reranker_failure_does_not_break_retrieve(
    hybrid_patches,
    caplog,
) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    captured_retrieve: list = []

    def _exploding_reranker(query: str, hits: list) -> list:
        raise RuntimeError('reranker exploded')

    def _capture_cgr(**kwargs):
        captured_retrieve.append(kwargs['router_retrieve'])
        return MagicMock()

    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_router_result_with_hits(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture_cgr,
            ),
            caplog.at_level('WARNING'),
        ):
            result = _run_doc_search(
                concept_graph_store=MagicMock(),
                concept_graph_reranker=_exploding_reranker,
            )

        assert result == {'distances': [[]], 'documents': [[]], 'metadatas': [[]]}
        assert len(_EnsembleCapture.instances) == 2
        assert len(_EnsembleCapture.instances[-1].retrievers) == 3

        hits = captured_retrieve[0]('test query', 20)
        assert len(hits) == 2
        assert hits[0].concept.name == 'alpha'
        assert any(
            'concept_graph_reranker failed' in record.message
            for record in caplog.records
        )
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_concept_graph_tiebreaker_threads_through(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    captured_kwargs: list[dict] = []

    def _capture_router_config(**kwargs):
        captured_kwargs.append(kwargs)
        return MagicMock()

    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.RouterConfig',
                side_effect=_capture_router_config,
            ),
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
        ):
            _run_doc_search(
                concept_graph_store=MagicMock(),
                concept_graph_tiebreaker='catrag',
                concept_graph_catrag_alpha=0.2,
            )

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]['tiebreaker'] == 'catrag'
        assert captured_kwargs[0]['catrag_anchor_alpha'] == 0.2
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def _capture_cgr_kwargs():
    captured: list[dict] = []

    def _capture(**kwargs):
        captured.append(kwargs)
        return MagicMock()

    return _capture, captured


def test_concept_graph_retriever_receives_chunk_lookup(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    _capture, captured = _capture_cgr_kwargs()
    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture,
            ),
        ):
            _run_doc_search(concept_graph_store=MagicMock())

        assert captured[0]['store'] is not None
        assert callable(captured[0]['chunk_lookup'])
        result = captured[0]['chunk_lookup']('a.cs')
        assert len(result) == 1
        assert result[0][0] == 'doc1 text'
        assert result[0][1]['source'] == 'a.cs'
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_chunk_lookup_resolves_by_basename_fallback(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    _capture, captured = _capture_cgr_kwargs()
    custom_result = SimpleNamespace(
        documents=[['chunk text']],
        metadatas=[[{'source': '/abs/path/to/a.cs'}]],
    )
    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture,
            ),
        ):
            _run_doc_search(
                concept_graph_store=MagicMock(),
                collection_result=custom_result,
            )

        result = captured[0]['chunk_lookup']('a.cs')
        assert len(result) == 1
        assert result[0][0] == 'chunk text'
        assert result[0][1]['source'] == '/abs/path/to/a.cs'
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_chunk_lookup_returns_empty_for_no_match(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    _capture, captured = _capture_cgr_kwargs()
    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture,
            ),
        ):
            _run_doc_search(concept_graph_store=MagicMock())

        assert captured[0]['chunk_lookup']('nonexistent.cs') == []
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_chunk_lookup_index_built_from_collection_result_chunks(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    _capture, captured = _capture_cgr_kwargs()
    custom_result = SimpleNamespace(
        documents=[['t1', 't2', 't3']],
        metadatas=[[{'source': 'src1.cs'}, {'source': 'src1.cs'}, {'source': 'src2.cs'}]],
    )
    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture,
            ),
        ):
            _run_doc_search(
                concept_graph_store=MagicMock(),
                collection_result=custom_result,
            )

        chunk_lookup = captured[0]['chunk_lookup']
        assert len(chunk_lookup('src1.cs')) == 2
        assert len(chunk_lookup('src2.cs')) == 1
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_cg_retriever_k_floored_to_20(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    _capture, captured = _capture_cgr_kwargs()
    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture,
            ),
        ):
            _run_doc_search(concept_graph_store=MagicMock(), k=4)

        assert captured[0]['k'] == 20
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_cg_retriever_k_unchanged_when_above_floor(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    _capture, captured = _capture_cgr_kwargs()
    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
            patch(
                'open_webui.retrieval.concepts.integration.retriever_adapter.ConceptGraphRetriever',
                side_effect=_capture,
            ),
        ):
            _run_doc_search(concept_graph_store=MagicMock(), k=25)

        assert captured[0]['k'] == 25
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_concept_graph_embed_alpha_threads_through(hybrid_patches) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    captured_kwargs: list[dict] = []

    def _capture_router_config(**kwargs):
        captured_kwargs.append(kwargs)
        return MagicMock()

    try:
        with (
            patch(
                'open_webui.retrieval.concepts.retrieve.router.RouterConfig',
                side_effect=_capture_router_config,
            ),
            patch(
                'open_webui.retrieval.concepts.retrieve.router.route',
                return_value=_empty_router_result(),
            ),
        ):
            _run_doc_search(
                concept_graph_store=MagicMock(),
                concept_graph_embed_alpha=0.35,
            )

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]['embed_blend_alpha'] == 0.35
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig
