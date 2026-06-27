"""Tests for the concept-graph langchain retriever adapter."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Sequence

import pytest

from open_webui.retrieval.concepts.integration.retriever_adapter import (
    ConceptGraphRetriever,
    _build_concept_page_content,
    _content_hash,
    _hit_to_document,
)
from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.concepts.schema import Artifact, ArtifactKind, Concept, ConceptKind

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _concept(
    name: str,
    *,
    concept_id: int = 1,
    kind: ConceptKind = ConceptKind.ATOMIC,
    definition: str | None = None,
    original_tokens: tuple[str, ...] | None = None,
) -> Concept:
    tokens = original_tokens if original_tokens is not None else (name,)
    return Concept(
        id=concept_id,
        name=name,
        kind=kind,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=definition,
        language_hint=None,
        original_tokens=tokens,
    )


def _phrase_concept(
    name: str,
    definition: str,
    *,
    concept_id: int = 1,
    original_tokens: tuple[str, ...] | None = None,
) -> Concept:
    return _concept(
        name,
        concept_id=concept_id,
        kind=ConceptKind.PHRASE,
        definition=definition,
        original_tokens=original_tokens,
    )


def _artifact(*, artifact_id: int = 1, path: str = '/src/foo.cs') -> Artifact:
    return Artifact(
        id=artifact_id,
        kind=ArtifactKind.CHUNK,
        path=path,
        chunk_index=0,
        language='csharp',
        byte_start=0,
        byte_end=100,
        last_modified_at=_TS,
    )


def _hit(
    name: str,
    *,
    concept_id: int = 1,
    score: float = 0.9,
    provenance: dict | None = None,
    concept: Concept | None = None,
    original_tokens: tuple[str, ...] | None = None,
) -> RetrievalHit:
    resolved_concept = concept or _concept(
        name,
        concept_id=concept_id,
        original_tokens=original_tokens,
    )
    return RetrievalHit(
        concept=resolved_concept,
        artifact=None,
        score=score,
        provenance=provenance or {'retriever': 'neighborhood'},
    )


class _FakeRouter:
    def __init__(self, hits: Sequence[RetrievalHit] | None = None) -> None:
        self.hits = list(hits or [])
        self.last_query: str | None = None
        self.last_k: int | None = None
        self.raise_error = False

    def __call__(self, query: str, k: int) -> Sequence[RetrievalHit]:
        self.last_query = query
        self.last_k = k
        if self.raise_error:
            raise RuntimeError('router exploded')
        return self.hits


def test_adapter_calls_router_and_returns_documents() -> None:
    fake = _FakeRouter([_hit('toolbar', concept_id=42, score=0.75)])
    adapter = ConceptGraphRetriever(router_retrieve=fake, k=5)

    docs = adapter.invoke('test query')

    assert fake.last_query == 'test query'
    assert len(docs) == 1
    assert docs[0].page_content == 'toolbar'
    assert docs[0].metadata['concept_name'] == 'toolbar'
    assert docs[0].metadata['concept_id'] == 42
    assert docs[0].metadata['score'] == 0.75


def test_adapter_passes_k_to_router() -> None:
    fake = _FakeRouter([])
    adapter = ConceptGraphRetriever(router_retrieve=fake, k=7)

    adapter.invoke('anything')

    assert fake.last_k == 7


def test_adapter_empty_router_result_returns_empty_list() -> None:
    fake = _FakeRouter([])
    adapter = ConceptGraphRetriever(router_retrieve=fake)

    assert adapter.invoke('empty') == []


def test_adapter_swallows_router_exception(caplog: pytest.LogCaptureFixture) -> None:
    fake = _FakeRouter()
    fake.raise_error = True
    adapter = ConceptGraphRetriever(router_retrieve=fake)

    with caplog.at_level('WARNING'):
        result = adapter.invoke('boom')

    assert result == []
    assert any('concept_graph_retriever failed' in record.message for record in caplog.records)


def test_adapter_metadata_has_retriever_tag() -> None:
    fake = _FakeRouter(
        [
            _hit('alpha'),
            _hit('beta', concept_id=2),
        ],
    )
    adapter = ConceptGraphRetriever(router_retrieve=fake)

    docs = adapter.invoke('tags')

    assert len(docs) == 2
    assert all(doc.metadata['retriever'] == 'concept_graph' for doc in docs)


def test_adapter_metadata_score_passes_through() -> None:
    fake = _FakeRouter([_hit('widget', score=0.42)])
    adapter = ConceptGraphRetriever(router_retrieve=fake)

    docs = adapter.invoke('score check')

    assert docs[0].metadata['score'] == 0.42


def test_adapter_metadata_source_falls_back_to_name() -> None:
    fake = _FakeRouter([_hit('selection')])
    adapter = ConceptGraphRetriever(router_retrieve=fake)

    docs = adapter.invoke('source fallback')

    assert docs[0].metadata['source'] == 'selection'


def test_metadata_includes_chunk_hash() -> None:
    fake = _FakeRouter([_hit('toolbar', concept_id=42, score=0.75)])
    adapter = ConceptGraphRetriever(router_retrieve=fake, k=5)

    docs = adapter.invoke('test query')

    assert len(docs) == 1
    chunk_hash = docs[0].metadata['_chunk_hash']
    assert isinstance(chunk_hash, str)
    assert len(chunk_hash) == 64
    assert all(c in '0123456789abcdef' for c in chunk_hash)


def test_chunk_hash_matches_retrieval_utils_content_hash() -> None:
    from open_webui.retrieval.concepts.integration.retriever_adapter import (
        _content_hash as adapter_content_hash,
    )
    from open_webui.retrieval.utils import _content_hash as utils_content_hash

    sample = 'toolbar concept text'
    assert adapter_content_hash(sample) == utils_content_hash(sample)


def test_chunk_hash_is_deterministic() -> None:
    hit = _hit('toolbar', concept_id=42, score=0.75)
    fake = _FakeRouter([hit])
    adapter = ConceptGraphRetriever(router_retrieve=fake, k=5)

    docs_first = adapter.invoke('test query')
    docs_second = adapter.invoke('test query again')

    assert docs_first[0].metadata['_chunk_hash'] == docs_second[0].metadata['_chunk_hash']


def test_chunk_hash_differs_for_different_concepts() -> None:
    fake = _FakeRouter(
        [
            _hit('alpha', concept_id=1),
            _hit('beta', concept_id=2),
        ],
    )
    adapter = ConceptGraphRetriever(router_retrieve=fake)

    docs = adapter.invoke('different concepts')

    assert docs[0].metadata['_chunk_hash'] != docs[1].metadata['_chunk_hash']


def test_page_content_concept_name_only_when_minimal() -> None:
    hit = _hit('selection', original_tokens=('selection',))
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.page_content == 'selection'


def test_page_content_includes_definition_for_phrase() -> None:
    concept = _phrase_concept(
        'pop-up toolbar',
        'a small floating UI panel that appears near the cursor',
        original_tokens=('pop', 'up', 'toolbar'),
    )
    hit = RetrievalHit(
        concept=concept,
        artifact=None,
        score=0.9,
        provenance={'retriever': 'neighborhood'},
    )
    doc = _hit_to_document(hit, collection_name=None)

    assert (
        doc.page_content
        == 'pop-up toolbar: a small floating UI panel that appears near the cursor'
        ' [tokens: pop, up, toolbar]'
    )


def test_page_content_includes_original_tokens_when_distinct() -> None:
    hit = _hit('chatgpt', original_tokens=('chat', 'gpt'))
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.page_content.endswith(' [tokens: chat, gpt]')


def test_page_content_skips_tokens_matching_name() -> None:
    hit = _hit('selection', original_tokens=('selection',))
    doc = _hit_to_document(hit, collection_name=None)

    assert '[tokens:' not in doc.page_content


def test_page_content_caps_tokens_at_six() -> None:
    tokens = tuple(f'tok{i}' for i in range(10))
    hit = _hit('widget', original_tokens=tokens)
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.page_content.endswith(
        ' [tokens: tok0, tok1, tok2, tok3, tok4, tok5]'
    )


def test_page_content_appends_basename_from_provenance() -> None:
    hit = _hit(
        'clipboard',
        provenance={
            'retriever': 'neighborhood',
            'artifact_path': '/tmp/foo/SelectionService.cs',
        },
    )
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.page_content.endswith(' (in: SelectionService.cs)')


def test_page_content_uses_named_in_paths_list() -> None:
    hit = _hit(
        'clipboard',
        provenance={
            'retriever': 'neighborhood',
            'named_in_paths': ['/a/Foo.cs', '/b/Bar.cs'],
        },
    )
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.page_content.endswith(' (in: Foo.cs)')


def test_page_content_truncates_at_400_chars() -> None:
    long_definition = 'x' * 500
    concept = _phrase_concept('longphrase', long_definition)
    hit = RetrievalHit(
        concept=concept,
        artifact=None,
        score=0.9,
        provenance={'retriever': 'neighborhood'},
    )
    doc = _hit_to_document(hit, collection_name=None)

    assert len(doc.page_content) == 400
    assert doc.page_content.endswith('…')


@pytest.mark.parametrize(
    'name',
    ['alpha', 'toolbar', 'selection', 'clipboard', 'chatgpt'],
)
def test_page_content_starts_with_concept_name(name: str) -> None:
    hit = _hit(name)
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.page_content[:len(name)] == name


def test_page_content_artifact_hit_unchanged() -> None:
    artifact = _artifact(path='/src/SelectionService.cs')
    hit = RetrievalHit(
        concept=None,
        artifact=artifact,
        score=0.8,
        provenance={'retriever': 'hybrid'},
    )
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.page_content == '/src/SelectionService.cs'


def test_chunk_hash_matches_enriched_page_content() -> None:
    hit = _hit(
        'clipboard',
        provenance={
            'retriever': 'neighborhood',
            'artifact_path': '/tmp/foo/SelectionService.cs',
        },
    )
    doc = _hit_to_document(hit, collection_name=None)

    expected = hashlib.sha256(doc.page_content.encode()).hexdigest()
    assert doc.metadata['_chunk_hash'] == expected
    assert doc.metadata['_chunk_hash'] == _content_hash(doc.page_content)


def test_concept_name_metadata_stays_unenriched() -> None:
    concept = _phrase_concept(
        'pop-up toolbar',
        'a small floating UI panel',
        original_tokens=('pop', 'up', 'toolbar'),
    )
    hit = RetrievalHit(
        concept=concept,
        artifact=None,
        score=0.9,
        provenance={'retriever': 'neighborhood'},
    )
    doc = _hit_to_document(hit, collection_name=None)

    assert doc.metadata['concept_name'] == 'pop-up toolbar'
    assert doc.page_content != doc.metadata['concept_name']
