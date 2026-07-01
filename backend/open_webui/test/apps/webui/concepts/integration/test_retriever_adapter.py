"""Tests for the concept-graph langchain retriever adapter."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Sequence

import pytest

from open_webui.retrieval.concepts.integration.retriever_adapter import (
    ConceptGraphRetriever,
    _build_concept_page_content,
    _chunk_query_score,
    _content_hash,
    _hit_to_document,
)
from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    EdgeType,
)

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


class _FakeStore:
    def __init__(
        self,
        artifacts_by_concept: dict[int, list[Artifact]],
        *,
        raise_on: set[int] | None = None,
    ) -> None:
        self._art = artifacts_by_concept
        self._raise_on = raise_on or set()

    def list_artifacts_for_concept(
        self,
        concept_id: int,
        *,
        edge_types: tuple = (EdgeType.IS_NAMED_IN,),
        limit: int | None = None,
    ) -> list[Artifact]:
        if concept_id in self._raise_on:
            raise KeyError(concept_id)
        arts = self._art.get(concept_id, [])
        return arts if limit is None else arts[:limit]


def _artifact_hit(
    path: str,
    *,
    artifact_id: int = 1,
    score: float = 0.8,
) -> RetrievalHit:
    return RetrievalHit(
        concept=None,
        artifact=_artifact(artifact_id=artifact_id, path=path),
        score=score,
        provenance={'retriever': 'hybrid'},
    )


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


def test_hit_to_document_emits_chunk_text_when_artifact_chunks_available() -> None:
    backup_path = '/src/BackupService.cs'
    store = _FakeStore(
        {
            1: [_artifact(path=backup_path)],
        },
    )
    chunk_text = 'async Task BackupAsync(...)'

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        if path == backup_path:
            return [(chunk_text, {'source': backup_path})]
        return []

    fake = _FakeRouter([_hit('backup', concept_id=1)])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
    )

    docs = adapter.invoke('backup caller')

    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert len(code_docs) == 1
    assert code_docs[0].page_content == chunk_text
    assert code_docs[0].metadata['concept_name'] == 'backup'
    assert code_docs[0].metadata['stream'] == 'code_chunks'


def test_hit_to_document_falls_back_to_concept_name_when_no_chunks_available() -> None:
    """No-store path: bare-atomic neighbor survives as the proxy-gate signal."""
    fake = _FakeRouter([_hit('backup', concept_id=1)])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=None,
    )

    docs = adapter.invoke('backup caller')

    neighbor_docs = [d for d in docs if d.metadata.get('stream') == 'neighbors']
    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert len(neighbor_docs) == 1
    assert neighbor_docs[0].page_content == 'backup'
    assert neighbor_docs[0].metadata['stream'] == 'neighbors'
    assert code_docs == []


def test_concept_neighbors_stream_bounded() -> None:
    hits = [_hit(f'concept{i}', concept_id=i, score=1.0 - i * 0.1) for i in range(1, 6)]
    fake = _FakeRouter(hits)
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        max_concept_neighbors=2,
        store=None,
    )

    docs = adapter.invoke('many concepts')

    assert len(docs) == 2
    assert all(doc.metadata['stream'] == 'neighbors' for doc in docs)
    assert all(doc.page_content.startswith(doc.metadata['concept_name']) for doc in docs)


def test_file_path_stream_bounded_and_deduped() -> None:
    store = _FakeStore(
        {
            1: [
                _artifact(artifact_id=1, path='/a.cs'),
                _artifact(artifact_id=2, path='/b.cs'),
                _artifact(artifact_id=3, path='/c.cs'),
            ],
            2: [
                _artifact(artifact_id=4, path='/b.cs'),
                _artifact(artifact_id=5, path='/d.cs'),
            ],
        },
    )
    fake = _FakeRouter(
        [
            _hit('alpha', concept_id=1),
            _hit('beta', concept_id=2),
        ],
    )
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        max_file_paths=3,
        max_concept_neighbors=0,
    )

    docs = adapter.invoke('file paths')

    file_docs = [d for d in docs if d.metadata.get('stream') == 'file_paths']
    assert len(file_docs) <= 3
    paths = [d.page_content for d in file_docs]
    assert len(paths) == len(set(paths))
    assert all(d.metadata['concept_name'] for d in file_docs)
    assert all(d.metadata['stream'] == 'file_paths' for d in file_docs)


def test_code_chunk_stream_bounded_and_deduped() -> None:
    store = _FakeStore(
        {
            1: [
                _artifact(artifact_id=1, path='/a.cs'),
                _artifact(artifact_id=2, path='/b.cs'),
            ],
            2: [_artifact(artifact_id=3, path='/b.cs')],
        },
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        if path == '/a.cs':
            return [('chunkA', {'source': '/a.cs'})]
        if path == '/b.cs':
            return [('chunkB', {'source': '/b.cs'})]
        return []

    fake = _FakeRouter(
        [
            _hit('alpha', concept_id=1),
            _hit('beta', concept_id=2),
        ],
    )
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
        max_code_chunks=4,
        max_concept_neighbors=0,
        max_file_paths=0,
    )

    docs = adapter.invoke('code chunks')

    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert len(code_docs) <= 4
    hashes = [d.metadata['_chunk_hash'] for d in code_docs]
    assert len(hashes) == len(set(hashes))
    assert all(d.metadata['stream'] == 'code_chunks' for d in code_docs)


def test_three_streams_total_within_k_budget() -> None:
    store = _FakeStore(
        {
            i: [
                _artifact(artifact_id=i * 10 + j, path=f'/file{i}_{j}.cs')
                for j in range(3)
            ]
            for i in range(1, 6)
        },
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        return [(f'chunk for {path}', {'source': path})]

    hits = [_hit(f'concept{i}', concept_id=i, score=1.0 - i * 0.05) for i in range(1, 6)]
    fake = _FakeRouter(hits)
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
    )

    docs = adapter.invoke('saturate streams')

    assert len(docs) <= 14
    by_stream: dict[str, int] = {}
    for doc in docs:
        stream = doc.metadata.get('stream', 'unknown')
        by_stream[stream] = by_stream.get(stream, 0) + 1
    assert by_stream.get('neighbors', 0) <= 3
    assert by_stream.get('file_paths', 0) <= 3
    assert by_stream.get('code_chunks', 0) <= 8


def test_store_none_emits_only_concept_neighbors() -> None:
    fake = _FakeRouter([_hit('toolbar', concept_id=42, score=0.75)])
    adapter = ConceptGraphRetriever(router_retrieve=fake)

    docs = adapter.invoke('test query')

    assert len(docs) == 1
    assert docs[0].metadata['stream'] == 'neighbors'


def test_list_artifacts_failure_skips_file_and_chunk_streams() -> None:
    store = _FakeStore({1: [_artifact(path='/a.cs')]}, raise_on={1})
    phrase = _phrase_concept('backup', 'a backup service', concept_id=1)

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        return [('chunkA', {'source': path})]

    fake = _FakeRouter(
        [
            RetrievalHit(
                concept=phrase,
                artifact=None,
                score=0.9,
                provenance={'retriever': 'neighborhood'},
            ),
        ],
    )
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
    )

    docs = adapter.invoke('backup')

    assert any(d.metadata.get('stream') == 'neighbors' for d in docs)
    assert not any(d.metadata.get('stream') == 'file_paths' for d in docs)
    assert not any(d.metadata.get('stream') == 'code_chunks' for d in docs)


def test_artifact_hit_folded_into_file_path_stream() -> None:
    artifact_path = '/src/SelectionService.cs'
    fake = _FakeRouter([_artifact_hit(artifact_path)])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=_FakeStore({}),
        max_concept_neighbors=0,
    )

    docs = adapter.invoke('artifact hit')

    file_docs = [d for d in docs if d.metadata.get('stream') == 'file_paths']
    assert len(file_docs) == 1
    assert file_docs[0].page_content == artifact_path


def test_chunk_query_score_covers_query_tokens() -> None:
    assert _chunk_query_score('await BackupAsync(filePath)', 'backup service') > 0
    assert _chunk_query_score(
        'using Microsoft.Extensions.DependencyInjection',
        'backup service',
    ) == 0.0


def test_code_chunk_reranked_by_query_relevance() -> None:
    path_a = '/src/BackupService.cs'
    path_b = '/src/ImageProcessor.cs'
    store = _FakeStore(
        {
            1: [
                _artifact(artifact_id=1, path=path_b),
                _artifact(artifact_id=2, path=path_a),
            ],
        },
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        if path == path_a:
            return [('backup service file revert logic', {'source': path_a})]
        if path == path_b:
            return [('image processor compress pipeline', {'source': path_b})]
        return []

    fake = _FakeRouter([_hit('backup', concept_id=1)])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
        max_concept_neighbors=0,
        max_file_paths=0,
        max_code_chunks=1,
    )

    docs = adapter.invoke('who calls the backup service')

    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert len(code_docs) == 1
    assert path_a in code_docs[0].metadata['source']
    assert 'backup service' in code_docs[0].page_content


def test_file_path_reranked_by_query_relevance() -> None:
    path_a = '/src/BackupService.cs'
    path_b = '/src/ImageProcessor.cs'
    store = _FakeStore(
        {
            1: [
                _artifact(artifact_id=1, path=path_b),
                _artifact(artifact_id=2, path=path_a),
            ],
        },
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        if path == path_a:
            return [('backup service file revert logic', {'source': path_a})]
        if path == path_b:
            return [('image processor compress pipeline', {'source': path_b})]
        return []

    fake = _FakeRouter([_hit('backup', concept_id=1)])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
        max_concept_neighbors=0,
        max_code_chunks=0,
        max_file_paths=2,
    )

    docs = adapter.invoke('who calls the backup service')

    file_docs = [d for d in docs if d.metadata.get('stream') == 'file_paths']
    assert len(file_docs) >= 1
    assert file_docs[0].page_content == path_a


def test_q04_backup_caller_prefers_backup_async_over_image_boilerplate() -> None:
    """Regression: q04 — backup at rank 4 must beat image/App.xaml.cs score-0 chunk."""
    image_path = '/tmp/zap_subset/Zap/App.xaml.cs'
    compression_path = '/tmp/zap_subset/Zap.Core/Services/CompressionService.cs'
    revert_path = '/tmp/zap_subset/Zap/ViewModels/MainViewModel.cs'

    store = _FakeStore(
        {
            1248: [_artifact(artifact_id=1, path=image_path)],
            706: [
                _artifact(artifact_id=2, path=revert_path),
                _artifact(artifact_id=3, path=compression_path),
            ],
        },
    )

    backup_async_chunk = (
        'file.BackupFilePath = await _backupService.BackupAsync(file.FilePath, ct);'
    )
    revert_chunk = (
        'await _backupService.RevertAsync(model.BackupFilePath, model.FilePath, ct);'
    )
    boilerplate_chunk = (
        'Host = Microsoft.Extensions.Hosting.Host.CreateDefaultBuilder()'
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        if path == image_path:
            return [(boilerplate_chunk, {'source': image_path})]
        if path == compression_path:
            return [(backup_async_chunk, {'source': compression_path})]
        if path == revert_path:
            return [(revert_chunk, {'source': revert_path})]
        return []

    hits = [
        _hit('image', concept_id=1248, score=2.0),
        _hit('service', concept_id=1832, score=2.0),
        _hit('original', concept_id=1536, score=2.0),
        _hit('backup', concept_id=706, score=2.0),
    ]
    fake = _FakeRouter(hits)
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
        max_concept_neighbors=0,
        max_file_paths=0,
        max_code_chunks=4,
    )

    query = 'who calls the backup service before mutating an original image file?'
    docs = adapter.invoke(query)

    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert code_docs, 'expected at least one code chunk'
    assert compression_path in code_docs[0].metadata['source']
    assert 'BackupAsync' in code_docs[0].page_content
    assert boilerplate_chunk not in [d.page_content for d in code_docs[:1]]


def test_code_chunk_ranking_falls_back_to_edge_weight_when_query_has_no_tokens() -> None:
    path_a = '/src/BackupService.cs'
    path_b = '/src/ImageProcessor.cs'
    store = _FakeStore(
        {
            1: [
                _artifact(artifact_id=1, path=path_b),
                _artifact(artifact_id=2, path=path_a),
            ],
        },
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        if path == path_b:
            return [('edge weight top chunk', {'source': path_b})]
        if path == path_a:
            return [('backup service file revert logic', {'source': path_a})]
        return []

    fake = _FakeRouter([_hit('backup', concept_id=1)])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
        max_concept_neighbors=0,
        max_file_paths=0,
    )

    docs = adapter.invoke('a?!')

    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert len(code_docs) >= 1
    assert code_docs[0].metadata['source'] == path_b


def test_code_chunk_ranking_preserves_bounds_and_dedup() -> None:
    store = _FakeStore(
        {
            i: [
                _artifact(artifact_id=i * 10 + j, path=f'/file{i}_{j}.cs')
                for j in range(5)
            ]
            for i in range(1, 4)
        },
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        return [
            (f'chunk alpha for {path}', {'source': path}),
            (f'chunk beta for {path}', {'source': path}),
        ]

    hits = [_hit(f'concept{i}', concept_id=i, score=1.0 - i * 0.1) for i in range(1, 4)]
    fake = _FakeRouter(hits)
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
        max_code_chunks=4,
        max_concept_neighbors=0,
        max_file_paths=0,
    )

    docs = adapter.invoke('alpha chunk ranking saturation')

    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert len(code_docs) <= 4
    hashes = [d.metadata['_chunk_hash'] for d in code_docs]
    assert len(hashes) == len(set(hashes))
    assert all(d.metadata['stream'] == 'code_chunks' for d in code_docs)


def test_de_token_drops_bare_atomic_neighbors_when_store_present() -> None:
    bare = _hit('backup', concept_id=1)
    phrase = RetrievalHit(
        concept=_phrase_concept('named-pipe', 'a pipe with a name', concept_id=2),
        artifact=None,
        score=0.8,
        provenance={'retriever': 'neighborhood'},
    )
    fake = _FakeRouter([bare, phrase])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=_FakeStore({}),
    )

    docs = adapter.invoke('backup named-pipe')

    neighbor_docs = [d for d in docs if d.metadata.get('stream') == 'neighbors']
    neighbor_names = {d.metadata['concept_name'] for d in neighbor_docs}
    assert 'backup' not in neighbor_names
    assert 'named-pipe' in neighbor_names
    assert all(d.metadata['stream'] == 'neighbors' for d in neighbor_docs)


def test_de_token_off_when_store_none() -> None:
    bare = _hit('backup', concept_id=1)
    phrase = RetrievalHit(
        concept=_phrase_concept('named-pipe', 'a pipe with a name', concept_id=2),
        artifact=None,
        score=0.8,
        provenance={'retriever': 'neighborhood'},
    )
    fake = _FakeRouter([bare, phrase])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=None,
    )

    docs = adapter.invoke('backup named-pipe')

    neighbor_docs = [d for d in docs if d.metadata.get('stream') == 'neighbors']
    neighbor_names = {d.metadata['concept_name'] for d in neighbor_docs}
    assert 'backup' in neighbor_names
    assert 'named-pipe' in neighbor_names
    assert all(d.metadata['stream'] == 'neighbors' for d in neighbor_docs)


def test_code_chunks_funnel_widened_to_8() -> None:
    """Default max_code_chunks=8 lets ranks 5–8 through (old cap was 4)."""
    store = _FakeStore(
        {
            1: [
                _artifact(artifact_id=j, path=f'/file_{j}.cs')
                for j in range(1, 11)
            ],
        },
    )

    def chunk_lookup(path: str) -> list[tuple[str, dict[str, str]]]:
        idx = path.split('_')[1].split('.')[0]
        return [(f'ranked chunk {idx} for query token alpha', {'source': path})]

    fake = _FakeRouter([_hit('alpha', concept_id=1)])
    adapter = ConceptGraphRetriever(
        router_retrieve=fake,
        store=store,
        chunk_lookup=chunk_lookup,
        max_concept_neighbors=0,
        max_file_paths=0,
    )

    docs = adapter.invoke('alpha chunk ranking widen funnel')

    code_docs = [d for d in docs if d.metadata.get('stream') == 'code_chunks']
    assert len(code_docs) <= 8
    assert len(code_docs) == 8
    contents = [d.page_content for d in code_docs]
    for rank in ('5', '6', '7', '8'):
        assert any(f'ranked chunk {rank}' in c for c in contents)
