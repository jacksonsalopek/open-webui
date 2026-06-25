"""Tests for ``open_webui.retrieval.concepts.extraction.extractor``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document

from open_webui.retrieval.concepts.extraction.extractor import (
    ExtractionResult,
    ExtractorOptions,
    extract,
    extract_chunks,
)
from open_webui.retrieval.concepts.extraction.glossary import Glossary, PhraseConcept
from open_webui.retrieval.concepts.extraction.identifiers import (
    CSHARP_DEFAULT_RULES,
    tokenize,
)
from open_webui.retrieval.concepts.schema import (
    ConceptKind,
    EdgeType,
    DefinesProps,
    ReferencesProps,
)

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def default_glossary() -> Glossary:
    return Glossary.default()


def _atomic_names(result: ExtractionResult) -> set[str]:
    return {concept.name for concept in result.concepts if concept.kind == ConceptKind.ATOMIC}


def _phrase_names(result: ExtractionResult) -> set[str]:
    return {concept.name for concept in result.concepts if concept.kind == ConceptKind.PHRASE}


def _edges_of_type(result: ExtractionResult, edge_type: EdgeType) -> list:
    return [edge for edge in result.edges if edge.type == edge_type]


def _concept_name_by_id(result: ExtractionResult, concept_id: int) -> str:
    return next(concept.name for concept in result.concepts if concept.id == concept_id)


def _co_occurrence_endpoint_names(result: ExtractionResult) -> set[str]:
    names: set[str] = set()
    for edge in _edges_of_type(result, EdgeType.CO_OCCURS_WITH):
        names.add(_concept_name_by_id(result, edge.src_id))
        names.add(_concept_name_by_id(result, edge.dst_id))
    return names


def test_extract_empty_chunk_yields_only_artifact(default_glossary: Glossary) -> None:
    doc = Document(page_content='', metadata={'source': '/empty.cs'})
    result = extract(doc, glossary=default_glossary, now=_TS)

    assert result.artifact.path == '/empty.cs'
    assert result.concepts == ()
    assert result.edges == ()
    assert result.stats.distinct_atomic_count == 0
    assert result.stats.distinct_phrase_count == 0


def test_extract_csharp_chunk_basic(default_glossary: Glossary) -> None:
    content = '''
/// <summary>
/// Toolbar view-model; watch for race condition during async updates.
/// </summary>
public sealed partial class ToolbarViewModel : ObservableObject
{
    public void ExecuteExtension() { }

    private void CancelActiveRun() { }
}
'''
    doc = Document(
        page_content=content,
        metadata={
            'source': '/ToolbarViewModel.cs',
            'code_split_language': 'csharp',
            'ast_symbol': 'ToolbarViewModel',
        },
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    assert len(_phrase_names(result)) >= 1
    assert 'race-condition' in _phrase_names(result)
    assert len(_atomic_names(result)) >= 3

    defines = _edges_of_type(result, EdgeType.DEFINES)
    references = _edges_of_type(result, EdgeType.REFERENCES)
    named_in = _edges_of_type(result, EdgeType.IS_NAMED_IN)

    define_targets = {edge.dst_id for edge in defines}
    symbol_atomics = set(tokenize('ToolbarViewModel', rules=CSHARP_DEFAULT_RULES))
    symbol_ids = {
        concept.id
        for concept in result.concepts
        if concept.kind == ConceptKind.ATOMIC and concept.name in symbol_atomics
    }
    assert symbol_ids.issubset(define_targets)

    reference_targets = {edge.dst_id for edge in references}
    assert define_targets.isdisjoint(reference_targets)

    assert len(named_in) == len(result.concepts)
    assert {edge.src_id for edge in named_in} == {concept.id for concept in result.concepts}


def test_phrase_span_masking_prevents_duplicate_atomics(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='race condition in the runtime',
        metadata={'source': '/x.cs', 'code_split_language': 'csharp'},
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    assert 'race-condition' in _phrase_names(result)
    assert 'race' not in _atomic_names(result)
    assert 'condition' not in _atomic_names(result)


def test_co_occurrence_edges_symmetric(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='Foo Bar Baz',
        metadata={'source': '/x.cs', 'code_split_language': 'csharp'},
    )
    result = extract(doc, glossary=default_glossary, now=_TS)
    assert len(result.concepts) == 3

    co_edges = _edges_of_type(result, EdgeType.CO_OCCURS_WITH)
    assert len(co_edges) == 6

    pairs = {(edge.src_id, edge.dst_id) for edge in co_edges}
    for left, right in ((1, 2), (1, 3), (2, 3)):
        assert (left, right) in pairs
        assert (right, left) in pairs


def test_defines_uses_ast_symbol(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='// body without the symbol name',
        metadata={
            'source': '/ToolbarViewModel.cs',
            'code_split_language': 'csharp',
            'ast_symbol': 'ToolbarViewModel',
        },
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    expected = set(tokenize('ToolbarViewModel', rules=CSHARP_DEFAULT_RULES))
    defines = _edges_of_type(result, EdgeType.DEFINES)
    defined_names = {
        next(concept.name for concept in result.concepts if concept.id == edge.dst_id)
        for edge in defines
    }
    assert expected.issubset(defined_names)
    assert expected == {'toolbar', 'view', 'model'}


def test_references_count_aggregation(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='token token token token token',
        metadata={'source': '/x.cs', 'code_split_language': 'csharp'},
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    references = _edges_of_type(result, EdgeType.REFERENCES)
    token_edge = next(
        edge
        for edge in references
        if next(concept.name for concept in result.concepts if concept.id == edge.dst_id) == 'token'
    )
    props = ReferencesProps(**dict(token_edge.properties))
    assert props.count == 5


def test_phrase_references_carry_positions(default_glossary: Glossary) -> None:
    text = 'race condition at start and another race condition later'
    doc = Document(
        page_content=text,
        metadata={'source': '/x.cs', 'code_split_language': 'csharp'},
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    expected_positions = tuple(hit.span[0] for hit in default_glossary.match(text))

    references = _edges_of_type(result, EdgeType.REFERENCES)
    phrase_id = next(
        concept.id for concept in result.concepts if concept.name == 'race-condition'
    )
    phrase_edge = next(edge for edge in references if edge.dst_id == phrase_id)
    props = ReferencesProps(**dict(phrase_edge.properties))
    assert props.count == 2
    assert tuple(props.positions or ()) == expected_positions


def test_language_falls_back_to_csharp_default(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='ToolbarViewModel',
        metadata={'source': '/x.cs', 'ast_symbol': 'ToolbarViewModel'},
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    assert _atomic_names(result) == {'toolbar', 'view', 'model'}


def test_extract_chunks_assigns_chunk_index_by_position(default_glossary: Glossary) -> None:
    chunks = [
        Document(page_content='chunk zero', metadata={'source': '/a.cs'}),
        Document(page_content='chunk one', metadata={'source': '/a.cs'}),
        Document(page_content='chunk two', metadata={'source': '/a.cs'}),
    ]
    results = extract_chunks(chunks, glossary=default_glossary, now=_TS)

    assert [result.artifact.chunk_index for result in results] == [0, 1, 2]


def test_concepts_have_stable_positional_ids(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='race condition in ToolbarViewModel with cancellation token',
        metadata={
            'source': '/ToolbarViewModel.cs',
            'code_split_language': 'csharp',
            'ast_symbol': 'ToolbarViewModel',
        },
    )
    first = extract(doc, glossary=default_glossary, now=_TS)
    second = extract(doc, glossary=default_glossary, now=_TS)

    assert [(concept.id, concept.name, concept.kind) for concept in first.concepts] == [
        (concept.id, concept.name, concept.kind) for concept in second.concepts
    ]
    assert [concept.id for concept in first.concepts] == list(
        range(1, len(first.concepts) + 1),
    )


def test_stats_counts_are_accurate(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='race condition in ToolbarViewModel with cancellation token',
        metadata={
            'source': '/ToolbarViewModel.cs',
            'code_split_language': 'csharp',
            'ast_symbol': 'ToolbarViewModel',
        },
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    atomics = [concept for concept in result.concepts if concept.kind == ConceptKind.ATOMIC]
    phrases = [concept for concept in result.concepts if concept.kind == ConceptKind.PHRASE]
    co_edges = _edges_of_type(result, EdgeType.CO_OCCURS_WITH)

    assert result.stats.distinct_atomic_count == len(atomics)
    assert result.stats.distinct_phrase_count == len(phrases)
    assert result.stats.co_occurrence_edge_count == len(co_edges)
    assert result.stats.chunk_text_length == len(doc.page_content)


def test_concept_phrase_kind_satisfies_invariant(default_glossary: Glossary) -> None:
    doc = Document(
        page_content='race condition and view model with cancellation token',
        metadata={'source': '/x.cs', 'code_split_language': 'csharp'},
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    for concept in result.concepts:
        if concept.kind == ConceptKind.PHRASE:
            assert concept.definition is not None


def test_lollipop_toolbar_viewmodel_chunk_fixture(default_glossary: Glossary) -> None:
    content = '''
/// <summary>
/// View-model for the floating Lollipop toolbar.
///
/// The toolbar's actions are supplied as a DI-registered collection of
/// <see cref="IToolbarExtension"/> objects — add or remove entries from the DI
/// registration in <c>App.xaml.cs</c> to customise the toolbar without touching
/// this class or <c>ToolbarWindow</c>. <see cref="ExtensionSettingsService"/>
/// filters which extensions are currently enabled; the toolbar rebuilds
/// automatically when that set changes.
/// </summary>
public sealed partial class ToolbarViewModel : ObservableObject
{
    private readonly SelectionService _selection;
    private readonly IReadOnlyList<IToolbarExtension> _allExtensions;
    private readonly ExtensionSettingsService _extensionSettings;
    private readonly DispatcherQueue _dispatcher;

    // Per-run cancellation source. Created on each ExecuteExtension call, cancelled
    // when the toolbar hides for any reason (via CancelActiveRun).
    private CancellationTokenSource? _runCts;
}
'''
    doc = Document(
        page_content=content,
        metadata={
            'source': '/Lollipop/ViewModels/ToolbarViewModel.cs',
            'code_split_language': 'csharp',
            'ast_symbol': 'ToolbarViewModel',
            'ast_kind': 'class_declaration',
            'ast_start_line': 11,
            'ast_end_line': 40,
        },
    )
    result = extract(doc, glossary=default_glossary, now=_TS)

    assert len(result.concepts) >= 10
    assert len(_phrase_names(result)) >= 1

    co_edges = _edges_of_type(result, EdgeType.CO_OCCURS_WITH)
    assert len(co_edges) >= 2

    atomic_ids = {
        concept.id for concept in result.concepts if concept.kind == ConceptKind.ATOMIC
    }
    atomic_pairs = {
        (edge.src_id, edge.dst_id)
        for edge in co_edges
        if edge.src_id in atomic_ids and edge.dst_id in atomic_ids
    }
    assert atomic_pairs


_STOPWORD_CHUNK = 'the toolbar is in the view'
_STOPWORD_CHUNK_METADATA = {'source': '/x.cs', 'code_split_language': 'csharp'}


def test_options_default_prunes_stopword_cooccurrences(default_glossary: Glossary) -> None:
    doc = Document(page_content=_STOPWORD_CHUNK, metadata=_STOPWORD_CHUNK_METADATA)
    result = extract(doc, glossary=default_glossary, now=_TS)

    co_endpoints = _co_occurrence_endpoint_names(result)
    assert 'the' not in co_endpoints
    assert 'is' not in co_endpoints
    assert 'in' not in co_endpoints
    assert {'toolbar', 'view'}.issubset(co_endpoints)


def test_options_false_emits_stopword_cooccurrences(default_glossary: Glossary) -> None:
    doc = Document(page_content=_STOPWORD_CHUNK, metadata=_STOPWORD_CHUNK_METADATA)
    result = extract(
        doc,
        glossary=default_glossary,
        now=_TS,
        options=ExtractorOptions(prune_stopword_cooccurrences=False),
    )

    co_endpoints = _co_occurrence_endpoint_names(result)
    assert {'the', 'is', 'in'}.issubset(co_endpoints)


def test_stopword_concepts_still_receive_references_and_is_named_in(
    default_glossary: Glossary,
) -> None:
    doc = Document(page_content=_STOPWORD_CHUNK, metadata=_STOPWORD_CHUNK_METADATA)
    result = extract(doc, glossary=default_glossary, now=_TS)

    the_id = next(
        concept.id for concept in result.concepts if concept.name == 'the'
    )
    references = _edges_of_type(result, EdgeType.REFERENCES)
    named_in = _edges_of_type(result, EdgeType.IS_NAMED_IN)

    assert any(edge.dst_id == the_id for edge in references)
    assert any(edge.src_id == the_id for edge in named_in)


def test_stats_records_pruned_count(default_glossary: Glossary) -> None:
    doc = Document(page_content=_STOPWORD_CHUNK, metadata=_STOPWORD_CHUNK_METADATA)
    result = extract(doc, glossary=default_glossary, now=_TS)

    assert result.stats.co_occurrence_pruned_count == 18
    assert result.stats.co_occurrence_edge_count == 2


def test_phrase_concept_never_treated_as_stopword(default_glossary: Glossary) -> None:
    glossary = default_glossary.merge(
        Glossary(
            [
                PhraseConcept(
                    name='in-the',
                    surface_forms=('in the',),
                    definition='Hypothetical phrase whose tokens are English stopwords.',
                    tags=(),
                ),
            ],
        ),
    )
    doc = Document(
        page_content='in the toolbar',
        metadata=_STOPWORD_CHUNK_METADATA,
    )
    result = extract(doc, glossary=glossary, now=_TS)

    phrase_id = next(
        concept.id for concept in result.concepts if concept.name == 'in-the'
    )
    toolbar_id = next(
        concept.id for concept in result.concepts if concept.name == 'toolbar'
    )
    co_pairs = {
        (edge.src_id, edge.dst_id)
        for edge in _edges_of_type(result, EdgeType.CO_OCCURS_WITH)
    }
    assert (phrase_id, toolbar_id) in co_pairs
    assert (toolbar_id, phrase_id) in co_pairs
