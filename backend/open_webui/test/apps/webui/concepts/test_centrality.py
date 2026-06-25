"""Tests for ``open_webui.retrieval.concepts.lifecycle.centrality``."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from open_webui.retrieval.concepts.extraction.extractor import extract
from open_webui.retrieval.concepts.extraction.glossary import Glossary
from open_webui.retrieval.concepts.extraction.stopwords import is_stopword
from open_webui.retrieval.concepts.lifecycle.builder import BuildPlan, BuilderPruneOptions, build
from open_webui.retrieval.concepts.lifecycle.centrality import (
    CentralityScores,
    clear_cache,
    compute,
    compute_and_persist,
    get_cached,
)
from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    CoOccursWithProps,
    DefinesProps,
    Edge,
    EdgeType,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
from langchain_core.documents import Document

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_TOOLBAR_FIXTURE = '''
public sealed partial class ToolbarViewModel : ObservableObject
{
    private readonly SelectionService _selection;
    private readonly IReadOnlyList<IToolbarExtension> _allExtensions;
    private readonly ExtensionSettingsService _extensionSettings;
    private readonly DispatcherQueue _dispatcher;
    private CancellationTokenSource? _runCts;

    public void ExecuteExtension() { }
    private void CancelActiveRun() { }
}
'''

_CODE_STOPWORDS = frozenset(
    {'if', 'null', 'return', 'string', 'for', 'var', 'void', 'true', 'false'},
)


@pytest.fixture
def default_glossary() -> Glossary:
    return Glossary.default()


def _upsert_concept(store: InMemoryGraphStore, name: str) -> int:
    return store.upsert_concept(
        Concept(
            id=0,
            name=name,
            kind=ConceptKind.ATOMIC,
            first_seen_at=_TS,
            last_seen_at=_TS,
            centrality_score=None,
            embedding=None,
            definition=None,
            language_hint=None,
            original_tokens=(name,),
        ),
    )


def _upsert_artifact(store: InMemoryGraphStore, path: str) -> int:
    return store.upsert_artifact(
        Artifact(
            id=0,
            kind=ArtifactKind.CHUNK,
            path=path,
            chunk_index=0,
            language='csharp',
            byte_start=0,
            byte_end=100,
            last_modified_at=_TS,
        ),
    )


def _link_co(store: InMemoryGraphStore, src: int, dst: int) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=src,
            dst_id=dst,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )


def _link_defines(store: InMemoryGraphStore, artifact_id: int, concept_id: int) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=artifact_id,
            dst_id=concept_id,
            props=DefinesProps(count=1),
        ),
    )


def test_compute_returns_both_scores() -> None:
    store = InMemoryGraphStore()
    hub = _upsert_concept(store, 'hub')
    for name in ('alpha', 'beta', 'gamma'):
        spoke = _upsert_concept(store, name)
        _link_co(store, spoke, hub)

    artifact = _upsert_artifact(store, '/a.cs')
    _link_defines(store, artifact, hub)

    scores = compute(store)
    assert scores.semantic
    assert scores.structural
    assert abs(sum(scores.semantic.values()) - 1.0) < 1e-6
    assert abs(sum(scores.structural.values()) - 1.0) < 1e-6


def test_compute_semantic_filters_to_cooccurrence() -> None:
    store = InMemoryGraphStore()
    anchor = _upsert_concept(store, 'anchor')
    for i in range(6):
        artifact = _upsert_artifact(store, f'/def{i}.cs')
        _link_defines(store, artifact, anchor)

    hub = _upsert_concept(store, 'hub')
    for name in ('s1', 's2', 's3', 's4', 's5'):
        spoke = _upsert_concept(store, name)
        _link_co(store, spoke, hub)
        _link_co(store, hub, spoke)

    scores = compute(store, iterations=30)
    n = len(scores.semantic)
    baseline = 1.0 / n

    assert scores.semantic[hub] > baseline * 2
    assert scores.semantic[anchor] < scores.semantic[hub]


def test_compute_structural_filters_to_defines_references() -> None:
    store = InMemoryGraphStore()
    hub = _upsert_concept(store, 'hub')
    for name in ('s1', 's2', 's3', 's4', 's5'):
        spoke = _upsert_concept(store, name)
        _link_co(store, spoke, hub)
        _link_co(store, hub, spoke)

    anchor = _upsert_concept(store, 'anchor')
    for i in range(6):
        artifact = _upsert_artifact(store, f'/ref{i}.cs')
        _link_defines(store, artifact, anchor)

    scores = compute(store, iterations=30)
    n = len(scores.structural)
    baseline = 1.0 / n

    assert scores.structural[hub] == pytest.approx(baseline, rel=0.15)
    assert scores.semantic[hub] > baseline * 2


def test_compute_and_persist_caches_results() -> None:
    store = InMemoryGraphStore()
    a = _upsert_concept(store, 'a')
    b = _upsert_concept(store, 'b')
    _link_co(store, a, b)

    compute_and_persist(store)
    cached = get_cached(store)
    assert cached is not None
    assert isinstance(cached, CentralityScores)
    assert cached.semantic == get_cached(store).semantic


def test_compute_and_persist_idempotent() -> None:
    store = InMemoryGraphStore()
    a = _upsert_concept(store, 'a')
    b = _upsert_concept(store, 'b')
    _link_co(store, a, b)

    first_at = compute_and_persist(store)
    first_scores = get_cached(store)
    assert first_scores is not None

    time.sleep(0.01)
    second_at = compute_and_persist(store)
    second_scores = get_cached(store)
    assert second_scores is not None
    assert second_at >= first_at
    assert second_scores.computed_at >= first_scores.computed_at


def test_clear_cache_evicts() -> None:
    store = InMemoryGraphStore()
    _upsert_concept(store, 'solo')
    compute_and_persist(store)
    assert get_cached(store) is not None

    clear_cache(store)
    assert get_cached(store) is None


def test_lollipop_semantic_centrality_top5_quality(
    tmp_path: Path,
    default_glossary: Glossary,
) -> None:
    glossary = default_glossary
    store = InMemoryGraphStore()

    lollipop_subset = Path('/tmp/lollipop_subset')
    toolbar_path = Path('/tmp/ToolbarViewModel.cs')

    if toolbar_path.is_file():
        plan = BuildPlan(
            roots=(toolbar_path.parent,),
            language_hint='csharp',
            include_globs=(toolbar_path.name,),
            builder_prune=BuilderPruneOptions(min_cooccurrence_weight=2),
        )
        build(plan, store)
        expect_domain = True
    elif lollipop_subset.is_dir() and any(lollipop_subset.glob('*.cs')):
        plan = BuildPlan(
            roots=(lollipop_subset,),
            language_hint='csharp',
            builder_prune=BuilderPruneOptions(min_cooccurrence_weight=2),
        )
        build(plan, store)
        expect_domain = False
    else:
        doc = Document(
            page_content=_TOOLBAR_FIXTURE,
            metadata={
                'source': str(tmp_path / 'ToolbarViewModel.cs'),
                'code_split_language': 'csharp',
                'ast_symbol': 'ToolbarViewModel',
            },
        )
        result = extract(doc, glossary=glossary, now=_TS)
        artifact_id = store.upsert_artifact(result.artifact)
        concept_ids = store.upsert_concepts_batch(list(result.concepts))
        key_to_id = {
            (c.name, c.kind): concept_ids[i] for i, c in enumerate(result.concepts)
        }

        def remap(node_id: int) -> int:
            if node_id == 0:
                return artifact_id
            concept = next(c for c in result.concepts if c.id == node_id)
            return key_to_id[(concept.name, concept.kind)]

        edges = [
            Edge(
                type=edge.type,
                src_id=remap(edge.src_id),
                dst_id=remap(edge.dst_id),
                properties=edge.properties,
            )
            for edge in result.edges
        ]
        store.upsert_edges_batch(edges)
        compute_and_persist(store)
        expect_domain = True

    scores = get_cached(store)
    assert scores is not None

    top_sem = sorted(scores.semantic.items(), key=lambda kv: -kv[1])[:5]
    top_str = sorted(scores.structural.items(), key=lambda kv: -kv[1])[:5]

    sem_names = {store.get_concept(cid).name for cid, _ in top_sem}  # type: ignore[union-attr]

    # Extractor-pruned + stopword-classified tokens must not dominate semantic rank.
    classified_stopword_hits = {
        name for name in sem_names if is_stopword(name, language='csharp')
    }
    assert not classified_stopword_hits, (
        f'semantic top-5 contains classified stopwords: {classified_stopword_hits}'
    )

    # Hard-coded C# keyword list from the Phase 1 quality gate. Tokens absent from
    # the stopword tables (e.g. ``if``, ``null``) may still leak until risk #4 lands.
    aspirational_leaks = sem_names & _CODE_STOPWORDS
    if aspirational_leaks:
        pytest.xfail(
            f'carry-forward risk #4: semantic top-5 still contains unclassified '
            f'C# keywords {aspirational_leaks}',
        )

    domain_hits = sem_names & {
        'toolbar',
        'extension',
        'viewmodel',
        'selection',
        'observable',
        'execute',
        'model',
    }
    if expect_domain:
        assert domain_hits, f'expected domain concepts in semantic top-5, got {sem_names}'
