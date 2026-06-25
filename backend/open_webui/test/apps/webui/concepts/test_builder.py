"""Tests for ``open_webui.retrieval.concepts.lifecycle.builder``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from open_webui.retrieval.concepts.lifecycle.builder import (
    BuildPlan,
    BuilderPruneOptions,
    build,
)
from open_webui.retrieval.concepts.schema import ConceptKind, EdgeType
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_LOLLIPOP_VIEWMODELS = Path('/tmp/lollipop_subset')
_LOLLIPOP_HOST = Path(
    '/Users/jacksonsalopek/dev/startups/ventana/lollipop/src/Lollipop/ViewModels',
)


def _count_co_occurrence_edges(store: InMemoryGraphStore) -> int:
    return sum(1 for key in store._edges if key[0] == EdgeType.CO_OCCURS_WITH)


def _co_occurrence_weight(
    store: InMemoryGraphStore,
    name_a: str,
    name_b: str,
) -> float | None:
    id_a = store._by_name_kind.get((name_a, ConceptKind.ATOMIC))
    id_b = store._by_name_kind.get((name_b, ConceptKind.ATOMIC))
    if id_a is None or id_b is None:
        return None
    edge = store._edges.get((EdgeType.CO_OCCURS_WITH, id_a, id_b))
    if edge is None:
        edge = store._edges.get((EdgeType.CO_OCCURS_WITH, id_b, id_a))
    if edge is None:
        return None
    return float(edge.properties['weight'])


def test_build_empty_root_returns_empty_result(tmp_path: Path) -> None:
    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(tmp_path,))
    result = build(plan, store)

    assert result.files_seen == 0
    assert result.files_extracted == 0
    assert result.chunks_extracted == 0
    assert result.concepts_upserted == 0
    assert result.artifacts_upserted == 0
    assert result.edges_persisted == 0


def test_build_skips_excluded_globs(tmp_path: Path) -> None:
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'obj').mkdir()
    (tmp_path / 'node_modules').mkdir()
    (tmp_path / 'bin' / 'Ignored.cs').write_text('class Ignored {}', encoding='utf-8')
    (tmp_path / 'obj' / 'Ignored.cs').write_text('class Ignored {}', encoding='utf-8')
    (tmp_path / 'node_modules' / 'Ignored.cs').write_text(
        'class Ignored {}',
        encoding='utf-8',
    )
    real = tmp_path / 'Real.cs'
    real.write_text(
        '''
public class RealViewModel
{
    private readonly ModelService _model;
}
''',
        encoding='utf-8',
    )

    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(tmp_path,), language_hint='csharp')
    result = build(plan, store)

    assert result.files_seen == 1
    assert result.files_extracted == 1
    assert result.files_skipped == 0


def test_build_skips_non_utf8_files(tmp_path: Path) -> None:
    bad = tmp_path / 'Bad.cs'
    bad.write_bytes(b'\xff\xfe' + b'class Bad {}')
    good = tmp_path / 'Good.cs'
    good.write_text('class Good {}', encoding='utf-8')

    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(tmp_path,), language_hint='csharp')
    result = build(plan, store)

    assert result.files_skipped == 1
    assert result.files_extracted == 1
    assert result.files_failed == 0


def test_build_skips_oversize_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    big = tmp_path / 'Big.cs'
    big.write_text('x' * (6 * 1024 * 1024), encoding='utf-8')
    small = tmp_path / 'Small.cs'
    small.write_text('class Small {}', encoding='utf-8')

    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(tmp_path,), language_hint='csharp')
    with caplog.at_level('WARNING'):
        result = build(plan, store)

    assert result.files_skipped == 1
    assert any('byte cap' in record.message for record in caplog.records)
    assert result.files_extracted == 1


def test_build_continues_on_per_file_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = tmp_path / 'Bad.cs'
    bad.write_text('class Bad {}', encoding='utf-8')
    good = tmp_path / 'Good.cs'
    good.write_text('class Good {}', encoding='utf-8')

    original = __import__(
        'open_webui.retrieval.concepts.lifecycle.builder',
        fromlist=['split_code'],
    ).split_code

    def flaky_split(text: str, language: str | None, **kwargs: object) -> list:
        if 'Bad' in str(kwargs.get('base_metadata', {}).get('source', '')):
            raise RuntimeError('boom')
        return original(text, language, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        'open_webui.retrieval.concepts.lifecycle.builder.split_code',
        flaky_split,
    )

    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(tmp_path,), language_hint='csharp')
    result = build(plan, store)

    assert result.files_failed == 1
    assert result.files_extracted == 1


def test_build_aggregates_co_occurrence_cross_file(tmp_path: Path) -> None:
    file_a = tmp_path / 'ViewA.cs'
    file_a.write_text(
        '''
public class ViewA
{
    private readonly ModelService _model;
    public void RenderView() { }
}
''',
        encoding='utf-8',
    )
    file_b = tmp_path / 'ViewB.cs'
    file_b.write_text(
        '''
public class ViewB
{
    private readonly ModelService _model;
    public void ShowView() { }
}
''',
        encoding='utf-8',
    )

    store = InMemoryGraphStore()
    plan = BuildPlan(
        roots=(tmp_path,),
        language_hint='csharp',
        builder_prune=BuilderPruneOptions(min_cooccurrence_weight=1),
    )
    build(plan, store)

    weight = _co_occurrence_weight(store, 'view', 'model')
    assert weight is not None
    assert weight >= 2.0


def test_build_min_cooccurrence_weight_drops_singletons(tmp_path: Path) -> None:
    single = tmp_path / 'Single.cs'
    single.write_text(
        '''
public class ToolbarViewModel
{
    private readonly SelectionService _selection;
    private readonly ExtensionSettingsService _extensionSettings;
    public void ExecuteExtension() { }
}
''',
        encoding='utf-8',
    )

    store = InMemoryGraphStore()
    plan = BuildPlan(
        roots=(tmp_path,),
        language_hint='csharp',
        builder_prune=BuilderPruneOptions(min_cooccurrence_weight=2),
    )
    result = build(plan, store)

    co_count = _count_co_occurrence_edges(store)
    non_co = sum(1 for key in store._edges if key[0] != EdgeType.CO_OCCURS_WITH)
    assert result.edges_pruned_by_weight > 0
    assert co_count < result.edges_emitted
    assert non_co > 0


def test_build_idempotent_on_rerun(tmp_path: Path) -> None:
    src = tmp_path / 'Widget.cs'
    src.write_text(
        '''
public class WidgetViewModel
{
    private readonly DataService _data;
}
''',
        encoding='utf-8',
    )

    store = InMemoryGraphStore()
    plan = BuildPlan(
        roots=(tmp_path,),
        language_hint='csharp',
        builder_prune=BuilderPruneOptions(min_cooccurrence_weight=1),
    )
    first = build(plan, store)
    edge_count_first = len(store._edges)
    second = build(plan, store)

    assert second.concepts_upserted == first.concepts_upserted
    assert second.artifacts_upserted == first.artifacts_upserted
    assert len(store._edges) == edge_count_first


def test_build_records_per_phase_progress_callback(tmp_path: Path) -> None:
    src = tmp_path / 'Item.cs'
    src.write_text('class Item {}', encoding='utf-8')

    events: list[str] = []

    def callback(phase: str, _payload: dict[str, Any]) -> None:
        events.append(phase)

    store = InMemoryGraphStore()
    plan = BuildPlan(roots=(tmp_path,), language_hint='csharp')
    build(plan, store, progress_callback=callback)

    assert events == [
        'discover',
        'extract',
        'aggregate',
        'persist_artifacts',
        'persist_concepts',
        'persist_edges',
        'centrality',
        'idf',
    ]


def _synthetic_csharp_tree(root: Path) -> None:
    """Representative mini C# tree when Lollipop mount is unavailable."""
    (root / 'ViewModels').mkdir(parents=True, exist_ok=True)
    (root / 'Services').mkdir(parents=True, exist_ok=True)
    (root / 'ViewModels' / 'ToolbarViewModel.cs').write_text(
        '''
public sealed partial class ToolbarViewModel : ObservableObject
{
    private readonly SelectionService _selection;
    private readonly ExtensionSettingsService _extensionSettings;
    private readonly DispatcherQueue _dispatcher;
    public void ExecuteExtension() { }
    private void CancelActiveRun() { }
}
''',
        encoding='utf-8',
    )
    (root / 'ViewModels' / 'SettingsViewModel.cs').write_text(
        '''
public sealed class SettingsViewModel
{
    private readonly ExtensionSettingsService _extensionSettings;
    public void SaveSettings() { }
}
''',
        encoding='utf-8',
    )
    (root / 'Services' / 'SelectionService.cs').write_text(
        '''
public sealed class SelectionService
{
    public string CurrentSelection { get; private set; }
    public void UpdateSelection(string value) { CurrentSelection = value; }
}
''',
        encoding='utf-8',
    )


def test_build_lollipop_subset_sanity(tmp_path: Path) -> None:
    if _LOLLIPOP_VIEWMODELS.is_dir() and any(_LOLLIPOP_VIEWMODELS.glob('*.cs')):
        roots = (_LOLLIPOP_VIEWMODELS,)
        source = 'container /tmp/lollipop_subset'
    elif _LOLLIPOP_HOST.is_dir():
        roots = (_LOLLIPOP_HOST,)
        source = 'host Lollipop ViewModels'
    else:
        _synthetic_csharp_tree(tmp_path)
        roots = (tmp_path,)
        source = 'synthetic tmp_path fixture'

    store = InMemoryGraphStore()
    plan = BuildPlan(
        roots=roots,
        language_hint='csharp',
        builder_prune=BuilderPruneOptions(min_cooccurrence_weight=1),
    )
    result = build(plan, store)

    assert result.files_extracted >= 1, f'no files from {source}'
    assert result.chunks_extracted >= 1
    assert result.concepts_upserted >= 50
    assert result.edges_persisted >= 50
