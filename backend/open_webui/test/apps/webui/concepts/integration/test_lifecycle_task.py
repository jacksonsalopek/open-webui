"""Tests for the concept-graph periodic rebuild lifecycle task."""

from __future__ import annotations

import asyncio
import os

# docker exec does not load start.sh's secret-key file; config import requires this.
if not os.environ.get('WEBUI_SECRET_KEY'):
    os.environ['WEBUI_SECRET_KEY'] = 'pytest-concept-graph-integration'

import pytest

from open_webui.retrieval.concepts.integration.lifecycle_task import (
    concept_graph_rebuild_loop,
)


def test_loop_calls_rebuild_when_enabled() -> None:
    calls = 0

    async def rebuild_fn() -> None:
        nonlocal calls
        calls += 1

    async def _run() -> None:
        task = asyncio.create_task(
            concept_graph_rebuild_loop(
                rebuild_fn=rebuild_fn,
                enabled_fn=lambda: True,
                interval_seconds_fn=lambda: 0,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert calls >= 2


def test_loop_skips_rebuild_when_disabled() -> None:
    called = False

    async def rebuild_fn() -> None:
        nonlocal called
        called = True

    async def _run() -> None:
        task = asyncio.create_task(
            concept_graph_rebuild_loop(
                rebuild_fn=rebuild_fn,
                enabled_fn=lambda: False,
                interval_seconds_fn=lambda: 0,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert called is False


def test_loop_recovers_from_rebuild_exception() -> None:
    calls = 0

    async def rebuild_fn() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('first rebuild failed')

    async def _run() -> None:
        task = asyncio.create_task(
            concept_graph_rebuild_loop(
                rebuild_fn=rebuild_fn,
                enabled_fn=lambda: True,
                interval_seconds_fn=lambda: 0,
            )
        )
        await asyncio.sleep(0.05)
        assert calls >= 2
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())


def test_loop_honors_cancellation() -> None:
    async def _run() -> None:
        task = asyncio.create_task(
            concept_graph_rebuild_loop(
                enabled_fn=lambda: True,
                interval_seconds_fn=lambda: 0,
            )
        )
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(_run())


def test_loop_rereads_enabled_each_iteration() -> None:
    enabled_states = [True, True, False]
    enabled_index = 0
    calls = 0

    def enabled_fn() -> bool:
        nonlocal enabled_index
        state = enabled_states[min(enabled_index, len(enabled_states) - 1)]
        enabled_index += 1
        return state

    async def rebuild_fn() -> None:
        nonlocal calls
        calls += 1

    async def _run() -> None:
        task = asyncio.create_task(
            concept_graph_rebuild_loop(
                rebuild_fn=rebuild_fn,
                enabled_fn=enabled_fn,
                interval_seconds_fn=lambda: 0,
            )
        )
        await asyncio.sleep(0.08)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert calls == 2


def test_make_rebuild_fn_calls_builder_with_roots(tmp_path) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from open_webui.config import CONCEPT_GRAPH_ROOTS
    from open_webui.retrieval.concepts.integration.lifecycle_task import (
        make_rebuild_fn,
    )

    original_roots = CONCEPT_GRAPH_ROOTS.value
    CONCEPT_GRAPH_ROOTS.value = f'{tmp_path}/r1:{tmp_path}/r2'
    try:
        store = MagicMock()
        app = SimpleNamespace(
            state=SimpleNamespace(concept_graph_store=store, concept_graph_dirty=True)
        )
        rebuild_fn = make_rebuild_fn(app)

        with patch('open_webui.retrieval.concepts.lifecycle.builder.build') as mock_build:
            mock_build.return_value = MagicMock(
                files_extracted=2,
                concepts_upserted=10,
            )
            asyncio.run(rebuild_fn())
            assert mock_build.call_count == 1
            call_plan = mock_build.call_args[0][0]
            assert len(call_plan.roots) == 2
            assert call_plan.roots[0] == tmp_path / 'r1'
            assert call_plan.roots[1] == tmp_path / 'r2'
    finally:
        CONCEPT_GRAPH_ROOTS.value = original_roots


def test_make_rebuild_fn_no_op_when_no_store() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from open_webui.retrieval.concepts.integration.lifecycle_task import (
        make_rebuild_fn,
    )

    app = SimpleNamespace(state=SimpleNamespace(concept_graph_store=None))
    rebuild_fn = make_rebuild_fn(app)

    with patch('open_webui.retrieval.concepts.lifecycle.builder.build') as mock_build:
        asyncio.run(rebuild_fn())
        mock_build.assert_not_called()


def test_make_rebuild_fn_no_op_when_roots_empty(caplog) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from open_webui.config import CONCEPT_GRAPH_ROOTS
    from open_webui.retrieval.concepts.integration.lifecycle_task import (
        make_rebuild_fn,
    )

    original_roots = CONCEPT_GRAPH_ROOTS.value
    CONCEPT_GRAPH_ROOTS.value = ''
    try:
        store = MagicMock()
        app = SimpleNamespace(state=SimpleNamespace(concept_graph_store=store))
        rebuild_fn = make_rebuild_fn(app)

        with patch('open_webui.retrieval.concepts.lifecycle.builder.build') as mock_build:
            with caplog.at_level('WARNING'):
                asyncio.run(rebuild_fn())
            mock_build.assert_not_called()
            assert any(
                'CONCEPT_GRAPH_ROOTS is empty' in record.message
                for record in caplog.records
            )
    finally:
        CONCEPT_GRAPH_ROOTS.value = original_roots


def test_make_rebuild_fn_clears_dirty_flag() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from open_webui.config import CONCEPT_GRAPH_ROOTS
    from open_webui.retrieval.concepts.integration.lifecycle_task import (
        make_rebuild_fn,
    )

    original_roots = CONCEPT_GRAPH_ROOTS.value
    CONCEPT_GRAPH_ROOTS.value = '/tmp/cg-test'
    try:
        store = MagicMock()
        app = SimpleNamespace(
            state=SimpleNamespace(concept_graph_store=store, concept_graph_dirty=True)
        )
        rebuild_fn = make_rebuild_fn(app)

        with patch('open_webui.retrieval.concepts.lifecycle.builder.build') as mock_build:
            mock_build.return_value = MagicMock(
                files_extracted=1,
                concepts_upserted=5,
            )
            asyncio.run(rebuild_fn())
            assert app.state.concept_graph_dirty is False
    finally:
        CONCEPT_GRAPH_ROOTS.value = original_roots
