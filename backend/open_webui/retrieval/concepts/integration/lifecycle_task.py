"""Async lifecycle task for periodic concept-graph rebuilds.

Phase 2 W1 ships the scaffold: spawn behavior, sleep loop,
cancellation, exception isolation. The rebuild itself is a placeholder
(does nothing but log). Wave 2 replaces the placeholder with the real
lifecycle.builder.build() call.

Reads CONCEPT_GRAPH_ENABLED, CONCEPT_GRAPH_REBUILD_INTERVAL_SECONDS at
task start. Re-reads them on each iteration so a config flip (without
restart) takes effect on the next cycle.

Not joined to the scheduler_worker_loop; this is a top-level
asyncio.create_task spawned from the FastAPI lifespan, following the
pattern of periodic_usage_pool_cleanup in main.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from open_webui.retrieval.concepts.extraction.glossary import Glossary

log = logging.getLogger(__name__)

_DISABLED_RECHECK_SECONDS = 60

# Wave 2 swaps this for the real builder call.
RebuildFn = Callable[[], Awaitable[None]]


async def _default_rebuild_placeholder() -> None:
    """Wave 1 placeholder — logs and exits. Wave 2 replaces this with
    a real lifecycle.builder.build(...) wrapped in to_thread."""
    log.info('concept_graph_rebuild: placeholder rebuild (Wave 1 scaffold)')


def _load_glossary(paths_raw: str) -> 'Glossary | None':
    """Load glossary from colon-separated YAML paths, else return None
    (builder will use Glossary.default() in that case)."""
    if not paths_raw.strip():
        return None
    from open_webui.retrieval.concepts.extraction.glossary import Glossary

    paths = [p.strip() for p in paths_raw.split(':') if p.strip()]
    if not paths:
        return None
    return Glossary.from_paths(paths)


def make_rebuild_fn(app: Any) -> RebuildFn:
    """Return an async rebuild_fn bound to ``app.state.concept_graph_store``.

    The closure reads CONCEPT_GRAPH_ROOTS + CONCEPT_GRAPH_GLOSSARY_PATHS,
    constructs a BuildPlan, and calls lifecycle.builder.build via
    asyncio.to_thread. On success, clears app.state.concept_graph_dirty
    so the ingest hook can re-trigger a future rebuild.

    No-ops (with a debug log) when:
      - app.state.concept_graph_store is missing or None
      - CONCEPT_GRAPH_ROOTS is empty
    """

    async def rebuild_fn() -> None:
        from open_webui.config import (
            CONCEPT_GRAPH_GLOSSARY_PATHS,
            CONCEPT_GRAPH_ROOTS,
        )
        from open_webui.retrieval.concepts.lifecycle.builder import (
            BuildPlan,
            build,
        )

        store = getattr(app.state, 'concept_graph_store', None)
        if store is None:
            log.debug('concept_graph rebuild: no store on app.state; skipping')
            return

        roots_raw = CONCEPT_GRAPH_ROOTS.value or ''
        roots = tuple(Path(p) for p in roots_raw.split(':') if p.strip())
        if not roots:
            log.warning(
                'concept_graph rebuild: CONCEPT_GRAPH_ROOTS is empty; skipping. '
                'Set CONCEPT_GRAPH_ROOTS=/path1:/path2 to enable rebuilds.'
            )
            return

        glossary = _load_glossary(CONCEPT_GRAPH_GLOSSARY_PATHS.value or '')
        plan = BuildPlan(roots=roots, glossary=glossary)

        log.info(
            'concept_graph rebuild: starting build over %d root(s): %s',
            len(roots),
            [str(p) for p in roots],
        )
        result = await asyncio.to_thread(build, plan, store)
        log.info(
            'concept_graph rebuild: completed files=%d concepts=%d',
            result.files_extracted,
            result.concepts_upserted,
        )

        try:
            setattr(app.state, 'concept_graph_dirty', False)
        except Exception:
            log.exception('concept_graph rebuild: failed to clear dirty flag')

    return rebuild_fn


def _default_interval_seconds() -> int:
    try:
        from open_webui.config import CONCEPT_GRAPH_REBUILD_INTERVAL_SECONDS

        return int(CONCEPT_GRAPH_REBUILD_INTERVAL_SECONDS.value)
    except ImportError:
        # Soft fallback for W1 when config additions are not yet importable.
        # Wave 2 can remove this once config is guaranteed at import time.
        return int(os.environ.get('CONCEPT_GRAPH_REBUILD_INTERVAL_SECONDS', '86400'))


def _default_enabled() -> bool:
    try:
        from open_webui.config import CONCEPT_GRAPH_ENABLED

        return bool(CONCEPT_GRAPH_ENABLED.value)
    except ImportError:
        return os.environ.get('CONCEPT_GRAPH_ENABLED', 'False').lower() == 'true'


async def concept_graph_rebuild_loop(
    *,
    rebuild_fn: RebuildFn = _default_rebuild_placeholder,
    interval_seconds_fn: Callable[[], int] = _default_interval_seconds,
    enabled_fn: Callable[[], bool] = _default_enabled,
) -> None:
    """Long-running asyncio task. Lifespan-spawned at startup.

    Behavior:
      1. On every loop iteration: read enabled_fn(). If False, sleep
         briefly (60s) and re-check. (The task stays alive so a config
         flip takes effect.)
      2. When enabled: call rebuild_fn(). Catch ALL exceptions, log
         at error level, do NOT crash the loop.
      3. After each rebuild (success or failure), sleep for
         interval_seconds_fn() seconds.
      4. Honor asyncio cancellation: re-raise CancelledError from
         asyncio.sleep so FastAPI shutdown can stop the loop cleanly.

    interval_seconds_fn and enabled_fn are callables (not values) so
    the task picks up live config changes on each iteration.
    """
    while True:
        if not enabled_fn():
            log.warning(
                'concept_graph_rebuild: disabled, sleeping %ds',
                _DISABLED_RECHECK_SECONDS,
            )
            await asyncio.sleep(_DISABLED_RECHECK_SECONDS)
            continue

        log.info('concept_graph_rebuild: starting iteration')
        start = time.monotonic()
        try:
            await rebuild_fn()
        except Exception:
            log.error('concept_graph_rebuild: rebuild raised', exc_info=True)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        log.info('concept_graph_rebuild: completed in %dms', elapsed_ms)

        await asyncio.sleep(interval_seconds_fn())


def spawn_concept_graph_rebuild_loop(app: Any = None) -> asyncio.Task:
    """Convenience wrapper for FastAPI lifespan use.

    When ``app`` is provided, the loop runs the real builder against
    app.state.concept_graph_store. When ``app`` is None, the loop uses
    the W1 placeholder rebuild (for tests).

    Returns the spawned Task so the caller can hold a reference (avoid
    GC) and cancel on shutdown.
    """
    rebuild_fn: RebuildFn
    if app is not None:
        rebuild_fn = make_rebuild_fn(app)
    else:
        rebuild_fn = _default_rebuild_placeholder

    return asyncio.create_task(
        concept_graph_rebuild_loop(rebuild_fn=rebuild_fn),
        name='concept_graph_rebuild',
    )
