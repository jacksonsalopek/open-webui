"""Post-vector-write hook that marks the concept graph as dirty.

When CONCEPT_GRAPH_ENABLED is True AND app.state has a concept_graph_store,
this hook sets app.state.concept_graph_dirty = True so the periodic
lifecycle task (see retrieval/concepts/integration/lifecycle_task.py)
picks up the rebuild on its next cycle.

Phase 2 gate-0 design choice: ingest does NOT trigger a synchronous
chunk-level graph write. See CONCEPT_GRAPH_PHASE2.md §"Hook 3" —
delta-apply is deferred to Phase 2.5. The dirty-flag pattern lights up
the hook surface without inventing a non-spec'd delta-apply path."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document

log = logging.getLogger(__name__)


def on_docs_saved(
    app_state: Any,
    *,
    collection_name: str,
    docs: Sequence[Document] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Mark the concept graph as needing a rebuild.

    Called from routers/retrieval.py::save_docs_to_vector_db after a
    successful VECTOR_DB_CLIENT.insert. NEVER raises — all exceptions
    are logged and swallowed; a graph-side failure must not regress
    the already-completed vector write.

    No-op when:
      - CONCEPT_GRAPH_ENABLED is False
      - app_state has no concept_graph_store attribute (W2-C didn't run)
    """
    try:
        from open_webui.config import CONCEPT_GRAPH_ENABLED

        if not CONCEPT_GRAPH_ENABLED.value:
            return

        if getattr(app_state, 'concept_graph_store', None) is None:
            log.debug('concept_graph_dirty signal ignored: no store on app.state')
            return

        app_state.concept_graph_dirty = True
        docs_count = len(docs) if docs else 0
        log.info(
            'concept_graph_dirty set after ingest: collection=%s docs=%d',
            collection_name,
            docs_count,
        )
    except Exception:
        log.exception('concept_graph ingest hook failed; vector write unaffected')
