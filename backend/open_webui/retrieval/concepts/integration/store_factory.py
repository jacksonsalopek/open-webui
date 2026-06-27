"""Config-aware GraphStore factory.

Reads the CONCEPT_GRAPH_* config vars from open_webui.config and
constructs a GraphStore. Used at FastAPI startup by the lifecycle task.
"""

from __future__ import annotations

import logging

from open_webui.config import (
    CONCEPT_GRAPH_EMBEDDING_DIM,
    CONCEPT_GRAPH_KUZU_PATH,
    CONCEPT_GRAPH_STORE_BACKEND,
)
from open_webui.retrieval.concepts.store.factory import create_graph_store
from open_webui.retrieval.concepts.store.protocol import GraphStore

log = logging.getLogger(__name__)


def create_graph_store_from_config() -> GraphStore:
    """Read concept-graph config and build a GraphStore.

    Reads:
      CONCEPT_GRAPH_STORE_BACKEND
      CONCEPT_GRAPH_KUZU_PATH (when backend == 'kuzu')
      CONCEPT_GRAPH_EMBEDDING_DIM

    Returns the constructed GraphStore. Logs the backend and dim on
    success; raises any underlying ValueError from the lower-level
    factory.
    """
    backend = CONCEPT_GRAPH_STORE_BACKEND.value
    embedding_dim = CONCEPT_GRAPH_EMBEDDING_DIM.value
    kuzu_path = CONCEPT_GRAPH_KUZU_PATH.value if backend.lower() == 'kuzu' else None

    store = create_graph_store(
        backend=backend,
        embedding_dim=embedding_dim,
        kuzu_path=kuzu_path,
    )

    if backend.lower() == 'kuzu':
        log.info(
            'Concept graph store created: backend=%s embedding_dim=%s kuzu_path=%s',
            backend,
            embedding_dim,
            kuzu_path,
        )
    else:
        log.info(
            'Concept graph store created: backend=%s embedding_dim=%s',
            backend,
            embedding_dim,
        )

    return store
