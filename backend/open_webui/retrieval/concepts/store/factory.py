"""Minimal substrate-agnostic factory for GraphStore implementations.

Phase 2 v1: dispatches between InMemoryGraphStore and KuzuGraphStore
based on a backend string. Mirrors the pattern in
open_webui.retrieval.vector.factory.
"""

from __future__ import annotations

from pathlib import Path

from open_webui.retrieval.concepts.store.protocol import GraphStore


def create_graph_store(
    *,
    backend: str,
    embedding_dim: int,
    kuzu_path: str | None = None,
) -> GraphStore:
    """Construct a GraphStore.

    backend: 'memory' or 'kuzu'. Case-insensitive.
    embedding_dim: required for both backends (used to size HNSW for kuzu;
                   matches the configured embedder dimension).
    kuzu_path: required when backend == 'kuzu'. Created if missing.

    Raises ValueError for unknown backend strings or missing required args.
    """
    normalized = backend.lower()

    if normalized == 'memory':
        from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore

        return InMemoryGraphStore()

    if normalized == 'kuzu':
        if not kuzu_path:
            raise ValueError('kuzu_path is required when backend is kuzu')

        Path(kuzu_path).parent.mkdir(parents=True, exist_ok=True)

        from open_webui.retrieval.concepts.store.kuzu_store import KuzuGraphStore

        return KuzuGraphStore(db_path=kuzu_path, embedding_dim=embedding_dim)

    raise ValueError(
        f"Unsupported concept graph store backend: {backend!r}. "
        "Supported backends: 'memory', 'kuzu'."
    )
