"""Tests for concept-graph store factory dispatch."""

from __future__ import annotations

import os

# docker exec does not load start.sh's secret-key file; config import requires this.
if not os.environ.get('WEBUI_SECRET_KEY'):
    os.environ['WEBUI_SECRET_KEY'] = 'pytest-concept-graph-integration'

import pytest

from open_webui.config import (
    CONCEPT_GRAPH_EMBEDDING_DIM,
    CONCEPT_GRAPH_STORE_BACKEND,
)
from open_webui.retrieval.concepts.integration.store_factory import (
    create_graph_store_from_config,
)
from open_webui.retrieval.concepts.store.factory import create_graph_store
from open_webui.retrieval.concepts.store.kuzu_store import KuzuGraphStore
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore


def test_create_memory_store_succeeds() -> None:
    store = create_graph_store(backend='memory', embedding_dim=16)
    assert hasattr(store, 'upsert_concept')
    assert callable(store.upsert_concept)


def test_create_memory_store_case_insensitive() -> None:
    for backend in ('Memory', 'MEMORY'):
        store = create_graph_store(backend=backend, embedding_dim=16)
        assert isinstance(store, InMemoryGraphStore)


def test_create_kuzu_store_in_tmp_path(tmp_path) -> None:
    kuzu_path = tmp_path / 'nested' / 'test.kuzu'
    store = create_graph_store(
        backend='kuzu',
        embedding_dim=16,
        kuzu_path=str(kuzu_path),
    )

    assert isinstance(store, KuzuGraphStore)
    assert kuzu_path.parent.is_dir()


def test_create_kuzu_store_requires_path() -> None:
    with pytest.raises(ValueError, match='kuzu_path'):
        create_graph_store(backend='kuzu', embedding_dim=16, kuzu_path=None)
    with pytest.raises(ValueError, match='kuzu_path'):
        create_graph_store(backend='kuzu', embedding_dim=16, kuzu_path='')


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Supported backends: 'memory', 'kuzu'"):
        create_graph_store(backend='postgres', embedding_dim=16)


def test_create_graph_store_from_config_reads_config() -> None:
    original_backend = CONCEPT_GRAPH_STORE_BACKEND.value
    original_dim = CONCEPT_GRAPH_EMBEDDING_DIM.value
    try:
        CONCEPT_GRAPH_STORE_BACKEND.value = 'memory'
        CONCEPT_GRAPH_EMBEDDING_DIM.value = 16

        store = create_graph_store_from_config()
        assert isinstance(store, InMemoryGraphStore)
    finally:
        CONCEPT_GRAPH_STORE_BACKEND.value = original_backend
        CONCEPT_GRAPH_EMBEDDING_DIM.value = original_dim
