"""Tests for the concept-graph ingest hook."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

if not os.environ.get('WEBUI_SECRET_KEY'):
    os.environ['WEBUI_SECRET_KEY'] = 'pytest-concept-graph-integration'

from langchain_core.documents import Document

from open_webui.config import CONCEPT_GRAPH_ENABLED
from open_webui.retrieval.concepts.integration.ingest_hook import on_docs_saved


def test_no_op_when_disabled() -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = False
    try:
        app_state = SimpleNamespace()
        on_docs_saved(app_state, collection_name='test-collection')
        assert not hasattr(app_state, 'concept_graph_dirty')
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_no_op_when_no_store(caplog) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    try:
        app_state = SimpleNamespace()
        with caplog.at_level(logging.DEBUG):
            on_docs_saved(app_state, collection_name='test-collection')
        assert not hasattr(app_state, 'concept_graph_dirty')
        assert any(
            'concept_graph_dirty signal ignored: no store on app.state' in r.message
            for r in caplog.records
        )
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_sets_dirty_when_enabled_and_store_present() -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    try:
        app_state = SimpleNamespace(concept_graph_store=object(), concept_graph_dirty=False)
        on_docs_saved(app_state, collection_name='my-collection')
        assert app_state.concept_graph_dirty is True
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig


def test_swallows_exceptions() -> None:
    result = on_docs_saved(None, collection_name='test-collection')
    assert result is None


def test_logs_docs_count(caplog) -> None:
    orig = CONCEPT_GRAPH_ENABLED.value
    CONCEPT_GRAPH_ENABLED.value = True
    try:
        app_state = SimpleNamespace(concept_graph_store=object(), concept_graph_dirty=False)
        docs = [Document(page_content='x'), Document(page_content='y')]
        with caplog.at_level(logging.INFO):
            on_docs_saved(app_state, collection_name='my-collection', docs=docs)
        assert app_state.concept_graph_dirty is True
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any('my-collection' in msg for msg in info_messages)
        assert any('docs=2' in msg for msg in info_messages)
    finally:
        CONCEPT_GRAPH_ENABLED.value = orig
