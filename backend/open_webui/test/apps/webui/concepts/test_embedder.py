"""Unit tests for the acceptance-harness embedder helpers."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from open_webui.retrieval.concepts.schema import Concept, ConceptKind
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
from open_webui.test.apps.webui.concepts.embedder import (
    FAKE_EMBED_DIM,
    embed_store_concepts,
    fake_embedder,
)

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _concept(name: str) -> Concept:
    return Concept(
        id=0,
        name=name,
        kind=ConceptKind.ATOMIC,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=None,
        language_hint=None,
        original_tokens=(),
    )


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_fake_embedder_deterministic():
    embed = fake_embedder()
    a = embed("toolbar clipboard copy")
    b = embed("toolbar clipboard copy")
    assert a == b


def test_fake_embedder_different_tokens_different_vectors():
    embed = fake_embedder()
    a = embed("foo bar")
    b = embed("baz qux")
    assert a != b
    assert _dot(a, b) < 0.5


def test_fake_embedder_token_overlap_increases_similarity():
    embed = fake_embedder()
    ab = embed("foo bar")
    fb = embed("foo baz")
    xy = embed("xxx yyy")
    assert _dot(ab, fb) > _dot(ab, xy)


def test_fake_embedder_empty_text_zero_vector():
    embed = fake_embedder()
    for text in ("", "  ", "\t"):
        vec = embed(text)
        assert len(vec) == FAKE_EMBED_DIM
        assert all(v == 0.0 for v in vec)


def test_fake_embedder_l2_normalized():
    embed = fake_embedder()
    vec = embed("selection service dispatcher")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == 0.0 or abs(norm - 1.0) < 1e-6


def test_embed_store_concepts_populates_embeddings():
    store = InMemoryGraphStore()
    for name in ("toolbar", "clipboard", "selection"):
        store.upsert_concept(_concept(name))

    count = embed_store_concepts(store, fake_embedder())
    assert count == 3

    scores = store.pagerank()
    for concept_id in scores:
        concept = store.get_concept(concept_id)
        assert concept is not None
        assert concept.embedding is not None
        assert len(concept.embedding) == FAKE_EMBED_DIM


def test_embed_store_concepts_idempotent():
    store = InMemoryGraphStore()
    for name in ("alpha", "beta"):
        store.upsert_concept(_concept(name))

    embed_fn = fake_embedder()
    assert embed_store_concepts(store, embed_fn) == 2
    assert embed_store_concepts(store, embed_fn) == 0


def test_embed_store_concepts_overwrite_true():
    store = InMemoryGraphStore()
    for name in ("one", "two", "three"):
        store.upsert_concept(_concept(name))

    embed_fn = fake_embedder()
    assert embed_store_concepts(store, embed_fn) == 3
    assert embed_store_concepts(store, embed_fn, overwrite=True) == 3
