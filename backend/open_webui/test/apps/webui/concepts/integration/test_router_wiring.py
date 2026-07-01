"""Unit tests for ``open_webui.retrieval.concepts.integration.router_wiring``."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from open_webui.retrieval.concepts.integration.router_wiring import (
    build_concept_graph_extras,
    build_concept_graph_reranker,
    build_sync_concept_graph_embed_fn,
)
from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.concepts.schema import Concept, ConceptKind

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class _FakeEncoder:
    def __init__(self, vector_for: dict[str, list[float]]):
        self.vector_for = vector_for
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, normalize_embeddings: bool = False):
        self.calls.append((text, normalize_embeddings))

        class _Arr:
            def __init__(self, data: list[float]) -> None:
                self._d = data

            def tolist(self) -> list[float]:
                return self._d

        return _Arr(self.vector_for.get(text, [0.0, 0.0, 0.0]))


def _concept(
    name: str,
    *,
    concept_id: int = 1,
    embedding: tuple[float, ...] | None = None,
) -> Concept:
    return Concept(
        id=concept_id,
        name=name,
        kind=ConceptKind.ATOMIC,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=embedding,
        definition=None,
        language_hint=None,
        original_tokens=(name,),
    )


def _hit(
    name: str,
    *,
    concept_id: int = 1,
    score: float = 0.9,
    embedding: tuple[float, ...] | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        concept=_concept(name, concept_id=concept_id, embedding=embedding),
        artifact=None,
        score=score,
        provenance={'retriever': 'neighborhood'},
    )


def test_build_sync_concept_graph_embed_fn_returns_none_for_none_ef() -> None:
    assert build_sync_concept_graph_embed_fn(None) is None


def test_build_sync_concept_graph_embed_fn_returns_none_for_ef_without_encode() -> None:
    assert build_sync_concept_graph_embed_fn(SimpleNamespace()) is None


def test_build_sync_concept_graph_embed_fn_wraps_callable_ef() -> None:
    fake = _FakeEncoder({'hello': [0.1, 0.2, 0.3]})
    embed_fn = build_sync_concept_graph_embed_fn(fake)
    assert embed_fn is not None

    result = embed_fn('hello')
    assert isinstance(result, tuple)
    assert all(isinstance(x, float) for x in result)
    assert result == (0.1, 0.2, 0.3)
    assert fake.calls == [('hello', True)]


def test_build_concept_graph_reranker_returns_none_when_embed_fn_is_none() -> None:
    assert build_concept_graph_reranker(None) is None


def test_build_concept_graph_reranker_returns_callable_with_correct_signature() -> None:
    embed_fn = build_sync_concept_graph_embed_fn(_FakeEncoder({'q': [1.0, 0.0], 'alpha': [0.9, 0.1]}))
    reranker = build_concept_graph_reranker(embed_fn)
    assert reranker is not None

    hits = [_hit('alpha', concept_id=1)]
    result = reranker('q', hits)
    assert isinstance(result, list)
    assert len(result) == len(hits)


def test_build_concept_graph_reranker_uses_name_only_cosine_scorer() -> None:
    shared_embedding = (0.0, 1.0)
    hits = [
        _hit('worse_name', concept_id=1, embedding=shared_embedding),
        _hit('better_name', concept_id=2, embedding=shared_embedding),
    ]
    embed_calls: list[str] = []

    def embed_fn(text: str) -> tuple[float, ...]:
        embed_calls.append(text)
        if text == 'target':
            return (1.0, 0.0)
        if text == 'worse_name':
            return (0.5, 0.5)
        if text == 'better_name':
            return (1.0, 0.0)
        return (0.0, 0.0)

    reranker = build_concept_graph_reranker(embed_fn)
    assert reranker is not None

    result = reranker('target', hits)
    assert [h.concept.name for h in result] == ['better_name', 'worse_name']
    assert 'worse_name' in embed_calls
    assert 'better_name' in embed_calls
    assert 'target' in embed_calls


def test_build_concept_graph_extras_no_ef() -> None:
    assert build_concept_graph_extras(SimpleNamespace(ef=None)) == {
        'concept_graph_embed_fn': None,
        'concept_graph_reranker': None,
    }


def test_build_concept_graph_extras_with_ef() -> None:
    fake = _FakeEncoder({'probe': [0.4, 0.5, 0.6]})
    extras = build_concept_graph_extras(SimpleNamespace(ef=fake))

    assert extras['concept_graph_embed_fn'] is not None
    assert extras['concept_graph_reranker'] is not None
    assert extras['concept_graph_embed_fn']('probe') == (0.4, 0.5, 0.6)


def test_build_concept_graph_extras_handles_app_state_without_ef_attr() -> None:
    assert build_concept_graph_extras(object()) == {
        'concept_graph_embed_fn': None,
        'concept_graph_reranker': None,
    }
