"""Tests for ``open_webui.retrieval.concepts.retrieve.reranker``."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pytest

from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.concepts.retrieve.reranker import (
    make_cosine_scorer,
    make_name_only_cosine_scorer,
    make_text_scorer,
    rerank_hits,
)
from open_webui.retrieval.concepts.schema import Artifact, ArtifactKind, Concept, ConceptKind

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


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


def _artifact(*, artifact_id: int = 1, path: str = '/src/foo.cs') -> Artifact:
    return Artifact(
        id=artifact_id,
        kind=ArtifactKind.CHUNK,
        path=path,
        chunk_index=0,
        language='csharp',
        byte_start=0,
        byte_end=100,
        last_modified_at=_TS,
    )


def _hit(
    name: str,
    *,
    concept_id: int = 1,
    score: float = 0.9,
    embedding: tuple[float, ...] | None = None,
    provenance: dict | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        concept=_concept(name, concept_id=concept_id, embedding=embedding),
        artifact=None,
        score=score,
        provenance=provenance or {'retriever': 'neighborhood'},
    )


def _artifact_hit(
    *,
    artifact_id: int = 1,
    path: str = '/src/foo.cs',
    score: float = 0.5,
    provenance: dict | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        concept=None,
        artifact=_artifact(artifact_id=artifact_id, path=path),
        score=score,
        provenance=provenance or {'retriever': 'hybrid'},
    )


def _fixed_scorer(scores: Sequence[float]):
    def scorer(_query: str, hits: Sequence[RetrievalHit]) -> Sequence[float]:
        return scores

    return scorer


def test_rerank_hits_empty_input_returns_empty() -> None:
    assert rerank_hits('q', [], scorer=_fixed_scorer([])) == []


def test_rerank_hits_reorders_by_score() -> None:
    hits = [_hit('a', concept_id=0), _hit('b', concept_id=1), _hit('c', concept_id=2)]
    result = rerank_hits('q', hits, scorer=_fixed_scorer([0.1, 0.9, 0.5]))

    assert [h.concept.name for h in result] == ['b', 'c', 'a']


def test_rerank_hits_stable_on_ties() -> None:
    hits = [_hit('first', concept_id=1), _hit('second', concept_id=2)]
    result = rerank_hits('q', hits, scorer=_fixed_scorer([0.5, 0.5]), stable=True)

    assert [h.concept.name for h in result] == ['first', 'second']


def test_rerank_hits_top_n_truncates() -> None:
    hits = [_hit(f'h{i}', concept_id=i) for i in range(5)]
    scores = [0.1, 0.9, 0.5, 0.3, 0.7]
    result = rerank_hits('q', hits, scorer=_fixed_scorer(scores), top_n=2)

    assert len(result) == 2
    assert [h.concept.name for h in result] == ['h1', 'h4']


def test_rerank_hits_top_n_none_returns_all() -> None:
    hits = [_hit(f'h{i}', concept_id=i) for i in range(5)]
    scores = [0.1, 0.9, 0.5, 0.3, 0.7]
    result = rerank_hits('q', hits, scorer=_fixed_scorer(scores), top_n=None)

    assert len(result) == 5
    assert [h.concept.name for h in result] == ['h1', 'h4', 'h2', 'h3', 'h0']


def test_rerank_hits_scorer_length_mismatch_raises() -> None:
    hits = [_hit('a'), _hit('b', concept_id=2)]

    with pytest.raises(ValueError, match='scorer returned 1 scores for 2 hits'):
        rerank_hits('q', hits, scorer=_fixed_scorer([0.5]))


def test_rerank_hits_scorer_exception_returns_input_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hits = [_hit('a'), _hit('b', concept_id=2)]

    def _boom(_query: str, _hits: Sequence[RetrievalHit]) -> Sequence[float]:
        raise RuntimeError('model unavailable')

    with caplog.at_level('WARNING'):
        result = rerank_hits('q', hits, scorer=_boom)

    assert result == hits
    assert any('rerank_hits scorer failed' in record.message for record in caplog.records)


def test_rerank_hits_preserves_provenance() -> None:
    prov_a = {'retriever': 'neighborhood', 'rank': 3}
    prov_b = {'retriever': 'hybrid', 'rank': 1}
    hits = [
        _hit('a', provenance=prov_a),
        _hit('b', concept_id=2, provenance=prov_b),
    ]
    result = rerank_hits('q', hits, scorer=_fixed_scorer([0.2, 0.8]))

    assert result[0].provenance == prov_b
    assert result[1].provenance == prov_a


def test_make_cosine_scorer_basic() -> None:
    hits = [
        _hit('aligned', embedding=(1.0, 0.0, 0.0)),
        _hit('orthogonal', concept_id=2, embedding=(0.0, 1.0, 0.0)),
        _hit('diagonal', concept_id=3, embedding=(0.5, 0.5, 0.0)),
    ]
    scorer = make_cosine_scorer(lambda _q: (1.0, 0.0, 0.0))
    scores = scorer('q', hits)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(0.707, abs=0.001)

    result = rerank_hits('q', hits, scorer=scorer)
    assert [h.concept.name for h in result] == ['aligned', 'diagonal', 'orthogonal']


def test_make_cosine_scorer_handles_zero_vector() -> None:
    hits = [_hit('zero', embedding=(0.0, 0.0, 0.0))]
    scorer = make_cosine_scorer(lambda _q: (1.0, 0.0, 0.0))

    assert scorer('q', hits)[0] == pytest.approx(0.0)


def test_make_cosine_scorer_handles_missing_embedding() -> None:
    hits = [
        _hit('with-embed', embedding=(1.0, 0.0)),
        _hit('no-embed', concept_id=2, embedding=None),
    ]
    scorer = make_cosine_scorer(lambda _q: (1.0, 0.0))
    result = rerank_hits('q', hits, scorer=scorer)

    assert result[0].concept.name == 'with-embed'
    assert result[1].concept.name == 'no-embed'


def test_make_cosine_scorer_with_artifact_hit_no_concept() -> None:
    hits = [
        _hit('concept', embedding=(1.0, 0.0)),
        _artifact_hit(path='/src/bar.cs'),
    ]
    scorer = make_cosine_scorer(lambda _q: (1.0, 0.0))
    result = rerank_hits('q', hits, scorer=scorer)

    assert result[0].concept is not None
    assert result[1].artifact is not None


def test_make_text_scorer_basic() -> None:
    hits = [_hit('alpha'), _hit('beta', concept_id=2)]

    def cross_encoder(_query: str, texts: Sequence[str]) -> Sequence[float]:
        return [0.3, 0.8]

    scorer = make_text_scorer(cross_encoder)
    assert scorer('q', hits) == [0.3, 0.8]


def test_make_text_scorer_custom_text_fn() -> None:
    hits = [_hit('alpha'), _hit('beta', concept_id=2)]
    captured: list[Sequence[str]] = []

    def cross_encoder(_query: str, texts: Sequence[str]) -> Sequence[float]:
        captured.append(list(texts))
        return [0.5, 0.5]

    scorer = make_text_scorer(
        cross_encoder,
        text_fn=lambda h: h.concept.name.upper() if h.concept else '',
    )
    scorer('q', hits)

    assert captured == [['ALPHA', 'BETA']]


def test_make_text_scorer_default_text_fn_for_concept() -> None:
    hits = [_hit('toolbar')]
    captured: list[Sequence[str]] = []

    def cross_encoder(_query: str, texts: Sequence[str]) -> Sequence[float]:
        captured.append(list(texts))
        return [0.9]

    scorer = make_text_scorer(cross_encoder)
    scorer('q', hits)

    assert captured == [['toolbar']]


def test_make_text_scorer_default_text_fn_for_artifact() -> None:
    hits = [_artifact_hit(path='/docs/readme.md')]
    captured: list[Sequence[str]] = []

    def cross_encoder(_query: str, texts: Sequence[str]) -> Sequence[float]:
        captured.append(list(texts))
        return [0.4]

    scorer = make_text_scorer(cross_encoder)
    scorer('q', hits)

    assert captured == [['/docs/readme.md']]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_make_name_only_cosine_scorer_re_embeds_concept_names() -> None:
    hits = [
        _hit('toolbar', concept_id=1, embedding=None),
        _hit('clipboard', concept_id=2, embedding=None),
    ]
    embed_calls: list[str] = []

    def query_embed_fn(text: str) -> tuple[float, ...]:
        embed_calls.append(text)
        if text == 'query':
            return (1.0, 0.0)
        if text == 'toolbar':
            return (0.9, 0.1)
        if text == 'clipboard':
            return (0.1, 0.9)
        return (0.0, 0.0)

    scorer = make_name_only_cosine_scorer(query_embed_fn=query_embed_fn)
    scores = scorer('query', hits)

    assert embed_calls == ['query', 'toolbar', 'clipboard']
    assert scores[0] == pytest.approx(_cosine((1.0, 0.0), (0.9, 0.1)))
    assert scores[1] == pytest.approx(_cosine((1.0, 0.0), (0.1, 0.9)))


def test_make_name_only_cosine_scorer_artifact_hits_get_neg_inf() -> None:
    hits = [_artifact_hit(path='/src/bar.cs')]
    scorer = make_name_only_cosine_scorer(query_embed_fn=lambda _q: (1.0, 0.0))

    assert scorer('q', hits)[0] == float('-inf')


def test_make_name_only_cosine_scorer_deterministic() -> None:
    hits = [
        _hit('alpha', concept_id=1, embedding=None),
        _hit('beta', concept_id=2, embedding=None),
    ]

    def query_embed_fn(text: str) -> tuple[float, ...]:
        return (float(len(text)), 0.0)

    scorer = make_name_only_cosine_scorer(query_embed_fn=query_embed_fn)
    first = scorer('query', hits)
    second = scorer('query', hits)

    assert first == second


def test_make_name_only_cosine_scorer_differs_from_make_cosine_scorer_when_embedding_is_rich() -> None:
    rich_embedding = (0.0, 1.0)
    name_embedding = (1.0, 0.0)
    hits = [_hit('alpha', embedding=rich_embedding)]

    def query_embed_fn(text: str) -> tuple[float, ...]:
        if text == 'alpha':
            return name_embedding
        return (1.0, 0.0)

    name_only_scorer = make_name_only_cosine_scorer(query_embed_fn=query_embed_fn)
    cosine_scorer = make_cosine_scorer(query_embed_fn=query_embed_fn)

    name_only_score = name_only_scorer('query', hits)[0]
    cosine_score = cosine_scorer('query', hits)[0]

    assert name_only_score == pytest.approx(_cosine((1.0, 0.0), name_embedding))
    assert cosine_score == pytest.approx(_cosine((1.0, 0.0), rich_embedding))
    assert name_only_score != cosine_score
