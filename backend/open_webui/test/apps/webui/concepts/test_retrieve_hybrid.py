"""Tests for ``open_webui.retrieval.concepts.retrieve.hybrid``."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType

import pytest

from open_webui.retrieval.concepts.retrieve.base import RetrievalQuery
from open_webui.retrieval.concepts.retrieve.hybrid import (
    HybridRetriever,
    HybridRetrieverConfig,
    _min_max_normalize,
)
from open_webui.retrieval.concepts.schema import (
    CoOccursWithProps,
    Concept,
    ConceptKind,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
from open_webui.retrieval.concepts.store.protocol import GraphStore

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class _CentralityScores:
    semantic: dict[int, float]
    structural: dict[int, float]


def _install_fake_centrality_module(*, scores: _CentralityScores | None) -> ModuleType:
    module = ModuleType('open_webui.retrieval.concepts.lifecycle.centrality')
    module.get_cached = lambda store: scores  # type: ignore[attr-defined]
    package = ModuleType('open_webui.retrieval.concepts.lifecycle')
    package.centrality = module  # type: ignore[attr-defined]
    sys.modules['open_webui.retrieval.concepts.lifecycle'] = package
    sys.modules['open_webui.retrieval.concepts.lifecycle.centrality'] = module
    return module


def _concept(
    name: str,
    *,
    kind: ConceptKind = ConceptKind.ATOMIC,
    embedding: tuple[float, ...] | None = None,
) -> Concept:
    return Concept(
        id=0,
        name=name,
        kind=kind,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=embedding,
        definition=(
            'A curated phrase concept.'
            if kind == ConceptKind.PHRASE
            else None
        ),
        language_hint=None,
        original_tokens=(name,),
    )


def _upsert(store: GraphStore, concept: Concept) -> int:
    return store.upsert_concept(concept)


def _link(store: GraphStore, src_id: int, dst_id: int) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=src_id,
            dst_id=dst_id,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


def test_raises_when_embedding_missing(store: InMemoryGraphStore) -> None:
    retriever = HybridRetriever()
    query = RetrievalQuery(text='find something', embedding=None)

    with pytest.raises(ValueError, match='requires query.embedding'):
        retriever.retrieve(query, store)


def test_vector_phase_only_when_centrality_weight_zero(store: InMemoryGraphStore) -> None:
    a = _upsert(store, _concept('alpha', embedding=(1.0, 0.0)))
    b = _upsert(store, _concept('beta', embedding=(0.9, 0.1)))
    _link(store, a, b)

    retriever = HybridRetriever(
        HybridRetrieverConfig(
            vector_top_k=2,
            centrality_weight=0.0,
            neighborhood_weight=0.0,
            vector_weight=1.0,
        ),
    )
    query = RetrievalQuery(text='alpha', embedding=(1.0, 0.0), top_k=2)
    hits = retriever.retrieve(query, store)

    assert [hit.concept.id for hit in hits if hit.concept] == [a, b]
    assert hits[0].provenance['sources'] == ['vector']


def test_neighborhood_expansion_adds_adjacent_candidates(store: InMemoryGraphStore) -> None:
    a = _upsert(store, _concept('alpha', embedding=(1.0, 0.0)))
    b = _upsert(store, _concept('beta', embedding=(0.0, 1.0)))
    c = _upsert(store, _concept('gamma', embedding=(0.0, 0.9)))
    _link(store, a, b)

    retriever = HybridRetriever(
        HybridRetrieverConfig(
            vector_top_k=1,
            neighborhood_weight=0.5,
            vector_weight=0.5,
            centrality_weight=0.0,
        ),
    )
    query = RetrievalQuery(text='alpha', embedding=(1.0, 0.0), top_k=5)
    hits = retriever.retrieve(query, store)

    hit_ids = {hit.concept.id for hit in hits if hit.concept}
    assert a in hit_ids
    assert b in hit_ids
    assert c not in hit_ids


def test_centrality_boost_when_available(store: InMemoryGraphStore) -> None:
    low = _upsert(store, _concept('low', embedding=(1.0, 0.0)))
    high = _upsert(store, _concept('high', embedding=(0.95, 0.05)))

    import open_webui.retrieval.concepts.retrieve.hybrid as hybrid_module

    hybrid_module._centrality_warned = False
    _install_fake_centrality_module(
        scores=_CentralityScores(
            semantic={low: 0.1, high: 1.0},
            structural={low: 0.1, high: 1.0},
        ),
    )

    retriever = HybridRetriever(
        HybridRetrieverConfig(
            vector_top_k=2,
            neighborhood_weight=0.0,
            vector_weight=0.2,
            centrality_weight=0.8,
        ),
    )
    query = RetrievalQuery(text='query', embedding=(1.0, 0.0), top_k=2)
    hits = retriever.retrieve(query, store)

    assert hits[0].concept is not None
    assert hits[0].concept.id == high


def test_centrality_silently_skipped_when_unavailable(
    store: InMemoryGraphStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import open_webui.retrieval.concepts.retrieve.hybrid as hybrid_module

    hybrid_module._centrality_warned = False
    _upsert(store, _concept('only', embedding=(1.0, 0.0)))

    retriever = HybridRetriever(
        HybridRetrieverConfig(
            vector_top_k=1,
            neighborhood_weight=0.0,
            centrality_weight=0.5,
            vector_weight=0.5,
        ),
    )
    query = RetrievalQuery(text='query', embedding=(1.0, 0.0), top_k=1)

    _install_fake_centrality_module(scores=None)
    with caplog.at_level(logging.WARNING):
        hits = retriever.retrieve(query, store)
        hits_again = retriever.retrieve(query, store)

    assert hits
    warning_records = [record for record in caplog.records if record.levelname == 'WARNING']
    assert len(warning_records) == 1
    assert hits_again


def test_kind_filter_propagates_to_vector_search(store: InMemoryGraphStore) -> None:
    atomic = _upsert(store, _concept('atomic', embedding=(1.0, 0.0)))
    phrase = _upsert(
        store,
        _concept('race-condition', kind=ConceptKind.PHRASE, embedding=(0.99, 0.01)),
    )

    retriever = HybridRetriever(
        HybridRetrieverConfig(
            vector_top_k=5,
            neighborhood_weight=0.0,
            centrality_weight=0.0,
            vector_weight=1.0,
        ),
    )
    query = RetrievalQuery(
        text='race condition',
        embedding=(1.0, 0.0),
        top_k=5,
        kind_filter=(ConceptKind.PHRASE,),
    )
    hits = retriever.retrieve(query, store)

    assert len(hits) == 1
    assert hits[0].concept is not None
    assert hits[0].concept.id == phrase
    assert atomic not in {hit.concept.id for hit in hits if hit.concept}


def test_top_k_truncation(store: InMemoryGraphStore) -> None:
    for index in range(5):
        _upsert(store, _concept(f'c{index}', embedding=(1.0, float(index) * 0.01)))

    retriever = HybridRetriever(
        HybridRetrieverConfig(
            vector_top_k=5,
            neighborhood_weight=0.0,
            centrality_weight=0.0,
        ),
    )
    query = RetrievalQuery(text='query', embedding=(1.0, 0.0), top_k=2)
    hits = retriever.retrieve(query, store)

    assert len(hits) == 2


def test_deterministic_given_same_store_and_query(store: InMemoryGraphStore) -> None:
    _upsert(store, _concept('a', embedding=(1.0, 0.0)))
    _upsert(store, _concept('b', embedding=(0.8, 0.2)))
    _upsert(store, _concept('c', embedding=(0.6, 0.4)))

    retriever = HybridRetriever(
        HybridRetrieverConfig(
            vector_top_k=3,
            neighborhood_weight=0.0,
            centrality_weight=0.0,
        ),
    )
    query = RetrievalQuery(text='query', embedding=(1.0, 0.0), top_k=3)

    first = retriever.retrieve(query, store)
    second = retriever.retrieve(query, store)

    assert [(hit.concept.id if hit.concept else None, hit.score) for hit in first] == [
        (hit.concept.id if hit.concept else None, hit.score) for hit in second
    ]


def test_min_max_normalize_single_candidate_and_all_zero() -> None:
    assert _min_max_normalize({1: 0.0}) == {1: 0.0}
    assert _min_max_normalize({1: 5.0}) == {1: 1.0}
    assert _min_max_normalize({1: 2.0, 2: 2.0}) == {1: 1.0, 2: 1.0}
