"""Tests for ``open_webui.retrieval.concepts.lifecycle.centrality``."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from open_webui.retrieval.concepts.extraction.stopwords import is_stopword
from open_webui.retrieval.concepts.lifecycle.centrality import (
    CentralityScores,
    clear_cache,
    compute,
    compute_and_persist,
    get_cached,
)
from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    CoOccursWithProps,
    DefinesProps,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _upsert_concept(store: InMemoryGraphStore, name: str) -> int:
    return store.upsert_concept(
        Concept(
            id=0,
            name=name,
            kind=ConceptKind.ATOMIC,
            first_seen_at=_TS,
            last_seen_at=_TS,
            centrality_score=None,
            embedding=None,
            definition=None,
            language_hint=None,
            original_tokens=(name,),
        ),
    )


def _upsert_artifact(store: InMemoryGraphStore, path: str) -> int:
    return store.upsert_artifact(
        Artifact(
            id=0,
            kind=ArtifactKind.CHUNK,
            path=path,
            chunk_index=0,
            language='csharp',
            byte_start=0,
            byte_end=100,
            last_modified_at=_TS,
        ),
    )


def _link_co(store: InMemoryGraphStore, src: int, dst: int) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=src,
            dst_id=dst,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )


def _link_defines(store: InMemoryGraphStore, artifact_id: int, concept_id: int) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=artifact_id,
            dst_id=concept_id,
            props=DefinesProps(count=1),
        ),
    )


def test_compute_returns_both_scores() -> None:
    store = InMemoryGraphStore()
    hub = _upsert_concept(store, 'hub')
    for name in ('alpha', 'beta', 'gamma'):
        spoke = _upsert_concept(store, name)
        _link_co(store, spoke, hub)

    artifact = _upsert_artifact(store, '/a.cs')
    _link_defines(store, artifact, hub)

    scores = compute(store)
    assert scores.semantic
    assert scores.structural
    assert abs(sum(scores.semantic.values()) - 1.0) < 1e-6
    assert abs(sum(scores.structural.values()) - 1.0) < 1e-6


def test_compute_semantic_filters_to_cooccurrence() -> None:
    store = InMemoryGraphStore()
    anchor = _upsert_concept(store, 'anchor')
    for i in range(6):
        artifact = _upsert_artifact(store, f'/def{i}.cs')
        _link_defines(store, artifact, anchor)

    hub = _upsert_concept(store, 'hub')
    for name in ('s1', 's2', 's3', 's4', 's5'):
        spoke = _upsert_concept(store, name)
        _link_co(store, spoke, hub)
        _link_co(store, hub, spoke)

    scores = compute(store, iterations=30)
    n = len(scores.semantic)
    baseline = 1.0 / n

    assert scores.semantic[hub] > baseline * 2
    assert scores.semantic[anchor] < scores.semantic[hub]


def test_compute_structural_filters_to_defines_references() -> None:
    store = InMemoryGraphStore()
    hub = _upsert_concept(store, 'hub')
    for name in ('s1', 's2', 's3', 's4', 's5'):
        spoke = _upsert_concept(store, name)
        _link_co(store, spoke, hub)
        _link_co(store, hub, spoke)

    anchor = _upsert_concept(store, 'anchor')
    for i in range(6):
        artifact = _upsert_artifact(store, f'/ref{i}.cs')
        _link_defines(store, artifact, anchor)

    scores = compute(store, iterations=30)
    n = len(scores.structural)
    baseline = 1.0 / n

    assert scores.structural[hub] == pytest.approx(baseline, rel=0.15)
    assert scores.semantic[hub] > baseline * 2


def test_compute_and_persist_caches_results() -> None:
    store = InMemoryGraphStore()
    a = _upsert_concept(store, 'a')
    b = _upsert_concept(store, 'b')
    _link_co(store, a, b)

    compute_and_persist(store)
    cached = get_cached(store)
    assert cached is not None
    assert isinstance(cached, CentralityScores)
    assert cached.semantic == get_cached(store).semantic


def test_compute_and_persist_idempotent() -> None:
    store = InMemoryGraphStore()
    a = _upsert_concept(store, 'a')
    b = _upsert_concept(store, 'b')
    _link_co(store, a, b)

    first_at = compute_and_persist(store)
    first_scores = get_cached(store)
    assert first_scores is not None

    time.sleep(0.01)
    second_at = compute_and_persist(store)
    second_scores = get_cached(store)
    assert second_scores is not None
    assert second_at >= first_at
    assert second_scores.computed_at >= first_scores.computed_at


def test_clear_cache_evicts() -> None:
    store = InMemoryGraphStore()
    _upsert_concept(store, 'solo')
    compute_and_persist(store)
    assert get_cached(store) is not None

    clear_cache(store)
    assert get_cached(store) is None


def test_lollipop_semantic_centrality_top5_quality(
    lollipop_subset_store: InMemoryGraphStore,
) -> None:
    """The top-5 concepts by semantic centrality should include at least one
    Lollipop domain term, validating that centrality surfaces signal over noise.

    Domain terms match concept names containing view/service/helper/extension/
    command/model substrings, or canonical glossary names from the Lollipop
    ViewModels/Services/Extensions corpus.

    Replaces the legacy 3-file-fallback version of this test.
    """
    store = lollipop_subset_store
    compute_and_persist(store)  # idempotent; builder already computed centrality

    scores = get_cached(store)
    assert scores is not None

    top5 = sorted(scores.semantic.items(), key=lambda kv: -kv[1])[:5]
    top5_names = [store.get_concept(cid).name for cid, _ in top5]  # type: ignore[union-attr]

    classified_stopword_hits = {
        name for name in top5_names if is_stopword(name, language='csharp')
    }
    assert not classified_stopword_hits, (
        f'semantic top-5 contains classified stopwords: {classified_stopword_hits}'
    )

    domain_substrings = (
        'view',
        'service',
        'helper',
        'extension',
        'command',
        'model',
        'toolbar',
        'selection',
    )
    matches = [
        n for n in top5_names if any(s in n.lower() for s in domain_substrings)
    ]
    assert len(matches) >= 1, (
        f'expected at least 1 domain-flavored concept in top-5 semantic '
        f'centrality; got {top5_names}'
    )
