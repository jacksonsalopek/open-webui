"""Tests for ``open_webui.retrieval.concepts.lifecycle.idf``.

Phase 1 wave-7: IDF replaces PageRank centrality as the primary tiebreaker
signal for retrieval. These tests pin the smoothing formula, the
artifact-concept pair iteration, and the cache lifecycle so future
refactors (e.g. Phase 2 kuzu-backed iteration) don't silently change
scores. Acceptance regressions on ``q08``/``q09`` are the integration-level
counterpart; this file pins the unit-level invariants.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from open_webui.retrieval.concepts.lifecycle.idf import (
    IdfScores,
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

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_artifact(aid: int, chunk_index: int) -> Artifact:
    return Artifact(
        id=aid,
        kind=ArtifactKind.CHUNK,
        path=f'art_{chunk_index}.cs',
        chunk_index=chunk_index,
        language='csharp',
        byte_start=None,
        byte_end=None,
        last_modified_at=_TS,
    )


def _make_atomic(cid: int, name: str) -> Concept:
    return Concept(
        id=cid,
        name=name,
        kind=ConceptKind.ATOMIC,
        first_seen_at=_TS,
        last_seen_at=_TS,
        centrality_score=None,
        embedding=None,
        definition=None,
        language_hint='csharp',
        original_tokens=(name,),
    )


def _make_store_with_artifacts(
    concepts: list[str],
    artifacts: int,
    df: dict[str, int],
) -> tuple[InMemoryGraphStore, dict[str, int]]:
    """Build a store where ``concept`` ``name`` is defined in ``df[name]``
    artifacts. Returns ``(store, {name: concept_id})``."""
    store = InMemoryGraphStore()
    name_to_id: dict[str, int] = {}
    for i, name in enumerate(concepts, start=1):
        cid = store.upsert_concept(_make_atomic(i, name))
        name_to_id[name] = cid

    artifact_ids: list[int] = []
    for j in range(artifacts):
        aid = store.upsert_artifact(_make_artifact(100 + j, j))
        artifact_ids.append(aid)

    for name, doc_freq in df.items():
        cid = name_to_id[name]
        for aid in artifact_ids[:doc_freq]:
            store.upsert_edge(
                edge_with_props(
                    src_id=aid,
                    dst_id=cid,
                    props=DefinesProps(count=1),
                ),
            )

    return store, name_to_id


def test_smoothed_idf_formula() -> None:
    """``log((N+1)/(df+1)) + 1`` — pinned to sklearn-style smoothing."""
    concepts = ['rare', 'common', 'everywhere']
    store, ids = _make_store_with_artifacts(
        concepts,
        artifacts=10,
        df={'rare': 1, 'common': 5, 'everywhere': 10},
    )

    scores = compute(store)
    assert scores.total_artifacts == 10
    assert scores.scores[ids['rare']] == pytest.approx(math.log(11 / 2) + 1)
    assert scores.scores[ids['common']] == pytest.approx(math.log(11 / 6) + 1)
    assert scores.scores[ids['everywhere']] == pytest.approx(math.log(11 / 11) + 1)


def test_rare_concept_outranks_common_concept() -> None:
    """The whole point of IDF: rarer concepts get higher scores than
    universally-present ones."""
    concepts = ['needle', 'haystack']
    store, ids = _make_store_with_artifacts(
        concepts,
        artifacts=20,
        df={'needle': 1, 'haystack': 20},
    )

    scores = compute(store)
    assert scores.scores[ids['needle']] > scores.scores[ids['haystack']]


def test_empty_store_returns_empty_scores() -> None:
    store = InMemoryGraphStore()
    scores = compute(store)
    assert scores.scores == {}
    assert scores.total_artifacts == 0


def test_compute_and_persist_round_trips_via_cache() -> None:
    store, ids = _make_store_with_artifacts(
        ['x'],
        artifacts=3,
        df={'x': 2},
    )
    assert get_cached(store) is None

    when = compute_and_persist(store)
    cached = get_cached(store)
    assert isinstance(cached, IdfScores)
    assert cached.computed_at == when
    # Only artifacts that touch an edge are counted (N=2, df=2 → score 1.0).
    expected = math.log((cached.total_artifacts + 1) / (2 + 1)) + 1.0
    assert cached.scores[ids['x']] == pytest.approx(expected)

    clear_cache(store)
    assert get_cached(store) is None


def test_clear_cache_global_evicts_all_stores() -> None:
    store_a, _ = _make_store_with_artifacts(['a'], artifacts=1, df={'a': 1})
    store_b, _ = _make_store_with_artifacts(['b'], artifacts=1, df={'b': 1})
    compute_and_persist(store_a)
    compute_and_persist(store_b)
    assert get_cached(store_a) is not None
    assert get_cached(store_b) is not None

    clear_cache()  # no argument → wipe everything
    assert get_cached(store_a) is None
    assert get_cached(store_b) is None


def test_only_artifact_touching_edges_counted() -> None:
    """``CO_OCCURS_WITH`` is concept-to-concept and must NOT contribute to df."""
    store = InMemoryGraphStore()
    alpha_id = store.upsert_concept(_make_atomic(1, 'alpha'))
    beta_id = store.upsert_concept(_make_atomic(2, 'beta'))
    art_id = store.upsert_artifact(_make_artifact(10, 0))

    store.upsert_edge(
        edge_with_props(
            src_id=art_id,
            dst_id=alpha_id,
            props=DefinesProps(count=1),
        ),
    )
    store.upsert_edge(
        edge_with_props(
            src_id=alpha_id,
            dst_id=beta_id,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )

    scores = compute(store)
    assert alpha_id in scores.scores  # alpha got the DEFINES edge
    assert beta_id not in scores.scores  # beta only had CO_OCCURS_WITH
