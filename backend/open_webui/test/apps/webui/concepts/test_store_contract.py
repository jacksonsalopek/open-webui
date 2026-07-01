"""Parameterized contract tests for ``GraphStore`` implementations.

The ``store_impl`` fixture is the substrate-swap guarantee: when step 6
lands ``KuzuGraphStore``, delete the ``pytest.skip`` below and both
implementations must pass this entire suite unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    CoOccursWithProps,
    DefinesProps,
    EdgeType,
    IsNamedInProps,
    ReferencesProps,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore
from open_webui.retrieval.concepts.store.protocol import GraphStore

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_TS2 = datetime(2025, 6, 2, 8, 30, 0, tzinfo=timezone.utc)
_TS3 = datetime(2025, 6, 3, 15, 45, 0, tzinfo=timezone.utc)


def _artifact(
    path: str,
    *,
    kind: ArtifactKind = ArtifactKind.CHUNK,
    chunk_index: int | None = 0,
    language: str | None = 'csharp',
    byte_start: int | None = 0,
    byte_end: int | None = 100,
    last_modified_at: datetime = _TS,
) -> Artifact:
    return Artifact(
        id=0,
        kind=kind,
        path=path,
        chunk_index=chunk_index,
        language=language,
        byte_start=byte_start,
        byte_end=byte_end,
        last_modified_at=last_modified_at,
    )


def _concept(
    name: str,
    *,
    kind: ConceptKind = ConceptKind.ATOMIC,
    tokens: tuple[str, ...] = (),
    embedding: tuple[float, ...] | None = None,
) -> Concept:
    return Concept(
        id=0,
        name=name,
        kind=kind,
        first_seen_at=_TS,
        last_seen_at=_TS2,
        centrality_score=None,
        embedding=embedding,
        definition=(
            'A defect where outcome depends on timing.'
            if kind == ConceptKind.PHRASE
            else None
        ),
        language_hint=None,
        original_tokens=tokens,
    )


def _upsert(store: GraphStore, concept: Concept) -> int:
    return store.upsert_concept(concept)


def _link(
    store: GraphStore,
    src_id: int,
    dst_id: int,
    edge_type: EdgeType = EdgeType.CO_OCCURS_WITH,
    **props: object,
) -> None:
    if edge_type == EdgeType.DEFINES:
        edge = edge_with_props(
            src_id=src_id,
            dst_id=dst_id,
            props=DefinesProps(count=int(props.get('count', 1))),
        )
    elif edge_type == EdgeType.CO_OCCURS_WITH:
        edge = edge_with_props(
            src_id=src_id,
            dst_id=dst_id,
            props=CoOccursWithProps(
                weight=float(props.get('weight', 1.0)),
                chunk_count=int(props.get('chunk_count', 1)),
            ),
        )
    elif edge_type == EdgeType.IS_CANONICAL_ALIAS_OF:
        from open_webui.retrieval.concepts.schema import IsCanonicalAliasOfProps

        edge = edge_with_props(
            src_id=src_id,
            dst_id=dst_id,
            props=IsCanonicalAliasOfProps(introduced_at=_TS),
        )
    elif edge_type == EdgeType.IS_NAMED_IN:
        edge = edge_with_props(
            src_id=src_id,
            dst_id=dst_id,
            props=IsNamedInProps(first_seen_at=props.get('first_seen_at', _TS)),
        )
    else:
        raise ValueError(f'unsupported edge type in test helper: {edge_type!r}')
    store.upsert_edge(edge)


def _link_concept_artifact(
    store: GraphStore,
    concept_id: int,
    artifact_id: int,
    edge_type: EdgeType,
    **props: object,
) -> None:
    if edge_type == EdgeType.IS_NAMED_IN:
        _link(store, concept_id, artifact_id, edge_type=EdgeType.IS_NAMED_IN, **props)
    elif edge_type == EdgeType.DEFINES:
        store.upsert_edge(
            edge_with_props(
                src_id=artifact_id,
                dst_id=concept_id,
                props=DefinesProps(count=int(props.get('count', 1))),
            ),
        )
    elif edge_type == EdgeType.REFERENCES:
        store.upsert_edge(
            edge_with_props(
                src_id=artifact_id,
                dst_id=concept_id,
                props=ReferencesProps(
                    count=int(props.get('count', 1)),
                    positions=props.get('positions'),  # type: ignore[arg-type]
                ),
            ),
        )
    else:
        raise ValueError(f'unsupported concept-artifact edge type: {edge_type!r}')


def _co_occurs_props(
    store: GraphStore,
    src_id: int,
    dst_id: int,
) -> dict[str, object]:
    if isinstance(store, InMemoryGraphStore):
        edge = store._edges.get((EdgeType.CO_OCCURS_WITH, src_id, dst_id))
        assert edge is not None
        return dict(edge.properties)
    from open_webui.retrieval.concepts.store.kuzu_store import KuzuGraphStore

    if isinstance(store, KuzuGraphStore):
        props = store._fetch_edge_properties(EdgeType.CO_OCCURS_WITH, src_id, dst_id)
        assert props is not None
        return props
    raise TypeError(f'unsupported store type: {type(store)!r}')


@pytest.fixture(params=['memory', 'kuzu'])
def store_impl(request: pytest.FixtureRequest, tmp_path) -> GraphStore:
    if request.param == 'memory':
        return InMemoryGraphStore()
    elif request.param == 'kuzu':
        from open_webui.retrieval.concepts.store.kuzu_store import KuzuGraphStore

        return KuzuGraphStore(tmp_path / 'test.kuzu', embedding_dim=16)


def test_upsert_concept_assigns_id(store_impl: GraphStore) -> None:
    first_id = _upsert(store_impl, _concept('toolbar'))
    second_id = _upsert(store_impl, _concept('toolbar'))
    phrase_id = _upsert(store_impl, _concept('race-condition', kind=ConceptKind.PHRASE))

    assert first_id > 0
    assert second_id == first_id
    assert phrase_id != first_id


def test_upsert_concept_merges_tokens(store_impl: GraphStore) -> None:
    _upsert(store_impl, _concept('toolbar', tokens=('Toolbar',)))
    concept_id = _upsert(store_impl, _concept('toolbar', tokens=('toolbar', 'ToolBar')))

    stored = store_impl.get_concept(concept_id)
    assert stored is not None
    assert stored.original_tokens == ('Toolbar', 'toolbar', 'ToolBar')


def test_upsert_concept_phrase_requires_definition(store_impl: GraphStore) -> None:
    with pytest.raises(ValueError, match='definition must be set iff kind is PHRASE'):
        _upsert(
            store_impl,
            Concept(
                id=0,
                name='race-condition',
                kind=ConceptKind.PHRASE,
                first_seen_at=_TS,
                last_seen_at=_TS2,
                centrality_score=None,
                embedding=None,
                definition=None,
                language_hint=None,
                original_tokens=(),
            ),
        )


def test_upsert_edge_idempotent(store_impl: GraphStore) -> None:
    a = _upsert(store_impl, _concept('alpha'))
    b = _upsert(store_impl, _concept('beta'))
    store_impl.upsert_edge(
        edge_with_props(
            src_id=a,
            dst_id=b,
            props=CoOccursWithProps(weight=1.0, chunk_count=2),
        ),
    )
    store_impl.upsert_edge(
        edge_with_props(
            src_id=a,
            dst_id=b,
            props=CoOccursWithProps(weight=1.0, chunk_count=3),
        ),
    )

    path = store_impl.shortest_path(
        a,
        b,
        edge_types=[EdgeType.CO_OCCURS_WITH],
        max_hops=1,
    )
    assert [c.id for c in path] == [a, b]


def test_neighborhood_radius_1(store_impl: GraphStore) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    n1 = _upsert(store_impl, _concept('n1'))
    n2 = _upsert(store_impl, _concept('n2'))
    n3 = _upsert(store_impl, _concept('n3'))
    for neighbor in (n1, n2, n3):
        _link(store_impl, anchor, neighbor)

    neighbors = store_impl.neighborhood(anchor, radius=1, limit=10)
    assert {c.id for c in neighbors} == {n1, n2, n3}


def test_neighborhood_radius_2_filtered_by_edge_type(store_impl: GraphStore) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    mid = _upsert(store_impl, _concept('mid'))
    far = _upsert(store_impl, _concept('far'))
    other = _upsert(store_impl, _concept('other'))

    _link(store_impl, anchor, mid, edge_type=EdgeType.CO_OCCURS_WITH)
    _link(store_impl, mid, far, edge_type=EdgeType.CO_OCCURS_WITH)
    artifact_id = store_impl.upsert_artifact(_artifact('/decoy.cs'))
    store_impl.upsert_edge(
        edge_with_props(
            src_id=artifact_id,
            dst_id=other,
            props=DefinesProps(count=1),
        ),
    )

    co_only = store_impl.neighborhood(
        anchor,
        radius=2,
        edge_types=[EdgeType.CO_OCCURS_WITH],
        limit=10,
    )
    assert {c.id for c in co_only} == {mid, far}
    assert other not in {c.id for c in co_only}


def test_neighborhood_respects_budget(store_impl: GraphStore) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    for i in range(8):
        neighbor = _upsert(store_impl, _concept(f'n{i}'))
        _link(store_impl, anchor, neighbor)

    neighbors = store_impl.neighborhood(anchor, radius=1, limit=5)
    assert len(neighbors) <= 5


def test_neighborhood_resolves_alias_anchor(store_impl: GraphStore) -> None:
    canonical = _upsert(store_impl, _concept('canonical'))
    alias = _upsert(store_impl, _concept('alias-name'))
    neighbor = _upsert(store_impl, _concept('neighbor'))
    _link(store_impl, alias, canonical, edge_type=EdgeType.IS_CANONICAL_ALIAS_OF)
    _link(store_impl, canonical, neighbor)

    neighbors = store_impl.neighborhood(alias, radius=1, limit=10)
    assert {c.id for c in neighbors} == {neighbor}


def test_shortest_path_finds_known_path(store_impl: GraphStore) -> None:
    a = _upsert(store_impl, _concept('a'))
    b = _upsert(store_impl, _concept('b'))
    c = _upsert(store_impl, _concept('c'))
    _link(store_impl, a, b)
    _link(store_impl, b, c)

    path = store_impl.shortest_path(a, c, max_hops=3)
    assert [node.id for node in path] == [a, b, c]


def test_shortest_path_returns_empty_when_disconnected(store_impl: GraphStore) -> None:
    a = _upsert(store_impl, _concept('a'))
    b = _upsert(store_impl, _concept('b'))
    assert store_impl.shortest_path(a, b, max_hops=3) == []


def test_pagerank_deterministic(store_impl: GraphStore) -> None:
    hub = _upsert(store_impl, _concept('hub'))
    for name in ('s1', 's2', 's3'):
        spoke = _upsert(store_impl, _concept(name))
        _link(store_impl, spoke, hub)

    first = store_impl.pagerank(edge_types=[EdgeType.CO_OCCURS_WITH], iterations=20)
    second = store_impl.pagerank(edge_types=[EdgeType.CO_OCCURS_WITH], iterations=20)
    assert first == second


def test_pagerank_concentrates_on_hub(store_impl: GraphStore) -> None:
    hub = _upsert(store_impl, _concept('hub'))
    spoke_ids: list[int] = []
    for name in ('s1', 's2', 's3', 's4', 's5'):
        spoke = _upsert(store_impl, _concept(name))
        spoke_ids.append(spoke)
        _link(store_impl, spoke, hub)

    scores = store_impl.pagerank(edge_types=[EdgeType.CO_OCCURS_WITH], iterations=30)
    hub_score = scores[hub]
    assert all(scores[s] < hub_score for s in spoke_ids)


def test_vector_search_ranks_by_similarity(store_impl: GraphStore) -> None:
    ids: list[int] = []
    embeddings = [
        (1.0, 0.0, 0.0),
        (0.9, 0.1, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.7, 0.7, 0.0),
    ]
    for i, emb in enumerate(embeddings):
        ids.append(_upsert(store_impl, _concept(f'c{i}', embedding=emb)))

    query = (1.0, 0.0, 0.0)
    results = store_impl.vector_search(query, limit=5)
    ranked_ids = [concept.id for concept, _score in results]
    assert ranked_ids[0] == ids[0]
    assert ranked_ids[1] == ids[1]
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_vector_search_filters_by_kind(store_impl: GraphStore) -> None:
    atomic_id = _upsert(
        store_impl,
        _concept('atomic-term', kind=ConceptKind.ATOMIC, embedding=(1.0, 0.0)),
    )
    _upsert(
        store_impl,
        _concept('phrase-term', kind=ConceptKind.PHRASE, embedding=(1.0, 0.0)),
    )

    results = store_impl.vector_search(
        (1.0, 0.0),
        kind=ConceptKind.ATOMIC,
        limit=5,
    )
    assert len(results) == 1
    assert results[0][0].id == atomic_id


def test_resolve_alias_follows_chain(store_impl: GraphStore) -> None:
    c = _upsert(store_impl, _concept('canonical'))
    b = _upsert(store_impl, _concept('middle'))
    a = _upsert(store_impl, _concept('alias'))
    _link(store_impl, a, b, edge_type=EdgeType.IS_CANONICAL_ALIAS_OF)
    _link(store_impl, b, c, edge_type=EdgeType.IS_CANONICAL_ALIAS_OF)

    assert store_impl.resolve_alias(a) == c


def test_resolve_alias_detects_cycle(store_impl: GraphStore) -> None:
    a = _upsert(store_impl, _concept('a'))
    b = _upsert(store_impl, _concept('b'))
    _link(store_impl, a, b, edge_type=EdgeType.IS_CANONICAL_ALIAS_OF)
    _link(store_impl, b, a, edge_type=EdgeType.IS_CANONICAL_ALIAS_OF)

    with pytest.raises(RuntimeError, match=str(a)):
        store_impl.resolve_alias(a)
    with pytest.raises(RuntimeError, match=str(b)):
        store_impl.resolve_alias(b)


def test_resolve_alias_noop_on_canonical(store_impl: GraphStore) -> None:
    canonical = _upsert(store_impl, _concept('canonical'))
    assert store_impl.resolve_alias(canonical) == canonical


def test_transaction_rollback_restores_state(store_impl: GraphStore) -> None:
    before_id = _upsert(store_impl, _concept('before'))
    tx = store_impl.begin_transaction()
    tx.__enter__()
    during_id = _upsert(store_impl, _concept('during'))
    tx.rollback()

    assert store_impl.get_concept(before_id) is not None
    assert store_impl.get_concept(during_id) is None


def test_transaction_commit_persists_state(store_impl: GraphStore) -> None:
    _upsert(store_impl, _concept('before'))
    tx = store_impl.begin_transaction()
    tx.__enter__()
    new_id = _upsert(store_impl, _concept('during'))
    tx.commit()

    assert store_impl.get_concept(new_id) is not None


def test_upsert_artifact_assigns_id(store_impl: GraphStore) -> None:
    artifact_id = store_impl.upsert_artifact(_artifact('/src/Foo.cs', chunk_index=1))
    assert artifact_id >= 1


def test_upsert_artifact_idempotent_on_path_and_chunk_index(
    store_impl: GraphStore,
) -> None:
    first_id = store_impl.upsert_artifact(_artifact('/src/Bar.cs', chunk_index=2))
    second_id = store_impl.upsert_artifact(
        _artifact('/src/Bar.cs', chunk_index=2, last_modified_at=_TS2),
    )
    assert second_id == first_id


def test_upsert_artifact_updates_mutable_fields(store_impl: GraphStore) -> None:
    artifact_id = store_impl.upsert_artifact(_artifact('/src/Baz.cs', chunk_index=4))
    store_impl.upsert_artifact(
        _artifact(
            '/src/Baz.cs',
            chunk_index=4,
            language='python',
            byte_start=10,
            byte_end=200,
            last_modified_at=_TS3,
        ),
    )

    stored = store_impl.get_artifact(artifact_id)
    assert stored is not None
    assert stored.id == artifact_id
    assert stored.language == 'python'
    assert stored.byte_start == 10
    assert stored.byte_end == 200
    assert stored.last_modified_at == _TS3


def test_upsert_artifact_kind_mismatch_raises(store_impl: GraphStore) -> None:
    store_impl.upsert_artifact(
        _artifact('/src/Qux.cs', kind=ArtifactKind.CHUNK, chunk_index=None),
    )
    with pytest.raises(ValueError, match='kind mismatch'):
        store_impl.upsert_artifact(
            _artifact(
                '/src/Qux.cs',
                kind=ArtifactKind.SOURCE_FILE,
                chunk_index=None,
            ),
        )


def test_upsert_artifact_source_file_uses_path_only_key(
    store_impl: GraphStore,
) -> None:
    first_id = store_impl.upsert_artifact(
        _artifact(
            '/repo/README.md',
            kind=ArtifactKind.SOURCE_FILE,
            chunk_index=None,
            language=None,
            byte_start=None,
            byte_end=None,
        ),
    )
    second_id = store_impl.upsert_artifact(
        _artifact(
            '/repo/README.md',
            kind=ArtifactKind.SOURCE_FILE,
            chunk_index=99,
            language='markdown',
            byte_start=0,
            byte_end=50,
            last_modified_at=_TS2,
        ),
    )
    assert second_id == first_id


def test_get_artifact_returns_none_for_missing_id(store_impl: GraphStore) -> None:
    assert store_impl.get_artifact(999_999) is None


def test_artifact_roundtrip_via_get(store_impl: GraphStore) -> None:
    original = _artifact(
        '/src/Widget.cs',
        chunk_index=7,
        language='csharp',
        byte_start=128,
        byte_end=512,
        last_modified_at=_TS2,
    )
    artifact_id = store_impl.upsert_artifact(original)
    stored = store_impl.get_artifact(artifact_id)

    assert stored is not None
    assert stored.id == artifact_id
    assert stored.kind == original.kind
    assert stored.path == original.path
    assert stored.chunk_index == original.chunk_index
    assert stored.language == original.language
    assert stored.byte_start == original.byte_start
    assert stored.byte_end == original.byte_end
    assert stored.last_modified_at == original.last_modified_at


def test_transaction_rollback_restores_artifact_state(store_impl: GraphStore) -> None:
    before_id = store_impl.upsert_artifact(_artifact('/src/before.cs'))
    tx = store_impl.begin_transaction()
    tx.__enter__()
    during_id = store_impl.upsert_artifact(_artifact('/src/during.cs'))
    tx.rollback()

    assert store_impl.get_artifact(before_id) is not None
    assert store_impl.get_artifact(during_id) is None


def test_upsert_concepts_batch_empty_returns_empty_list(
    store_impl: GraphStore,
) -> None:
    assert store_impl.upsert_concepts_batch([]) == []


def test_upsert_concepts_batch_assigns_ids_in_input_order(
    store_impl: GraphStore,
) -> None:
    concepts = [_concept(f'concept-{i}') for i in range(5)]
    returned_ids = store_impl.upsert_concepts_batch(concepts)

    assert len(returned_ids) == 5
    for i, concept_id in enumerate(returned_ids):
        stored = store_impl.get_concept(concept_id)
        assert stored is not None
        assert stored.name == concepts[i].name


def test_upsert_concepts_batch_idempotent_on_duplicates_within_batch(
    store_impl: GraphStore,
) -> None:
    concept_a = _concept('alpha-batch')
    concept_b = _concept('beta-batch')
    returned_ids = store_impl.upsert_concepts_batch([concept_a, concept_b, concept_a])

    assert returned_ids[0] == returned_ids[2]
    stored = store_impl.get_concept(returned_ids[0])
    assert stored is not None
    assert stored.name == concept_a.name


def test_upsert_concepts_batch_merges_with_existing(store_impl: GraphStore) -> None:
    concept_a = _concept('merge-alpha')
    concept_b = _concept('merge-beta')
    preloaded_id = store_impl.upsert_concept(concept_a)

    returned_ids = store_impl.upsert_concepts_batch([concept_a, concept_b])

    assert returned_ids[0] == preloaded_id
    assert returned_ids[1] != preloaded_id
    assert store_impl.get_concept(returned_ids[1]) is not None


def test_upsert_artifacts_batch_handles_mixed_kinds(store_impl: GraphStore) -> None:
    chunk1 = _artifact('/src/Mixed.cs', chunk_index=0)
    source_file = _artifact(
        '/repo/Mixed.cs',
        kind=ArtifactKind.SOURCE_FILE,
        chunk_index=None,
        language=None,
        byte_start=None,
        byte_end=None,
    )
    chunk2 = _artifact('/src/Mixed.cs', chunk_index=1)

    returned_ids = store_impl.upsert_artifacts_batch([chunk1, source_file, chunk2])

    assert len(returned_ids) == 3
    assert len(set(returned_ids)) == 3
    assert store_impl.get_artifact(returned_ids[0]) is not None
    assert store_impl.get_artifact(returned_ids[1]) is not None
    assert store_impl.get_artifact(returned_ids[2]) is not None
    stored_chunk1 = store_impl.get_artifact(returned_ids[0])
    stored_chunk2 = store_impl.get_artifact(returned_ids[2])
    assert stored_chunk1 is not None and stored_chunk2 is not None
    assert stored_chunk1.chunk_index == 0
    assert stored_chunk2.chunk_index == 1


def test_upsert_edges_batch_groups_by_type(store_impl: GraphStore) -> None:
    concept_a = store_impl.upsert_concept(_concept('edge-alpha'))
    concept_b = store_impl.upsert_concept(_concept('edge-beta'))
    artifact_id = store_impl.upsert_artifact(_artifact('/src/EdgeBatch.cs'))

    edges = [
        edge_with_props(
            src_id=concept_a,
            dst_id=concept_b,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
        edge_with_props(
            src_id=artifact_id,
            dst_id=concept_a,
            props=DefinesProps(count=2),
        ),
        edge_with_props(
            src_id=artifact_id,
            dst_id=concept_b,
            props=ReferencesProps(count=3, positions=(10, 20)),
        ),
        edge_with_props(
            src_id=concept_b,
            dst_id=concept_a,
            props=CoOccursWithProps(weight=0.5, chunk_count=1),
        ),
    ]
    store_impl.upsert_edges_batch(edges)

    path_ab = store_impl.shortest_path(
        concept_a,
        concept_b,
        edge_types=[EdgeType.CO_OCCURS_WITH],
        max_hops=1,
    )
    path_ba = store_impl.shortest_path(
        concept_b,
        concept_a,
        edge_types=[EdgeType.CO_OCCURS_WITH],
        max_hops=1,
    )
    assert [c.id for c in path_ab] == [concept_a, concept_b]
    assert [c.id for c in path_ba] == [concept_b, concept_a]


def test_upsert_edges_batch_merges_properties_on_repeat(
    store_impl: GraphStore,
) -> None:
    concept_a = store_impl.upsert_concept(_concept('merge-edge-a'))
    concept_b = store_impl.upsert_concept(_concept('merge-edge-b'))
    edge = edge_with_props(
        src_id=concept_a,
        dst_id=concept_b,
        props=CoOccursWithProps(weight=1.0, chunk_count=1),
    )

    store_impl.upsert_edges_batch([edge])
    store_impl.upsert_edges_batch([edge])

    props = _co_occurs_props(store_impl, concept_a, concept_b)
    assert props['chunk_count'] == 2
    assert props['weight'] == 1.0


def test_upsert_edges_batch_raises_on_missing_endpoint(
    store_impl: GraphStore,
) -> None:
    existing = store_impl.upsert_concept(_concept('only-one'))
    edge = edge_with_props(
        src_id=999_999,
        dst_id=existing,
        props=CoOccursWithProps(weight=1.0, chunk_count=1),
    )

    with pytest.raises(ValueError, match='999999'):
        store_impl.upsert_edges_batch([edge])


def test_find_concept_by_name_returns_id_on_exact_match(
    store_impl: GraphStore,
) -> None:
    concept_id = _upsert(store_impl, _concept('toolbar'))
    found = store_impl.find_concept_by_name('toolbar')
    assert found == concept_id
    assert store_impl.find_concept_by_name('toolbar', kind=ConceptKind.ATOMIC) == concept_id


def test_find_concept_by_name_returns_none_when_missing(
    store_impl: GraphStore,
) -> None:
    _upsert(store_impl, _concept('toolbar'))
    assert store_impl.find_concept_by_name('missing-symbol') is None
    assert store_impl.find_concept_by_name('toolbar', kind=ConceptKind.PHRASE) is None


def test_find_concept_by_name_kind_filter_disambiguates(
    store_impl: GraphStore,
) -> None:
    atomic_id = _upsert(store_impl, _concept('view'))
    phrase_id = _upsert(store_impl, _concept('view-model', kind=ConceptKind.PHRASE))

    assert store_impl.find_concept_by_name('view-model', kind=ConceptKind.PHRASE) == phrase_id
    assert store_impl.find_concept_by_name('view-model', kind=ConceptKind.ATOMIC) is None
    assert store_impl.find_concept_by_name('view', kind=ConceptKind.ATOMIC) == atomic_id


def test_neighborhood_is_deterministic(store_impl: GraphStore) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    for i, weight in enumerate((4.0, 1.0, 3.0, 2.0, 5.0)):
        neighbor = _upsert(store_impl, _concept(f'n{i}'))
        _link(store_impl, anchor, neighbor, weight=weight)
    mid = _upsert(store_impl, _concept('mid'))
    _link(store_impl, anchor, mid, weight=2.5)
    far = _upsert(store_impl, _concept('far'))
    _link(store_impl, mid, far, weight=6.0)

    first = store_impl.neighborhood(anchor, radius=2, limit=20)
    second = store_impl.neighborhood(anchor, radius=2, limit=20)
    assert first == second


def test_neighborhood_orders_by_edge_weight_descending(
    store_impl: GraphStore,
) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    for i, weight in enumerate((1.0, 5.0, 3.0, 2.0)):
        neighbor = _upsert(store_impl, _concept(f'n{i}'))
        _link(store_impl, anchor, neighbor, weight=weight)

    neighbors = store_impl.neighborhood(anchor, radius=1, limit=20)
    result_weights = [
        float(_co_occurs_props(store_impl, anchor, concept.id)['weight'])
        for concept in neighbors
    ]
    assert result_weights == [5.0, 3.0, 2.0, 1.0]


def test_neighborhood_tiebreaks_by_concept_id_ascending(
    store_impl: GraphStore,
) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    _upsert(store_impl, _concept('padding-1'))
    n3 = _upsert(store_impl, _concept('n3'))
    _upsert(store_impl, _concept('padding-2'))
    n5 = _upsert(store_impl, _concept('n5'))
    _upsert(store_impl, _concept('padding-3'))
    n7 = _upsert(store_impl, _concept('n7'))
    for neighbor_id in (n3, n5, n7):
        _link(store_impl, anchor, neighbor_id, weight=1.0)

    neighbors = store_impl.neighborhood(anchor, radius=1, limit=20)
    assert [concept.id for concept in neighbors] == [n3, n5, n7]


def test_neighborhood_limit_drops_lowest_weight(store_impl: GraphStore) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    for i, weight in enumerate((1.0, 5.0, 3.0, 2.0)):
        neighbor = _upsert(store_impl, _concept(f'n{i}'))
        _link(store_impl, anchor, neighbor, weight=weight)

    neighbors = store_impl.neighborhood(anchor, radius=1, limit=2)
    assert len(neighbors) == 2
    result_weights = [
        float(_co_occurs_props(store_impl, anchor, concept.id)['weight'])
        for concept in neighbors
    ]
    assert result_weights == [5.0, 3.0]


def test_neighborhood_orders_hop1_before_hop2(store_impl: GraphStore) -> None:
    anchor = _upsert(store_impl, _concept('anchor'))
    hop1 = _upsert(store_impl, _concept('hop1'))
    hop2 = _upsert(store_impl, _concept('hop2'))
    _link(store_impl, anchor, hop1, weight=1.0)
    _link(store_impl, hop1, hop2, weight=5.0)

    neighbors = store_impl.neighborhood(anchor, radius=2, limit=20)
    assert [concept.id for concept in neighbors] == [hop1, hop2]


def test_vector_search_is_deterministic(store_impl: GraphStore) -> None:
    embeddings = [
        (1.0, 0.0, 0.0),
        (0.9, 0.1, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    for i, emb in enumerate(embeddings):
        _upsert(store_impl, _concept(f'c{i}', embedding=emb))

    query = (1.0, 0.0, 0.0)
    first = store_impl.vector_search(query, limit=5)
    second = store_impl.vector_search(query, limit=5)
    assert first == second


def test_vector_search_tiebreaks_by_concept_id_ascending(
    store_impl: GraphStore,
) -> None:
    shared_embedding = (1.0, 0.0, 0.0)
    lower_id = _upsert(
        store_impl,
        _concept('lower-id', embedding=shared_embedding),
    )
    _upsert(store_impl, _concept('padding'))
    higher_id = _upsert(
        store_impl,
        _concept('higher-id', embedding=shared_embedding),
    )

    results = store_impl.vector_search(shared_embedding, limit=5)
    tied = [concept.id for concept, _score in results if concept.id in {lower_id, higher_id}]
    assert tied[:2] == [lower_id, higher_id]


def test_pagerank_is_deterministic(store_impl: GraphStore) -> None:
    hub = _upsert(store_impl, _concept('hub'))
    for name in ('s1', 's2', 's3'):
        spoke = _upsert(store_impl, _concept(name))
        _link(store_impl, spoke, hub)

    first = store_impl.pagerank(edge_types=[EdgeType.CO_OCCURS_WITH], iterations=20)
    second = store_impl.pagerank(edge_types=[EdgeType.CO_OCCURS_WITH], iterations=20)
    assert set(first) == set(second)
    for concept_id in first:
        assert abs(first[concept_id] - second[concept_id]) <= 1e-6


def _build_ppr_chain_cluster(store_impl: GraphStore) -> dict[str, int]:
    """Chain anchor->A->B->C plus disconnected D->E->F."""
    ids: dict[str, int] = {}
    for name in ('anchor', 'A', 'B', 'C', 'D', 'E', 'F'):
        ids[name] = _upsert(store_impl, _concept(name))
    _link(store_impl, ids['anchor'], ids['A'])
    _link(store_impl, ids['A'], ids['B'])
    _link(store_impl, ids['B'], ids['C'])
    _link(store_impl, ids['D'], ids['E'])
    _link(store_impl, ids['E'], ids['F'])
    return ids


def _build_ppr_hub_vs_local(store_impl: GraphStore) -> dict[str, int]:
    """Local chain disconnected from a distant star that wins global PageRank."""
    ids: dict[str, int] = {}
    for name in ('anchor', 'A', 'B', 'D', 'E', 'F', 'x1', 'x2', 'x3'):
        ids[name] = _upsert(store_impl, _concept(name))
    _link(store_impl, ids['anchor'], ids['A'])
    _link(store_impl, ids['A'], ids['B'])
    _link(store_impl, ids['D'], ids['E'])
    _link(store_impl, ids['E'], ids['F'])
    for name in ('x1', 'x2', 'x3', 'F', 'E'):
        _link(store_impl, ids[name], ids['D'])
    return ids


def _build_ppr_disjoint_seeds(store_impl: GraphStore) -> dict[str, int]:
    """Two disjoint chains plus an isolated node."""
    ids: dict[str, int] = {}
    for name in ('seed1', 's1a', 's1b', 'seed2', 's2a', 's2b', 'isolated'):
        ids[name] = _upsert(store_impl, _concept(name))
    _link(store_impl, ids['seed1'], ids['s1a'])
    _link(store_impl, ids['s1a'], ids['s1b'])
    _link(store_impl, ids['seed2'], ids['s2a'])
    _link(store_impl, ids['s2a'], ids['s2b'])
    return ids


def _build_ppr_edge_type_filter(store_impl: GraphStore) -> dict[str, int]:
    """CO_OCCURS_WITH local path vs IS_CANONICAL_ALIAS_OF remote path.

    DEFINES/REFERENCES are artifact→concept in production schema and do not
    participate in concept-only PageRank/PPR walks on Kuzu; alias edges are
    the cross-backend structural stand-in for a second filtered edge family.
    """
    ids: dict[str, int] = {}
    for name in ('seed', 'co_near', 'co_far', 'alias_near', 'alias_far'):
        ids[name] = _upsert(store_impl, _concept(name))
    _link(store_impl, ids['seed'], ids['co_near'])
    _link(store_impl, ids['co_near'], ids['co_far'])
    _link(
        store_impl,
        ids['seed'],
        ids['alias_near'],
        edge_type=EdgeType.IS_CANONICAL_ALIAS_OF,
    )
    _link(
        store_impl,
        ids['alias_near'],
        ids['alias_far'],
        edge_type=EdgeType.IS_CANONICAL_ALIAS_OF,
    )
    return ids


def _build_ppr_damping_graph(store_impl: GraphStore) -> dict[str, int]:
    """Linear chain seed -> hop1 -> hop2 for damping comparison."""
    ids: dict[str, int] = {}
    for name in ('seed', 'hop1', 'hop2'):
        ids[name] = _upsert(store_impl, _concept(name))
    _link(store_impl, ids['seed'], ids['hop1'])
    _link(store_impl, ids['hop1'], ids['hop2'])
    return ids


def test_ppr_empty_seeds_returns_empty(store_impl: GraphStore) -> None:
    _upsert(store_impl, _concept('solo'))
    assert store_impl.personalized_pagerank([]) == {}


def test_ppr_unknown_seeds_dropped(store_impl: GraphStore) -> None:
    valid_id = _upsert(store_impl, _concept('valid'))
    assert store_impl.personalized_pagerank([99999]) == {}
    scores = store_impl.personalized_pagerank([valid_id, 99999])
    assert valid_id in scores
    assert scores[valid_id] > 0.0


def test_ppr_concentrates_on_seed_neighborhood(store_impl: GraphStore) -> None:
    ids = _build_ppr_chain_cluster(store_impl)
    scores = store_impl.personalized_pagerank([ids['anchor']], iterations=30)
    local = [ids['anchor'], ids['A'], ids['B'], ids['C']]
    distant = [ids['D'], ids['E'], ids['F']]
    local_min = min(scores[n] for n in local)
    distant_max = max(scores[n] for n in distant)
    assert local_min > distant_max


def test_ppr_is_deterministic(store_impl: GraphStore) -> None:
    ids = _build_ppr_chain_cluster(store_impl)
    first = store_impl.personalized_pagerank([ids['anchor']], iterations=20)
    second = store_impl.personalized_pagerank([ids['anchor']], iterations=20)
    assert set(first) == set(second)
    for concept_id in first:
        assert abs(first[concept_id] - second[concept_id]) <= 1e-6


def test_ppr_differs_from_global_pagerank(store_impl: GraphStore) -> None:
    ids = _build_ppr_hub_vs_local(store_impl)
    pr = store_impl.pagerank(iterations=30)
    ppr = store_impl.personalized_pagerank([ids['anchor']], iterations=30)
    ppr_local = [ids['anchor'], ids['A'], ids['B']]
    pr_top = max(pr, key=pr.get)
    ppr_top = max(ppr, key=ppr.get)
    assert ppr_top in ppr_local
    assert pr_top == ids['D']
    assert pr_top not in ppr_local
    assert ppr[ids['anchor']] > ppr[ids['D']]


def test_ppr_multiple_seeds_unions(store_impl: GraphStore) -> None:
    ids = _build_ppr_disjoint_seeds(store_impl)
    dual = store_impl.personalized_pagerank(
        [ids['seed1'], ids['seed2']],
        iterations=30,
    )
    seed1_neighbors = [ids['seed1'], ids['s1a'], ids['s1b']]
    seed2_neighbors = [ids['seed2'], ids['s2a'], ids['s2b']]
    isolated_score = dual[ids['isolated']]
    for neighbor_id in seed1_neighbors + seed2_neighbors:
        assert dual[neighbor_id] > isolated_score


def test_ppr_respects_edge_type_filter(store_impl: GraphStore) -> None:
    ids = _build_ppr_edge_type_filter(store_impl)
    co_scores = store_impl.personalized_pagerank(
        [ids['seed']],
        edge_types=[EdgeType.CO_OCCURS_WITH],
        iterations=30,
    )
    alias_scores = store_impl.personalized_pagerank(
        [ids['seed']],
        edge_types=[EdgeType.IS_CANONICAL_ALIAS_OF],
        iterations=30,
    )
    assert co_scores != alias_scores
    assert co_scores[ids['co_far']] > co_scores[ids['alias_far']]
    assert alias_scores[ids['alias_far']] > alias_scores[ids['co_far']]


def test_ppr_damping_effect(store_impl: GraphStore) -> None:
    ids = _build_ppr_damping_graph(store_impl)
    low_damp = store_impl.personalized_pagerank(
        [ids['seed']],
        damping=0.5,
        iterations=30,
    )
    high_damp = store_impl.personalized_pagerank(
        [ids['seed']],
        damping=0.95,
        iterations=30,
    )
    assert high_damp[ids['hop2']] > low_damp[ids['hop2']]


def _assert_embedding_matches(
    store_impl: GraphStore,
    concept_id: int,
    expected: tuple[float, ...] | None,
) -> None:
    """Assert stored embedding matches ``expected`` (Kuzu may zero-pad)."""
    stored = store_impl.get_concept(concept_id)
    assert stored is not None
    if expected is None:
        assert stored.embedding is None
        return
    assert stored.embedding is not None
    assert stored.embedding[: len(expected)] == expected
    if len(stored.embedding) > len(expected):
        assert all(v == 0.0 for v in stored.embedding[len(expected) :])


def test_list_concepts_returns_all(store_impl: GraphStore) -> None:
    names = ('alpha', 'beta', 'gamma', 'delta', 'epsilon')
    expected_ids = {_upsert(store_impl, _concept(name)) for name in names}
    listed = list(store_impl.list_concepts())
    assert len(listed) == 5
    assert {c.id for c in listed} == expected_ids


def test_list_concepts_is_id_ascending(store_impl: GraphStore) -> None:
    for name in ('zebra', 'alpha', 'mango', 'beta'):
        _upsert(store_impl, _concept(name))
    ids = [c.id for c in store_impl.list_concepts()]
    assert ids == sorted(ids)
    assert len(ids) >= 2
    assert ids == sorted(set(ids))


def test_list_concepts_is_deterministic(store_impl: GraphStore) -> None:
    for name in ('one', 'two', 'three'):
        _upsert(store_impl, _concept(name))
    first = list(store_impl.list_concepts())
    second = list(store_impl.list_concepts())
    assert first == second


def test_list_concepts_empty_store(store_impl: GraphStore) -> None:
    assert list(store_impl.list_concepts()) == []


def test_set_concept_embedding_overwrites_existing(store_impl: GraphStore) -> None:
    cid = _upsert(store_impl, _concept('embed-me', embedding=(1.0, 0.0)))
    store_impl.set_concept_embedding(cid, (0.0, 1.0))
    _assert_embedding_matches(store_impl, cid, (0.0, 1.0))
    store_impl.set_concept_embedding(cid, None)
    assert store_impl.get_concept(cid) is not None
    assert store_impl.get_concept(cid).embedding is None


def test_set_concept_embedding_unknown_id_raises_keyerror(
    store_impl: GraphStore,
) -> None:
    with pytest.raises(KeyError):
        store_impl.set_concept_embedding(999_999, (1.0, 0.0))


def test_set_concept_embedding_empty_tuple_raises_valueerror(
    store_impl: GraphStore,
) -> None:
    cid = _upsert(store_impl, _concept('empty-tuple'))
    with pytest.raises(ValueError, match='empty-tuple'):
        store_impl.set_concept_embedding(cid, ())


def test_set_concept_embedding_reflected_in_vector_search(
    store_impl: GraphStore,
) -> None:
    a_id = _upsert(store_impl, _concept('concept-a', embedding=(1.0, 0.0)))
    b_id = _upsert(store_impl, _concept('concept-b', embedding=(0.0, 1.0)))
    query = (1.0, 0.0)

    before = store_impl.vector_search(query, limit=1)
    assert before[0][0].id == a_id
    assert before[0][1] > 0.0

    store_impl.set_concept_embedding(a_id, (0.0, 1.0))
    after = store_impl.vector_search(query, limit=2)

    if after[0][0].id == a_id:
        assert len(after) >= 2
        assert after[0][1] == after[1][1]
        assert a_id < after[1][0].id
    else:
        assert after[0][0].id != a_id


def test_list_artifacts_for_concept_returns_is_named_in_artifacts(
    store_impl: GraphStore,
) -> None:
    concept_id = _upsert(store_impl, _concept('named-in-target'))
    artifact_a = store_impl.upsert_artifact(_artifact('/src/Alpha.cs', chunk_index=0))
    artifact_b = store_impl.upsert_artifact(_artifact('/src/Beta.cs', chunk_index=1))
    _link_concept_artifact(
        store_impl,
        concept_id,
        artifact_a,
        edge_type=EdgeType.IS_NAMED_IN,
    )
    _link_concept_artifact(
        store_impl,
        concept_id,
        artifact_b,
        edge_type=EdgeType.IS_NAMED_IN,
    )

    artifacts = store_impl.list_artifacts_for_concept(concept_id)
    assert {a.id for a in artifacts} == {artifact_a, artifact_b}


def test_list_artifacts_for_concept_deterministic_order(
    store_impl: GraphStore,
) -> None:
    concept_id = _upsert(store_impl, _concept('order-target'))
    low = store_impl.upsert_artifact(_artifact('/src/low.cs', chunk_index=0))
    mid = store_impl.upsert_artifact(_artifact('/src/mid.cs', chunk_index=1))
    high = store_impl.upsert_artifact(_artifact('/src/high.cs', chunk_index=2))
    _link_concept_artifact(
        store_impl,
        concept_id,
        low,
        edge_type=EdgeType.DEFINES,
        count=1,
    )
    _link_concept_artifact(
        store_impl,
        concept_id,
        mid,
        edge_type=EdgeType.DEFINES,
        count=3,
    )
    _link_concept_artifact(
        store_impl,
        concept_id,
        high,
        edge_type=EdgeType.DEFINES,
        count=5,
    )

    first = store_impl.list_artifacts_for_concept(
        concept_id,
        edge_types=(EdgeType.DEFINES,),
    )
    second = store_impl.list_artifacts_for_concept(
        concept_id,
        edge_types=(EdgeType.DEFINES,),
    )
    assert first == second
    assert [a.id for a in first] == [high, mid, low]


def test_list_artifacts_for_concept_limit(store_impl: GraphStore) -> None:
    concept_id = _upsert(store_impl, _concept('limit-target'))
    artifact_ids: list[int] = []
    for i in range(5):
        artifact_id = store_impl.upsert_artifact(
            _artifact(f'/src/limit{i}.cs', chunk_index=i),
        )
        artifact_ids.append(artifact_id)
        _link_concept_artifact(
            store_impl,
            concept_id,
            artifact_id,
            edge_type=EdgeType.IS_NAMED_IN,
        )

    artifacts = store_impl.list_artifacts_for_concept(concept_id, limit=3)
    assert len(artifacts) == 3


def test_list_artifacts_for_concept_edge_types_filter(
    store_impl: GraphStore,
) -> None:
    concept_id = _upsert(store_impl, _concept('filter-target'))
    named_in_artifact = store_impl.upsert_artifact(
        _artifact('/src/named.cs', chunk_index=0),
    )
    defines_artifact = store_impl.upsert_artifact(
        _artifact('/src/defines.cs', chunk_index=1),
    )
    _link_concept_artifact(
        store_impl,
        concept_id,
        named_in_artifact,
        edge_type=EdgeType.IS_NAMED_IN,
    )
    _link_concept_artifact(
        store_impl,
        concept_id,
        defines_artifact,
        edge_type=EdgeType.DEFINES,
    )

    named_only = store_impl.list_artifacts_for_concept(
        concept_id,
        edge_types=(EdgeType.IS_NAMED_IN,),
    )
    both = store_impl.list_artifacts_for_concept(
        concept_id,
        edge_types=(EdgeType.DEFINES, EdgeType.IS_NAMED_IN),
    )
    assert {a.id for a in named_only} == {named_in_artifact}
    assert {a.id for a in both} == {named_in_artifact, defines_artifact}


def test_list_artifacts_for_concept_raises_on_unknown_concept_id(
    store_impl: GraphStore,
) -> None:
    with pytest.raises(KeyError):
        store_impl.list_artifacts_for_concept(999_999)


def test_list_artifacts_for_concept_empty_when_no_edges(
    store_impl: GraphStore,
) -> None:
    concept_id = _upsert(store_impl, _concept('lonely'))
    assert store_impl.list_artifacts_for_concept(concept_id) == []
