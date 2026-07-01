"""Unit tests for concept-graph MCP core tool handlers."""

from __future__ import annotations

from datetime import datetime, timezone

from open_webui.retrieval.concepts.mcp.context import CallerContext
from open_webui.retrieval.concepts.mcp.serialize import assert_no_embedding
from open_webui.retrieval.concepts.mcp.tools import (
    explain_region,
    find_concept,
    impact_analysis,
    trace_neighborhood,
    where_used,
)
from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    Concept,
    ConceptKind,
    CoOccursWithProps,
    IsNamedInProps,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.memory_store import InMemoryGraphStore

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _concept(
    name: str,
    *,
    kind: ConceptKind = ConceptKind.ATOMIC,
    concept_id: int | None = None,
    embedding: tuple[float, ...] | None = (0.1, 0.2, 0.3),
) -> Concept:
    return Concept(
        id=concept_id or 0,
        name=name,
        kind=kind,
        first_seen_at=_NOW,
        last_seen_at=_NOW,
        centrality_score=0.5,
        embedding=embedding,
        definition='defined term' if kind == ConceptKind.PHRASE else None,
        language_hint='csharp',
        original_tokens=(name,),
    )


def _artifact(path: str, *, artifact_id: int | None = None) -> Artifact:
    return Artifact(
        id=artifact_id or 0,
        kind=ArtifactKind.SOURCE_FILE,
        path=path,
        chunk_index=None,
        language='csharp',
        byte_start=0,
        byte_end=100,
        last_modified_at=_NOW,
    )


def _link_named_in(store: InMemoryGraphStore, concept_id: int, artifact_id: int) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=concept_id,
            dst_id=artifact_id,
            props=IsNamedInProps(first_seen_at=_NOW),
        ),
    )


def _link_co_occurs(
    store: InMemoryGraphStore,
    left_id: int,
    right_id: int,
) -> None:
    store.upsert_edge(
        edge_with_props(
            src_id=left_id,
            dst_id=right_id,
            props=CoOccursWithProps(weight=1.0, chunk_count=1),
        ),
    )


def build_fixture_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    anchor = _concept('test')
    neighbor = _concept('neighbor')
    login = _concept('login')
    logging = _concept('logging')
    artifact = _artifact('/src/test.cs')

    anchor_id = store.upsert_concept(anchor)
    neighbor_id = store.upsert_concept(neighbor)
    login_id = store.upsert_concept(login)
    logging_id = store.upsert_concept(logging)
    artifact_id = store.upsert_artifact(artifact)

    _link_named_in(store, anchor_id, artifact_id)
    _link_named_in(store, neighbor_id, artifact_id)
    _link_named_in(store, login_id, artifact_id)
    _link_named_in(store, logging_id, artifact_id)
    _link_co_occurs(store, anchor_id, neighbor_id)
    _link_co_occurs(store, login_id, logging_id)
    return store


def bypass_caller() -> CallerContext:
    return CallerContext(user_id='test', accessible_artifact_paths=frozenset(), bypass_acl=True)


def acl_caller(*paths: str) -> CallerContext:
    return CallerContext(
        user_id='restricted',
        accessible_artifact_paths=frozenset(paths),
        bypass_acl=False,
    )


def test_find_concept_happy_path():
    store = build_fixture_store()
    result = find_concept('test', store=store, caller=bypass_caller())

    assert result['status'] == 'ok'
    assert result['tool'] == 'find_concept'
    assert result['concept']['name'] == 'test'
    assert isinstance(result['neighbors'], list)
    assert any(n['name'] == 'neighbor' for n in result['neighbors'])


def test_find_concept_not_found():
    store = build_fixture_store()
    result = find_concept('missing', store=store, caller=bypass_caller())
    assert result['status'] == 'not_found'


def test_find_concept_ambiguous():
    store = InMemoryGraphStore()
    artifact = _artifact('/src/ambiguous.cs')
    artifact_id = store.upsert_artifact(artifact)
    for kind in (ConceptKind.ATOMIC, ConceptKind.PHRASE):
        concept_id = store.upsert_concept(_concept('tip', kind=kind))
        _link_named_in(store, concept_id, artifact_id)

    result = find_concept('tip', store=store, caller=bypass_caller())
    assert result['status'] == 'ambiguous'
    assert len(result['candidates']) == 2


def test_find_concept_acl_invisible():
    store = build_fixture_store()
    caller = acl_caller('/other/path.cs')
    result = find_concept('test', store=store, caller=caller)
    assert result['status'] == 'not_found'


def test_where_used():
    store = build_fixture_store()
    result = where_used('test', store=store, caller=bypass_caller())

    assert result['status'] == 'ok'
    assert result['concept']['name'] == 'test'
    assert len(result['artifacts']) == 1
    assert result['artifacts'][0]['path'] == '/src/test.cs'


def test_explain_region():
    store = build_fixture_store()
    result = explain_region('test', radius=2, store=store, caller=bypass_caller())

    assert result['status'] == 'ok'
    assert result['concept']['name'] == 'test'
    assert any(n['name'] == 'neighbor' for n in result['neighbors'])
    assert isinstance(result['ppr_ranked'], list)


def test_trace_neighborhood():
    store = build_fixture_store()
    result = trace_neighborhood(
        'what is test',
        store=store,
        caller=bypass_caller(),
    )

    assert result['status'] in {'ok', 'empty'}
    assert 'intent' in result
    assert isinstance(result['hits'], list)


def test_impact_analysis():
    store = build_fixture_store()
    result = impact_analysis(
        'login',
        'logging',
        store=store,
        caller=bypass_caller(),
    )

    assert result['status'] == 'ok'
    assert result['path_found'] is True
    assert len(result['path']) >= 2


def test_impact_analysis_no_path():
    store = build_fixture_store()
    result = impact_analysis(
        'test',
        'logging',
        store=store,
        caller=bypass_caller(),
    )

    assert result['status'] == 'ok'
    assert result['path_found'] is False


def test_unavailable_when_store_none():
    caller = bypass_caller()
    assert find_concept('test', store=None, caller=caller)['status'] == 'unavailable'
    assert where_used('test', store=None, caller=caller)['status'] == 'unavailable'
    assert explain_region('test', store=None, caller=caller)['status'] == 'unavailable'
    assert trace_neighborhood('query', store=None, caller=caller)['status'] == 'unavailable'
    assert impact_analysis('a', 'b', store=None, caller=caller)['status'] == 'unavailable'


def test_embedding_never_serialized():
    store = build_fixture_store()
    caller = bypass_caller()
    responses = [
        find_concept('test', store=store, caller=caller),
        where_used('test', store=store, caller=caller),
        explain_region('test', store=store, caller=caller),
        trace_neighborhood('what is test', store=store, caller=caller),
        impact_analysis('login', 'logging', store=store, caller=caller),
    ]
    for response in responses:
        assert_no_embedding(response)


def test_bypass_acl():
    store = build_fixture_store()
    caller = bypass_caller()
    result = find_concept('test', store=store, caller=caller)
    assert result['status'] == 'ok'
    assert result['provenance']['acl_applied'] is False
