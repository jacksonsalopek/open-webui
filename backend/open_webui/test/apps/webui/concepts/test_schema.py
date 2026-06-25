"""Round-trip and invariant tests for ``open_webui.retrieval.concepts.schema``."""

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
    Edge,
    EdgeType,
    IsCanonicalAliasOfProps,
    IsNamedInProps,
    ReferencesProps,
    artifact_from_dict,
    artifact_to_dict,
    concept_from_dict,
    concept_to_dict,
    edge_from_dict,
    edge_to_dict,
    edge_with_props,
)

_TS = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_TS2 = datetime(2025, 6, 2, 8, 30, 0, tzinfo=timezone.utc)


def _concept_for_kind(kind: ConceptKind) -> Concept:
    return Concept(
        id=1,
        name='race-condition' if kind == ConceptKind.PHRASE else 'toolbar',
        kind=kind,
        first_seen_at=_TS,
        last_seen_at=_TS2,
        centrality_score=0.42 if kind == ConceptKind.ATOMIC else None,
        embedding=(0.1, 0.2, 0.3) if kind == ConceptKind.ATOMIC else None,
        definition=(
            'A defect where outcome depends on timing of concurrent events.'
            if kind == ConceptKind.PHRASE
            else None
        ),
        language_hint='csharp' if kind != ConceptKind.ROLE else None,
        original_tokens=('RaceCondition', 'race_condition'),
    )


@pytest.mark.parametrize('kind', list(ConceptKind))
def test_concept_roundtrip(kind: ConceptKind) -> None:
    original = _concept_for_kind(kind)
    restored = concept_from_dict(concept_to_dict(original))
    assert restored == original


def test_definition_invariant() -> None:
    with pytest.raises(ValueError, match='definition must be set iff kind is PHRASE'):
        Concept(
            id=1,
            name='race-condition',
            kind=ConceptKind.PHRASE,
            first_seen_at=_TS,
            last_seen_at=_TS2,
            centrality_score=None,
            embedding=None,
            definition=None,
            language_hint=None,
            original_tokens=(),
        )

    with pytest.raises(ValueError, match='definition must be set iff kind is PHRASE'):
        Concept(
            id=2,
            name='toolbar',
            kind=ConceptKind.ATOMIC,
            first_seen_at=_TS,
            last_seen_at=_TS2,
            centrality_score=None,
            embedding=None,
            definition='not allowed on atomic',
            language_hint=None,
            original_tokens=(),
        )


@pytest.mark.parametrize('kind', list(ArtifactKind))
def test_artifact_roundtrip(kind: ArtifactKind) -> None:
    original = Artifact(
        id=10,
        kind=kind,
        path='src/ToolbarViewModel.cs',
        chunk_index=3 if kind == ArtifactKind.CHUNK else None,
        language='csharp',
        byte_start=100 if kind == ArtifactKind.CHUNK else None,
        byte_end=500 if kind == ArtifactKind.CHUNK else None,
        last_modified_at=_TS,
    )
    restored = artifact_from_dict(artifact_to_dict(original))
    assert restored == original


_V1_EDGE_CASES = [
    (
        EdgeType.DEFINES,
        DefinesProps(count=7),
    ),
    (
        EdgeType.REFERENCES,
        ReferencesProps(count=3, positions=(10, 20, 30)),
    ),
    (
        EdgeType.CO_OCCURS_WITH,
        CoOccursWithProps(weight=0.75, chunk_count=4),
    ),
    (
        EdgeType.IS_NAMED_IN,
        IsNamedInProps(first_seen_at=_TS),
    ),
    (
        EdgeType.IS_CANONICAL_ALIAS_OF,
        IsCanonicalAliasOfProps(
            introduced_at=_TS,
            planned_removal_at=_TS2,
            removed_at=None,
        ),
    ),
]


@pytest.mark.parametrize('edge_type, props', _V1_EDGE_CASES, ids=[t.name for t, _ in _V1_EDGE_CASES])
def test_edge_roundtrip(edge_type: EdgeType, props: object) -> None:
    original = edge_with_props(src_id=1, dst_id=2, props=props)  # type: ignore[arg-type]
    assert original.type == edge_type
    restored = edge_from_dict(edge_to_dict(original))
    assert restored == original


def test_edgetype_codes_stable() -> None:
    # Storage layers persist EdgeType as integers; renumbering breaks on-disk
    # graphs and cross-phase migrations. This test fails loudly if codes drift.
    assert EdgeType.DEFINES == 1
    assert EdgeType.REFERENCES == 2
    assert EdgeType.CO_OCCURS_WITH == 3
    assert EdgeType.IS_NAMED_IN == 4
    assert EdgeType.IS_CANONICAL_ALIAS_OF == 5
    assert EdgeType.IS_DISCUSSED_IN == 6
    assert EdgeType.IS_DEFINED_BY == 7
    assert EdgeType.IS_OWNED_BY == 8
    assert EdgeType.WAS_INTRODUCED_BY == 9
    assert EdgeType.WAS_LAST_CHANGED_BY == 10
    assert EdgeType.SUPERSEDES == 11
    assert EdgeType.IS_DEPRECATED_BY == 12


def test_properties_immutable() -> None:
    edge = edge_with_props(
        src_id=1,
        dst_id=2,
        props=DefinesProps(count=1),
    )
    with pytest.raises(TypeError):
        edge.properties['x'] = 'y'  # type: ignore[index]
