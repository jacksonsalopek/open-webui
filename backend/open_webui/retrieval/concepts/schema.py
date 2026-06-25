"""Substrate-independent graph schema for the concept knowledge graph.

Application code (extractors, stores, retrievers) depends on these shapes
rather than on Kuzu, Neo4j, or any other persistence driver. The integer
``EdgeType`` codes are reserved across Phase 1–4 so storage layers can
persist relationship kinds without enum churn when new edge types land.

Every dataclass round-trips through ``*_to_dict`` / ``*_from_dict`` so
stores can serialize to JSON or columnar formats without importing
engine-specific types.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any

log = logging.getLogger(__name__)


class ConceptKind(str, Enum):
    ATOMIC = 'atomic'
    PHRASE = 'phrase'
    ROLE = 'role'


class ArtifactKind(str, Enum):
    SOURCE_FILE = 'source_file'
    CHUNK = 'chunk'
    DOC_FILE = 'doc_file'
    # Reserved for v2+ — defined now so enum extension does not break storage.
    PR_DESCRIPTION = 'pr_description'
    COMMIT_MESSAGE = 'commit_message'
    SLACK_THREAD = 'slack_thread'


class EdgeType(IntEnum):
    DEFINES = 1
    REFERENCES = 2
    CO_OCCURS_WITH = 3
    IS_NAMED_IN = 4
    IS_CANONICAL_ALIAS_OF = 5
    # Reserved for v2–v4 — integer mapping must stay stable across phases.
    IS_DISCUSSED_IN = 6
    IS_DEFINED_BY = 7
    IS_OWNED_BY = 8
    WAS_INTRODUCED_BY = 9
    WAS_LAST_CHANGED_BY = 10
    SUPERSEDES = 11
    IS_DEPRECATED_BY = 12


@dataclass(frozen=True, slots=True)
class Concept:
    id: int
    name: str
    kind: ConceptKind
    first_seen_at: datetime
    last_seen_at: datetime
    centrality_score: float | None
    embedding: tuple[float, ...] | None
    definition: str | None
    language_hint: str | None
    original_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.definition is None) != (self.kind != ConceptKind.PHRASE):
            raise ValueError(
                'definition must be set iff kind is PHRASE; '
                f'got kind={self.kind!r}, definition={self.definition!r}',
            )


@dataclass(frozen=True, slots=True)
class Artifact:
    id: int
    kind: ArtifactKind
    path: str
    chunk_index: int | None
    language: str | None
    byte_start: int | None
    byte_end: int | None
    last_modified_at: datetime


@dataclass(frozen=True, slots=True)
class Edge:
    type: EdgeType
    src_id: int
    dst_id: int
    properties: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.properties, MappingProxyType):
            object.__setattr__(
                self,
                'properties',
                MappingProxyType(dict(self.properties)),
            )


@dataclass(frozen=True, slots=True)
class DefinesProps:
    count: int


@dataclass(frozen=True, slots=True)
class ReferencesProps:
    count: int
    positions: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class CoOccursWithProps:
    weight: float
    chunk_count: int


@dataclass(frozen=True, slots=True)
class IsNamedInProps:
    first_seen_at: datetime


@dataclass(frozen=True, slots=True)
class IsCanonicalAliasOfProps:
    introduced_at: datetime
    planned_removal_at: datetime | None = None
    removed_at: datetime | None = None


EdgeProps = (
    DefinesProps
    | ReferencesProps
    | CoOccursWithProps
    | IsNamedInProps
    | IsCanonicalAliasOfProps
)

_EDGE_TYPE_FOR_PROPS: dict[type[EdgeProps], EdgeType] = {
    DefinesProps: EdgeType.DEFINES,
    ReferencesProps: EdgeType.REFERENCES,
    CoOccursWithProps: EdgeType.CO_OCCURS_WITH,
    IsNamedInProps: EdgeType.IS_NAMED_IN,
    IsCanonicalAliasOfProps: EdgeType.IS_CANONICAL_ALIAS_OF,
}


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _props_to_dict(props: EdgeProps) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(props):
        value = getattr(props, field.name)
        if isinstance(value, datetime):
            result[field.name] = _iso(value)
        elif isinstance(value, tuple):
            result[field.name] = list(value)
        else:
            result[field.name] = value
    return result


def edge_with_props(
    *,
    src_id: int,
    dst_id: int,
    props: EdgeProps,
) -> Edge:
    edge_type = _EDGE_TYPE_FOR_PROPS[type(props)]
    return Edge(
        type=edge_type,
        src_id=src_id,
        dst_id=dst_id,
        properties=_props_to_dict(props),
    )


def concept_to_dict(concept: Concept) -> dict[str, Any]:
    return {
        'id': concept.id,
        'name': concept.name,
        'kind': concept.kind.value,
        'first_seen_at': _iso(concept.first_seen_at),
        'last_seen_at': _iso(concept.last_seen_at),
        'centrality_score': concept.centrality_score,
        'embedding': list(concept.embedding) if concept.embedding is not None else None,
        'definition': concept.definition,
        'language_hint': concept.language_hint,
        'original_tokens': list(concept.original_tokens),
    }


def concept_from_dict(data: Mapping[str, Any]) -> Concept:
    embedding_raw = data.get('embedding')
    tokens_raw = data.get('original_tokens', [])
    return Concept(
        id=int(data['id']),
        name=str(data['name']),
        kind=ConceptKind(str(data['kind'])),
        first_seen_at=_parse_dt(str(data['first_seen_at'])),
        last_seen_at=_parse_dt(str(data['last_seen_at'])),
        centrality_score=(
            float(data['centrality_score'])
            if data.get('centrality_score') is not None
            else None
        ),
        embedding=tuple(float(x) for x in embedding_raw) if embedding_raw is not None else None,
        definition=(
            str(data['definition']) if data.get('definition') is not None else None
        ),
        language_hint=(
            str(data['language_hint']) if data.get('language_hint') is not None else None
        ),
        original_tokens=tuple(str(x) for x in tokens_raw),
    )


def artifact_to_dict(artifact: Artifact) -> dict[str, Any]:
    return {
        'id': artifact.id,
        'kind': artifact.kind.value,
        'path': artifact.path,
        'chunk_index': artifact.chunk_index,
        'language': artifact.language,
        'byte_start': artifact.byte_start,
        'byte_end': artifact.byte_end,
        'last_modified_at': _iso(artifact.last_modified_at),
    }


def artifact_from_dict(data: Mapping[str, Any]) -> Artifact:
    return Artifact(
        id=int(data['id']),
        kind=ArtifactKind(str(data['kind'])),
        path=str(data['path']),
        chunk_index=(
            int(data['chunk_index']) if data.get('chunk_index') is not None else None
        ),
        language=str(data['language']) if data.get('language') is not None else None,
        byte_start=int(data['byte_start']) if data.get('byte_start') is not None else None,
        byte_end=int(data['byte_end']) if data.get('byte_end') is not None else None,
        last_modified_at=_parse_dt(str(data['last_modified_at'])),
    )


def edge_to_dict(edge: Edge) -> dict[str, Any]:
    return {
        'type': int(edge.type),
        'src_id': edge.src_id,
        'dst_id': edge.dst_id,
        'properties': dict(edge.properties),
    }


def edge_from_dict(data: Mapping[str, Any]) -> Edge:
    return Edge(
        type=EdgeType(int(data['type'])),
        src_id=int(data['src_id']),
        dst_id=int(data['dst_id']),
        properties=dict(data['properties']),
    )
