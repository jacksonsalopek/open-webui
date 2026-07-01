"""JSON-safe serializers for MCP tool responses."""

from __future__ import annotations

from typing import Any

from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.concepts.schema import Artifact, Concept, EdgeType


def edge_type_to_string(edge_type: EdgeType) -> str:
    """Serialize EdgeType codes as lowercase snake strings."""
    return edge_type.name.lower()


def serialize_concept(concept: Concept) -> dict[str, Any]:
    """ConceptSummary — embedding and lifecycle timestamps are never included."""
    result: dict[str, Any] = {
        'id': concept.id,
        'name': concept.name,
        'kind': concept.kind.value,
        'definition': concept.definition,
        'centrality_score': concept.centrality_score,
        'original_tokens': list(concept.original_tokens),
    }
    if concept.language_hint is not None:
        result['language_hint'] = concept.language_hint
    assert_no_embedding(result)
    return result


def serialize_artifact(
    artifact: Artifact,
    *,
    edge_type: EdgeType | None = None,
) -> dict[str, Any]:
    """ArtifactSummary with optional MCP edge_type enrichment."""
    result: dict[str, Any] = {
        'id': artifact.id,
        'path': artifact.path,
        'kind': artifact.kind.value,
        'chunk_index': artifact.chunk_index,
        'language': artifact.language,
    }
    if edge_type is not None:
        result['edge_type'] = edge_type_to_string(edge_type)
    if artifact.byte_start is not None:
        result['byte_start'] = artifact.byte_start
    if artifact.byte_end is not None:
        result['byte_end'] = artifact.byte_end
    assert_no_embedding(result)
    return result


def serialize_neighbor(
    concept: Concept,
    edge_type: EdgeType,
    *,
    hop_distance: int | None = None,
) -> dict[str, Any]:
    """NeighborSummary — partial concept fields plus edge metadata."""
    result: dict[str, Any] = {
        'name': concept.name,
        'kind': concept.kind.value,
        'edge_type': edge_type_to_string(edge_type),
    }
    if hop_distance is not None:
        result['hop_distance'] = hop_distance
    assert_no_embedding(result)
    return result


def serialize_hit(hit: RetrievalHit) -> dict[str, Any]:
    """RetrievalHitSummary — drops raw embedding cosines from provenance."""
    allowed_prov = ('hop_distance', 'seed_id', 'retriever', 'ppr')
    provenance = {
        key: hit.provenance[key]
        for key in allowed_prov
        if key in hit.provenance
    }
    if hit.concept is not None:
        result: dict[str, Any] = {
            'hit_type': 'concept',
            'name': hit.concept.name,
            'path': None,
            'kind': hit.concept.kind.value,
            'score': hit.score,
            'provenance': provenance,
        }
    else:
        artifact = hit.artifact
        assert artifact is not None
        result = {
            'hit_type': 'artifact',
            'name': None,
            'path': artifact.path,
            'kind': artifact.kind.value,
            'score': hit.score,
            'provenance': provenance,
        }
    assert_no_embedding(result)
    return result


def serialize_path_step(
    concept: Concept,
    edge_type_to_next: EdgeType | None,
) -> dict[str, Any]:
    """PathStep for impact_analysis shortest-path output."""
    result: dict[str, Any] = {
        'concept': serialize_concept(concept),
        'edge_type_to_next': (
            edge_type_to_string(edge_type_to_next)
            if edge_type_to_next is not None
            else None
        ),
    }
    assert_no_embedding(result)
    return result


def assert_no_embedding(obj: Any) -> None:
    """Hard invariant: embedding must never appear in serialized MCP output."""
    if isinstance(obj, dict):
        if 'embedding' in obj:
            raise AssertionError('embedding must never be serialized in MCP responses')
        for value in obj.values():
            assert_no_embedding(value)
    elif isinstance(obj, list):
        for item in obj:
            assert_no_embedding(item)
