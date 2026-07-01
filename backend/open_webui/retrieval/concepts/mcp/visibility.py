"""Concept-graph visibility inheritance for MCP tool responses."""

from __future__ import annotations

from open_webui.retrieval.concepts.retrieve.base import RetrievalHit
from open_webui.retrieval.concepts.schema import Artifact, Concept, EdgeType
from open_webui.retrieval.concepts.store.protocol import GraphStore

from .context import CallerContext


def is_concept_visible(
    concept_id: int,
    store: GraphStore,
    accessible_paths: frozenset[str],
) -> bool:
    """True iff the concept has >=1 IS_NAMED_IN edge to an accessible artifact."""
    artifacts = store.list_artifacts_for_concept(
        concept_id,
        edge_types=(EdgeType.IS_NAMED_IN,),
        limit=None,
    )
    return any(artifact.path in accessible_paths for artifact in artifacts)


def filter_concepts(
    concepts: list[Concept],
    store: GraphStore,
    caller: CallerContext,
) -> list[Concept]:
    if caller.bypass_acl:
        return concepts
    return [
        concept
        for concept in concepts
        if is_concept_visible(concept.id, store, caller.accessible_artifact_paths)
    ]


def filter_artifacts(
    artifacts: list[Artifact],
    caller: CallerContext,
) -> list[Artifact]:
    if caller.bypass_acl:
        return artifacts
    return [
        artifact
        for artifact in artifacts
        if artifact.path in caller.accessible_artifact_paths
    ]


def filter_hits(
    hits: list[RetrievalHit],
    store: GraphStore,
    caller: CallerContext,
) -> list[RetrievalHit]:
    if caller.bypass_acl:
        return hits
    filtered: list[RetrievalHit] = []
    for hit in hits:
        if hit.concept is not None:
            if is_concept_visible(
                hit.concept.id,
                store,
                caller.accessible_artifact_paths,
            ):
                filtered.append(hit)
        elif hit.artifact is not None:
            if hit.artifact.path in caller.accessible_artifact_paths:
                filtered.append(hit)
    return filtered
