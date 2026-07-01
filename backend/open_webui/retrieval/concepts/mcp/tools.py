"""Core MCP tool handlers — thin wrappers over GraphStore + router."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from open_webui.retrieval.concepts.retrieve.base import RetrievalQuery
from open_webui.retrieval.concepts.retrieve.router import (
    RouterConfig,
    classify_intent,
    route,
)
from open_webui.retrieval.concepts.schema import Concept, ConceptKind, EdgeType
from open_webui.retrieval.concepts.store.protocol import GraphStore

from .context import CallerContext
from .serialize import (
    assert_no_embedding,
    serialize_artifact,
    serialize_concept,
    serialize_hit,
    serialize_neighbor,
    serialize_path_step,
)
from .visibility import filter_artifacts, filter_concepts, filter_hits, is_concept_visible

_NEIGHBORHOOD_EDGE_TYPES = (
    EdgeType.CO_OCCURS_WITH,
    EdgeType.DEFINES,
    EdgeType.REFERENCES,
)
_WHERE_USED_EDGE_TYPES = (EdgeType.IS_NAMED_IN, EdgeType.REFERENCES)
_IMPACT_EDGE_TYPES = (
    EdgeType.CO_OCCURS_WITH,
    EdgeType.DEFINES,
    EdgeType.REFERENCES,
    EdgeType.IS_NAMED_IN,
)


@dataclass(frozen=True, slots=True)
class ResolvedConcept:
    concept: Concept


@dataclass(frozen=True, slots=True)
class ResolveAmbiguous:
    candidates: tuple[Concept, ...]


@dataclass(frozen=True, slots=True)
class ResolveNotFound:
    name: str
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class ResolveUnavailable:
    pass


ResolveResult = (
    ResolvedConcept | ResolveAmbiguous | ResolveNotFound | ResolveUnavailable
)


def _base_provenance(
    caller: CallerContext,
    wrapped_methods: list[str],
    *,
    store_available: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    return {
        'store_available': store_available,
        'acl_applied': not caller.bypass_acl,
        'accessible_artifact_count': len(caller.accessible_artifact_paths),
        'wrapped_methods': wrapped_methods,
        'router_intent': None,
        'retriever_used': None,
        **extra,
    }


def _envelope(
    tool: str,
    status: str,
    elapsed_ms: int,
    provenance: dict[str, Any],
    **payload: Any,
) -> dict[str, Any]:
    result = {
        'status': status,
        'tool': tool,
        'elapsed_ms': elapsed_ms,
        'provenance': provenance,
        **payload,
    }
    assert_no_embedding(result)
    return result


def _parse_kind(kind: str | None) -> ConceptKind | None:
    if kind is None:
        return None
    try:
        return ConceptKind(kind)
    except ValueError as exc:
        raise ValueError(f'invalid kind: {kind!r}') from exc


def resolve_concept(
    name: str,
    kind: str | None,
    store: GraphStore | None,
    caller: CallerContext,
) -> ResolveResult:
    if store is None:
        return ResolveUnavailable()

    if kind is not None:
        try:
            parsed_kind = _parse_kind(kind)
        except ValueError:
            raise
        concept_id = store.find_concept_by_name(name, parsed_kind)
        if concept_id is None:
            return ResolveNotFound(name=name, kind=kind)
        resolved_id = store.resolve_alias(concept_id)
        concept = store.get_concept(resolved_id)
        if concept is None:
            return ResolveNotFound(name=name, kind=kind)
        if not caller.bypass_acl and not is_concept_visible(
            concept.id,
            store,
            caller.accessible_artifact_paths,
        ):
            return ResolveNotFound(name=name, kind=kind)
        return ResolvedConcept(concept=concept)

    matches = [c for c in store.list_concepts() if c.name == name]
    matches = filter_concepts(matches, store, caller)
    if not matches:
        return ResolveNotFound(name=name)
    if len(matches) > 1:
        return ResolveAmbiguous(candidates=tuple(matches))

    concept_id = store.resolve_alias(matches[0].id)
    concept = store.get_concept(concept_id)
    if concept is None:
        return ResolveNotFound(name=name)
    if not caller.bypass_acl and not is_concept_visible(
        concept.id,
        store,
        caller.accessible_artifact_paths,
    ):
        return ResolveNotFound(name=name)
    return ResolvedConcept(concept=concept)


def _resolution_payload(result: ResolveResult) -> dict[str, Any]:
    if isinstance(result, ResolveAmbiguous):
        return {
            'status': 'ambiguous',
            'candidates': [
                serialize_concept(concept) for concept in result.candidates
            ],
        }
    if isinstance(result, ResolveNotFound):
        payload: dict[str, Any] = {'status': 'not_found'}
        if result.kind is not None:
            payload['name'] = result.name
            payload['kind'] = result.kind
        else:
            payload['concept_name'] = result.name
        return payload
    if isinstance(result, ResolveUnavailable):
        return {
            'status': 'unavailable',
            'message': 'Concept graph unavailable',
        }
    return {}


def _edge_type_between(
    store: GraphStore,
    src_id: int,
    dst_id: int,
    edge_types: tuple[EdgeType, ...],
) -> EdgeType | None:
    best_type: EdgeType | None = None
    best_weight = -1.0
    for edge_type in edge_types:
        neighbors = store.neighborhood(
            src_id,
            radius=1,
            edge_types=(edge_type,),
            limit=1000,
        )
        if any(neighbor.id == dst_id for neighbor in neighbors):
            weight = 1.0
            if best_type is None or weight > best_weight:
                best_type = edge_type
                best_weight = weight
    return best_type


def _neighbors_with_edges(
    store: GraphStore,
    anchor_id: int,
    *,
    radius: int,
    limit: int,
    edge_types: tuple[EdgeType, ...],
    caller: CallerContext,
) -> list[dict[str, Any]]:
    resolved = store.resolve_alias(anchor_id)
    seen: set[int] = {resolved}
    frontier: list[int] = [resolved]
    collected: list[tuple[Concept, EdgeType, int]] = []

    for hop in range(1, radius + 1):
        next_frontier: list[int] = []
        for node_id in frontier:
            ring = store.neighborhood(
                node_id,
                radius=1,
                edge_types=edge_types,
                limit=limit,
            )
            for neighbor in ring:
                if neighbor.id in seen:
                    continue
                seen.add(neighbor.id)
                next_frontier.append(neighbor.id)
                edge_type = _edge_type_between(
                    store,
                    node_id,
                    neighbor.id,
                    edge_types,
                ) or EdgeType.CO_OCCURS_WITH
                collected.append((neighbor, edge_type, hop))
        frontier = next_frontier
        if not frontier:
            break

    visible = filter_concepts([concept for concept, _, _ in collected], store, caller)
    visible_ids = {concept.id for concept in visible}
    result: list[dict[str, Any]] = []
    for concept, edge_type, hop in collected:
        if concept.id not in visible_ids:
            continue
        result.append(serialize_neighbor(concept, edge_type, hop_distance=hop))
        if len(result) >= limit:
            break
    return result


def _one_hop_neighbors(
    store: GraphStore,
    anchor_id: int,
    *,
    limit: int,
    edge_types: tuple[EdgeType, ...],
    caller: CallerContext,
) -> list[dict[str, Any]]:
    neighbors = store.neighborhood(
        anchor_id,
        radius=1,
        edge_types=edge_types,
        limit=limit,
    )
    visible = filter_concepts(neighbors, store, caller)
    summaries: list[dict[str, Any]] = []
    for neighbor in visible[:limit]:
        edge_type = _edge_type_between(
            store,
            anchor_id,
            neighbor.id,
            edge_types,
        ) or EdgeType.CO_OCCURS_WITH
        summaries.append(serialize_neighbor(neighbor, edge_type))
    return summaries


def find_concept(
    name: str,
    kind: str | None = None,
    *,
    store: GraphStore | None,
    caller: CallerContext,
) -> dict[str, Any]:
    started = time.perf_counter()
    tool = 'find_concept'

    if store is None:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'unavailable',
            elapsed,
            _base_provenance(caller, [], store_available=False),
            message='Concept graph unavailable',
        )

    if not name:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'error',
            elapsed,
            _base_provenance(caller, []),
            message='name must be non-empty',
        )

    try:
        resolved = resolve_concept(name, kind, store, caller)
    except ValueError:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'error',
            elapsed,
            _base_provenance(caller, ['find_concept_by_name']),
            message=f'invalid kind: {kind!r}',
        )

    if not isinstance(resolved, ResolvedConcept):
        elapsed = int((time.perf_counter() - started) * 1000)
        payload = _resolution_payload(resolved)
        return _envelope(
            tool,
            payload.pop('status'),
            elapsed,
            _base_provenance(
                caller,
                ['find_concept_by_name', 'resolve_alias', 'get_concept'],
            ),
            **payload,
        )

    concept = resolved.concept
    neighbors = _one_hop_neighbors(
        store,
        concept.id,
        limit=10,
        edge_types=_NEIGHBORHOOD_EDGE_TYPES,
        caller=caller,
    )
    elapsed = int((time.perf_counter() - started) * 1000)
    return _envelope(
        tool,
        'ok',
        elapsed,
        _base_provenance(
            caller,
            [
                'find_concept_by_name',
                'resolve_alias',
                'get_concept',
                'neighborhood',
            ],
        ),
        concept=serialize_concept(concept),
        neighbors=neighbors,
    )


def where_used(
    concept_name: str,
    *,
    store: GraphStore | None,
    caller: CallerContext,
) -> dict[str, Any]:
    started = time.perf_counter()
    tool = 'where_used'

    if store is None:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'unavailable',
            elapsed,
            _base_provenance(caller, [], store_available=False),
            message='Concept graph unavailable',
        )

    resolved = resolve_concept(concept_name, None, store, caller)
    if not isinstance(resolved, ResolvedConcept):
        elapsed = int((time.perf_counter() - started) * 1000)
        payload = _resolution_payload(resolved)
        if payload.get('status') == 'not_found':
            payload['concept_name'] = concept_name
        return _envelope(
            tool,
            payload.pop('status'),
            elapsed,
            _base_provenance(
                caller,
                ['find_concept_by_name', 'get_concept'],
            ),
            **payload,
        )

    concept = resolved.concept
    artifacts = store.list_artifacts_for_concept(
        concept.id,
        edge_types=_WHERE_USED_EDGE_TYPES,
        limit=20,
    )
    visible = filter_artifacts(list(artifacts), caller)
    serialized = [
        serialize_artifact(artifact, edge_type=EdgeType.IS_NAMED_IN)
        for artifact in visible
    ]
    elapsed = int((time.perf_counter() - started) * 1000)
    provenance = _base_provenance(
        caller,
        ['find_concept_by_name', 'get_concept', 'list_artifacts_for_concept'],
    )
    if not serialized:
        return _envelope(
            tool,
            'empty',
            elapsed,
            provenance,
            concept=serialize_concept(concept),
            artifacts=[],
        )
    return _envelope(
        tool,
        'ok',
        elapsed,
        provenance,
        concept=serialize_concept(concept),
        artifacts=serialized,
    )


def explain_region(
    concept_name: str,
    radius: int = 2,
    *,
    store: GraphStore | None,
    caller: CallerContext,
) -> dict[str, Any]:
    started = time.perf_counter()
    tool = 'explain_region'

    if store is None:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'unavailable',
            elapsed,
            _base_provenance(caller, [], store_available=False),
            message='Concept graph unavailable',
        )

    if radius < 1 or radius > 4:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'error',
            elapsed,
            _base_provenance(caller, []),
            message=f'radius must be between 1 and 4, got {radius}',
        )

    resolved = resolve_concept(concept_name, None, store, caller)
    if not isinstance(resolved, ResolvedConcept):
        elapsed = int((time.perf_counter() - started) * 1000)
        payload = _resolution_payload(resolved)
        return _envelope(
            tool,
            payload.pop('status'),
            elapsed,
            _base_provenance(caller, ['find_concept_by_name', 'get_concept']),
            **payload,
        )

    concept = resolved.concept
    neighbors = _neighbors_with_edges(
        store,
        concept.id,
        radius=radius,
        limit=50,
        edge_types=_NEIGHBORHOOD_EDGE_TYPES,
        caller=caller,
    )
    ppr_scores = store.personalized_pagerank(
        [concept.id],
        edge_types=_NEIGHBORHOOD_EDGE_TYPES,
    )
    ppr_ranked: list[dict[str, Any]] = []
    for concept_id, score in sorted(
        ppr_scores.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        ppr_concept = store.get_concept(concept_id)
        if ppr_concept is None:
            continue
        if not caller.bypass_acl and not is_concept_visible(
            ppr_concept.id,
            store,
            caller.accessible_artifact_paths,
        ):
            continue
        ppr_ranked.append(
            {
                'concept': serialize_concept(ppr_concept),
                'ppr_score': score,
            },
        )
        if len(ppr_ranked) >= 20:
            break

    elapsed = int((time.perf_counter() - started) * 1000)
    provenance = _base_provenance(
        caller,
        [
            'find_concept_by_name',
            'get_concept',
            'neighborhood',
            'personalized_pagerank',
        ],
        radius=radius,
    )
    if not neighbors and not ppr_ranked:
        return _envelope(
            tool,
            'empty',
            elapsed,
            provenance,
            concept=serialize_concept(concept),
            neighbors=[],
            ppr_ranked=[],
        )
    return _envelope(
        tool,
        'ok',
        elapsed,
        provenance,
        concept=serialize_concept(concept),
        neighbors=neighbors,
        ppr_ranked=ppr_ranked,
    )


def trace_neighborhood(
    query: str,
    *,
    store: GraphStore | None,
    caller: CallerContext,
    router_config: RouterConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    tool = 'trace_neighborhood'
    cfg = router_config or RouterConfig()

    if store is None:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'unavailable',
            elapsed,
            _base_provenance(caller, [], store_available=False),
            message='Concept graph unavailable',
        )

    if not query:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'error',
            elapsed,
            _base_provenance(caller, []),
            message='query must be non-empty',
        )

    classified = classify_intent(query, config=cfg)
    result = route(
        RetrievalQuery(text=query, top_k=20),
        store,
        config=cfg,
    )
    hits = filter_hits(result.hits, store, caller)
    serialized_hits = [serialize_hit(hit) for hit in hits[:20]]

    elapsed = int((time.perf_counter() - started) * 1000)
    provenance = _base_provenance(
        caller,
        ['classify_intent', 'route'],
        router_intent=result.intent.intent.value,
        retriever_used=result.retriever_used,
        extracted_symbols=list(classified.extracted_symbols),
        extracted_phrases=list(classified.extracted_phrases),
        classifier_provenance=dict(classified.classifier_provenance),
    )
    payload = {
        'intent': result.intent.intent.value,
        'hits': serialized_hits,
        'extracted_symbols': list(classified.extracted_symbols),
        'extracted_phrases': list(classified.extracted_phrases),
        'classifier_provenance': dict(classified.classifier_provenance),
    }
    if result.hits and not serialized_hits:
        return _envelope(tool, 'empty', elapsed, provenance, **payload)
    return _envelope(tool, 'ok', elapsed, provenance, **payload)


def impact_analysis(
    concept_a: str,
    concept_b: str,
    *,
    store: GraphStore | None,
    caller: CallerContext,
) -> dict[str, Any]:
    started = time.perf_counter()
    tool = 'impact_analysis'

    if store is None:
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'unavailable',
            elapsed,
            _base_provenance(caller, [], store_available=False),
            message='Concept graph unavailable',
        )

    resolved_a = resolve_concept(concept_a, None, store, caller)
    resolved_b = resolve_concept(concept_b, None, store, caller)

    if isinstance(resolved_a, ResolveAmbiguous):
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'ambiguous',
            elapsed,
            _base_provenance(caller, ['find_concept_by_name', 'get_concept']),
            endpoint='concept_a',
            candidates=[serialize_concept(c) for c in resolved_a.candidates],
        )
    if isinstance(resolved_b, ResolveAmbiguous):
        elapsed = int((time.perf_counter() - started) * 1000)
        return _envelope(
            tool,
            'ambiguous',
            elapsed,
            _base_provenance(caller, ['find_concept_by_name', 'get_concept']),
            endpoint='concept_b',
            candidates=[serialize_concept(c) for c in resolved_b.candidates],
        )

    missing: list[str] = []
    if not isinstance(resolved_a, ResolvedConcept):
        missing.append('concept_a')
    if not isinstance(resolved_b, ResolvedConcept):
        missing.append('concept_b')
    if missing:
        elapsed = int((time.perf_counter() - started) * 1000)
        payload: dict[str, Any] = {'missing': missing[0] if len(missing) == 1 else 'both'}
        return _envelope(
            tool,
            'not_found',
            elapsed,
            _base_provenance(caller, ['find_concept_by_name', 'get_concept']),
            **payload,
        )

    concept_left = resolved_a.concept
    concept_right = resolved_b.concept
    path_concepts = list(
        store.shortest_path(
            concept_left.id,
            concept_right.id,
            edge_types=_IMPACT_EDGE_TYPES,
            max_hops=6,
        ),
    )
    visible_in_path = [
        node
        for node in path_concepts
        if caller.bypass_acl
        or is_concept_visible(node.id, store, caller.accessible_artifact_paths)
    ]
    path_found = False
    if (
        len(visible_in_path) >= 2
        and visible_in_path[0].id == concept_left.id
        and visible_in_path[-1].id == concept_right.id
    ):
        path_found = True
        for index in range(len(visible_in_path) - 1):
            if _edge_type_between(
                store,
                visible_in_path[index].id,
                visible_in_path[index + 1].id,
                _IMPACT_EDGE_TYPES,
            ) is None:
                path_found = False
                break

    steps: list[dict[str, Any]] = []
    if path_found:
        for index, node in enumerate(visible_in_path):
            next_edge: EdgeType | None = None
            if index + 1 < len(visible_in_path):
                next_edge = _edge_type_between(
                    store,
                    node.id,
                    visible_in_path[index + 1].id,
                    _IMPACT_EDGE_TYPES,
                )
            steps.append(serialize_path_step(node, next_edge))

    elapsed = int((time.perf_counter() - started) * 1000)
    provenance = _base_provenance(
        caller,
        ['find_concept_by_name', 'get_concept', 'shortest_path'],
        max_hops=6,
        path_length=len(steps),
    )
    if not path_found:
        message = 'No path within 6 hops'
        if path_concepts and not visible_path:
            message = 'No path within 6 hops (ACL may hide edges)'
        return _envelope(
            tool,
            'ok',
            elapsed,
            provenance,
            concept_a=serialize_concept(concept_left),
            concept_b=serialize_concept(concept_right),
            path=[],
            path_found=False,
            message=message,
        )
    return _envelope(
        tool,
        'ok',
        elapsed,
        provenance,
        concept_a=serialize_concept(concept_left),
        concept_b=serialize_concept(concept_right),
        path=steps,
        path_found=True,
    )
