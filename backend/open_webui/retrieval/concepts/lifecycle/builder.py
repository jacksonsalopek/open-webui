"""Full-codebase ingest orchestrator for the concept knowledge graph.

Walks source roots, chunks via ``code_splitter``, extracts concepts/edges,
aggregates ``CO_OCCURS_WITH`` cross-file, applies builder-level prune,
and persists via store batch upserts. Optionally runs centrality precompute.

Re-running ``build(plan, store)`` against the same plan is idempotent:
the store's batch upsert semantics preserve stable ids and merge edges.
Orphan artifacts from deleted files are NOT cleaned up (Phase 1 limitation).
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_webui.retrieval.concepts.extraction.extractor import (
    DEFAULT_OPTIONS,
    ExtractionResult,
    ExtractorOptions,
    extract_chunks,
)
from open_webui.retrieval.concepts.extraction.glossary import Glossary
from open_webui.retrieval.concepts.lifecycle import centrality, idf
from open_webui.retrieval.concepts.schema import (
    CoOccursWithProps,
    Concept,
    ConceptKind,
    Edge,
    EdgeType,
    edge_with_props,
)
from open_webui.retrieval.concepts.store.protocol import GraphStore
from open_webui.retrieval.loaders.code_splitter import split_code

log = logging.getLogger(__name__)

_MAX_FILE_BYTES = 5 * 1024 * 1024

_SUFFIX_TO_LANGUAGE: dict[str, str] = {
    '.cs': 'csharp',
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
}


@dataclass(frozen=True, slots=True)
class BuilderPruneOptions:
    """Builder-level edge prune knobs.

    ``min_cooccurrence_weight`` is the second compression lever (extractor's
    stopword prune was the first). After all chunks are extracted and
    co-occurrence weights aggregated across the codebase, edges with
    weight < this threshold are NOT persisted. Default 2 means a pair
    must co-occur in at least 2 distinct chunks to make it into the graph.
    Set to 1 to disable. Documented carry-forward risk #2 from Phase 1
    status doc."""

    min_cooccurrence_weight: int = 2
    """Drop CO_OCCURS_WITH edges with aggregated weight below this.
    Applies AFTER cross-chunk aggregation."""


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Inputs to one rebuild run. Frozen so a run is reproducible from
    its plan."""

    roots: tuple[Path, ...]
    language_hint: str | None = None
    chunk_size: int = 1500
    chunk_overlap: int = 150
    glossary: Glossary | None = None
    extractor_options: ExtractorOptions = DEFAULT_OPTIONS
    include_globs: tuple[str, ...] = (
        '**/*.cs',
        '**/*.py',
        '**/*.ts',
        '**/*.tsx',
    )
    exclude_globs: tuple[str, ...] = (
        '**/bin/**',
        '**/obj/**',
        '**/node_modules/**',
        '**/__pycache__/**',
        '**/.git/**',
        '**/.venv/**',
        '**/dist/**',
        '**/build/**',
    )
    builder_prune: BuilderPruneOptions = field(default_factory=BuilderPruneOptions)


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Observability output. Wall-clock + counts + per-stage timing."""

    plan: BuildPlan
    started_at: datetime
    finished_at: datetime
    files_seen: int
    files_extracted: int
    files_skipped: int
    files_failed: int
    chunks_extracted: int
    concepts_upserted: int
    artifacts_upserted: int
    edges_emitted: int
    edges_persisted: int
    edges_pruned_by_weight: int
    centrality_computed_at: datetime | None
    idf_computed_at: datetime | None = None

    @property
    def wall_clock_ms(self) -> int:
        delta = self.finished_at - self.started_at
        return int(delta.total_seconds() * 1000)


ConceptKey = tuple[str, ConceptKind]
PairKey = tuple[ConceptKey, ConceptKey]


def _matches_glob(path: Path, pattern: str) -> bool:
    posix = path.as_posix()
    if path.match(pattern):
        return True
    return fnmatch.fnmatch(posix, pattern)


def _discover_files(plan: BuildPlan) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in plan.roots:
        if not root.exists():
            log.warning('build root does not exist: %s', root)
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            if path in seen:
                continue
            if not any(_matches_glob(path, pat) for pat in plan.include_globs):
                continue
            if any(_matches_glob(path, pat) for pat in plan.exclude_globs):
                continue
            seen.add(path)
            discovered.append(path)
    return sorted(discovered)


def _language_for_path(path: Path, hint: str | None) -> str | None:
    if hint is not None:
        return hint
    return _SUFFIX_TO_LANGUAGE.get(path.suffix.lower())


def _read_file(path: Path) -> tuple[str | None, str | None]:
    """Return (text, skip_reason). skip_reason is set when the file is skipped."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, f'stat failed: {exc}'

    if size == 0:
        return None, 'empty file'

    if size > _MAX_FILE_BYTES:
        return None, f'file exceeds {_MAX_FILE_BYTES} byte cap ({size} bytes)'

    try:
        return path.read_text(encoding='utf-8'), None
    except UnicodeDecodeError:
        return None, 'non-utf-8 content'


def _extract_file(
    path: Path,
    text: str,
    *,
    plan: BuildPlan,
    glossary: Glossary,
    now: datetime,
) -> list[ExtractionResult]:
    language = _language_for_path(path, plan.language_hint)
    docs = split_code(
        text,
        language,
        chunk_size=plan.chunk_size,
        chunk_overlap=plan.chunk_overlap,
        base_metadata={'source': str(path)},
    )
    if not docs:
        return []
    return extract_chunks(
        docs,
        glossary=glossary,
        now=now,
        options=plan.extractor_options,
    )


def _canonical_pair(a: ConceptKey, b: ConceptKey) -> PairKey:
    return (a, b) if a <= b else (b, a)


def _aggregate_co_occurrence(
    results: Sequence[ExtractionResult],
    *,
    min_weight: int,
) -> tuple[dict[PairKey, CoOccursWithProps], int, int, int]:
    """Aggregate CO_OCCURS_WITH across all extraction results.

    Returns (persisted_pairs, edges_emitted, edges_pruned_by_weight, edges_persisted_directed).
    """
    aggregated: dict[PairKey, CoOccursWithProps] = {}
    edges_emitted = 0

    for result in results:
        id_to_key: dict[int, ConceptKey] = {
            concept.id: (concept.name, concept.kind) for concept in result.concepts
        }
        for edge in result.edges:
            if edge.type != EdgeType.CO_OCCURS_WITH:
                continue
            edges_emitted += 1
            if edge.src_id >= edge.dst_id:
                continue
            key_a = id_to_key[edge.src_id]
            key_b = id_to_key[edge.dst_id]
            pair = _canonical_pair(key_a, key_b)
            incoming = CoOccursWithProps(
                weight=float(edge.properties['weight']),
                chunk_count=int(edge.properties['chunk_count']),
            )
            existing = aggregated.get(pair)
            if existing is None:
                aggregated[pair] = incoming
            else:
                aggregated[pair] = CoOccursWithProps(
                    weight=existing.weight + incoming.weight,
                    chunk_count=existing.chunk_count + incoming.chunk_count,
                )

    persisted: dict[PairKey, CoOccursWithProps] = {}
    pruned = 0
    for pair, props in aggregated.items():
        if props.weight < min_weight:
            pruned += 1
            continue
        persisted[pair] = props

    directed_persisted = len(persisted) * 2
    return persisted, edges_emitted, pruned, directed_persisted


def _dedupe_concepts(results: Sequence[ExtractionResult]) -> list[Concept]:
    merged: dict[ConceptKey, Concept] = {}
    for result in results:
        for concept in result.concepts:
            key = (concept.name, concept.kind)
            existing = merged.get(key)
            if existing is None:
                merged[key] = replace(concept, id=0)
                continue
            merged[key] = Concept(
                id=0,
                name=existing.name,
                kind=existing.kind,
                first_seen_at=min(existing.first_seen_at, concept.first_seen_at),
                last_seen_at=max(existing.last_seen_at, concept.last_seen_at),
                centrality_score=existing.centrality_score,
                embedding=existing.embedding or concept.embedding,
                definition=existing.definition or concept.definition,
                language_hint=existing.language_hint or concept.language_hint,
                original_tokens=tuple(
                    dict.fromkeys((*existing.original_tokens, *concept.original_tokens)),
                ),
            )
    return [merged[key] for key in sorted(merged)]


def _remap_edges(
    results: Sequence[ExtractionResult],
    artifact_ids: list[int],
    concept_id_map: dict[ConceptKey, int],
    co_occurrence: dict[PairKey, CoOccursWithProps],
) -> list[Edge]:
    edges: list[Edge] = []
    artifact_index = 0

    for result in results:
        real_artifact_id = artifact_ids[artifact_index]
        artifact_index += 1
        pos_to_key: dict[int, ConceptKey] = {
            concept.id: (concept.name, concept.kind) for concept in result.concepts
        }

        def remap(node_id: int) -> int:
            if node_id == 0:
                return real_artifact_id
            return concept_id_map[pos_to_key[node_id]]

        for edge in result.edges:
            if edge.type == EdgeType.CO_OCCURS_WITH:
                continue
            edges.append(
                Edge(
                    type=edge.type,
                    src_id=remap(edge.src_id),
                    dst_id=remap(edge.dst_id),
                    properties=edge.properties,
                ),
            )

    for (key_a, key_b), props in sorted(co_occurrence.items()):
        id_a = concept_id_map[key_a]
        id_b = concept_id_map[key_b]
        edges.append(edge_with_props(src_id=id_a, dst_id=id_b, props=props))
        edges.append(edge_with_props(src_id=id_b, dst_id=id_a, props=props))

    return edges


def build(
    plan: BuildPlan,
    store: GraphStore,
    *,
    compute_centrality: bool = True,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> BuildResult:
    """Run a full codebase rebuild against ``store``.

    Calls extractor for each file, batches per-file ExtractionResults,
    aggregates CO_OCCURS_WITH cross-file, applies BuilderPruneOptions,
    persists via store.upsert_*_batch. Optionally computes centrality
    at the end.

    Not transactional across files. If a file extraction raises, log
    + count + continue. If a store batch raises, abort the whole build
    (raise to caller).
    """
    started_at = datetime.now(timezone.utc)
    glossary = plan.glossary or Glossary.default()

    def _progress(phase: str, payload: dict[str, Any] | None = None) -> None:
        if progress_callback is not None:
            progress_callback(phase, payload or {})

    paths = _discover_files(plan)
    log.info('discover: %d candidate files under %d root(s)', len(paths), len(plan.roots))
    _progress('discover', {'files_seen': len(paths)})

    all_results: list[ExtractionResult] = []
    files_extracted = 0
    files_skipped = 0
    files_failed = 0
    chunks_extracted = 0
    total = len(paths)

    for idx, path in enumerate(paths):
        _progress('extract', {'file': str(path), 'idx': idx, 'total': total})
        text, skip_reason = _read_file(path)
        if skip_reason is not None:
            files_skipped += 1
            log.warning('skipped %s: %s', path, skip_reason)
            continue

        assert text is not None
        try:
            results = _extract_file(
                path,
                text,
                plan=plan,
                glossary=glossary,
                now=started_at,
            )
        except Exception:
            files_failed += 1
            log.exception('extract failed for %s', path)
            continue

        files_extracted += 1
        chunks_extracted += len(results)
        all_results.extend(results)
        log.debug('extracted %s: %d chunk(s)', path, len(results))

    co_occurrence, edges_emitted, edges_pruned, co_directed = _aggregate_co_occurrence(
        all_results,
        min_weight=plan.builder_prune.min_cooccurrence_weight,
    )
    log.info(
        'aggregate: co_occurrence_pairs=%d pruned=%d edges_emitted=%d',
        len(co_occurrence),
        edges_pruned,
        edges_emitted,
    )
    _progress(
        'aggregate',
        {
            'edges_emitted': edges_emitted,
            'edges_pruned_by_weight': edges_pruned,
            'co_occurrence_pairs': len(co_occurrence),
        },
    )

    artifacts = [result.artifact for result in all_results]
    log.info('persist_artifacts: %d artifact(s)', len(artifacts))
    _progress('persist_artifacts', {'count': len(artifacts)})
    artifact_ids = store.upsert_artifacts_batch(artifacts)

    concepts = _dedupe_concepts(all_results)
    log.info('persist_concepts: %d concept(s)', len(concepts))
    _progress('persist_concepts', {'count': len(concepts)})
    concept_ids = store.upsert_concepts_batch(concepts)
    concept_id_map = {
        (concept.name, concept.kind): concept_ids[i]
        for i, concept in enumerate(concepts)
    }

    edges = _remap_edges(all_results, artifact_ids, concept_id_map, co_occurrence)
    non_co_count = sum(
        1
        for result in all_results
        for edge in result.edges
        if edge.type != EdgeType.CO_OCCURS_WITH
    )
    edges_persisted = non_co_count + co_directed

    log.info('persist_edges: %d edge(s)', len(edges))
    _progress('persist_edges', {'count': len(edges)})
    store.upsert_edges_batch(edges)

    centrality_computed_at: datetime | None = None
    idf_computed_at: datetime | None = None
    if compute_centrality:
        log.info('centrality: computing PageRank variants')
        _progress('centrality', {})
        centrality_computed_at = centrality.compute_and_persist(store)
        log.info('idf: computing document-frequency scores')
        _progress('idf', {})
        idf_computed_at = idf.compute_and_persist(store)

    finished_at = datetime.now(timezone.utc)
    return BuildResult(
        plan=plan,
        started_at=started_at,
        finished_at=finished_at,
        files_seen=len(paths),
        files_extracted=files_extracted,
        files_skipped=files_skipped,
        files_failed=files_failed,
        chunks_extracted=chunks_extracted,
        concepts_upserted=len(concepts),
        artifacts_upserted=len(artifacts),
        edges_emitted=edges_emitted,
        edges_persisted=edges_persisted,
        edges_pruned_by_weight=edges_pruned,
        centrality_computed_at=centrality_computed_at,
        idf_computed_at=idf_computed_at,
    )
