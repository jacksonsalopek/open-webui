"""Pure-data concept extraction from code-splitter Document chunks.

The extractor composes glossary phrase matching, identifier tokenization, and
stopword classification into ``ExtractionResult`` records. It does not touch
the graph store — the builder (step 7) persists artifacts, concepts, and edges.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations

from langchain_core.documents import Document

from open_webui.retrieval.concepts.extraction.glossary import Glossary, PhraseHit
from open_webui.retrieval.concepts.extraction.identifiers import (
    CSHARP_DEFAULT_RULES,
    TokenRules,
    rules_for_language,
    tokenize,
    tokenize_text,
)
from open_webui.retrieval.concepts.extraction.stopwords import StopwordClass, classify
from open_webui.retrieval.concepts.schema import (
    Artifact,
    ArtifactKind,
    CoOccursWithProps,
    Concept,
    ConceptKind,
    DefinesProps,
    Edge,
    EdgeType,
    IsNamedInProps,
    ReferencesProps,
    edge_with_props,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractorOptions:
    """Tunable knobs for extraction. Defaults are tuned for the Lollipop
    Phase 1 acceptance target — change with care."""

    prune_stopword_cooccurrences: bool = True
    """When True, CO_OCCURS_WITH edges are NOT emitted between a stopword
    atomic and any other concept. Stopword concepts still receive REFERENCES
    and IS_NAMED_IN edges (so they remain queryable as nodes), they just
    don't contribute to the dense co-occurrence subgraph. Reason: stopwords
    co-occur with everything by definition, blowing up edge count to
    O(N_stopwords × N_total) per chunk with no semantic signal.

    Set False ONLY for diagnostic / debugging runs."""

    cooccurrence_concept_cap: int = 200
    """Existing cap from wave 4 — exposed as a knob for chunk-size tuning."""


DEFAULT_OPTIONS = ExtractorOptions()


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    chunk_text_length: int
    distinct_atomic_count: int
    distinct_phrase_count: int
    co_occurrence_edge_count: int
    stopword_atomic_count: int
    spans_consumed_by_phrases: int
    co_occurrence_pruned_count: int = 0


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Pure-data extraction output: artifact + concepts + edges from
    one chunk. Endpoint ids in `edges` use POSITIONAL conventions:
    the artifact is referenced by `artifact.id` (always 0 in fresh
    output); concepts are referenced by `concept.id` which equals
    their 1-indexed position in `concepts`. The builder (step 7)
    will replace these positional ids with real graph ids returned
    by `upsert_concept` / `upsert_artifact` before calling
    `upsert_edge`.
    """

    artifact: Artifact
    concepts: tuple[Concept, ...]
    edges: tuple[Edge, ...]
    stats: ExtractionStats


@dataclass
class _PhraseAggregate:
    phrase: PhraseHit
    hits: list[PhraseHit]


def extract(
    doc: Document,
    *,
    glossary: Glossary,
    rules: TokenRules | None = None,
    chunk_index: int | None = None,
    now: datetime | None = None,
    options: ExtractorOptions = DEFAULT_OPTIONS,
) -> ExtractionResult:
    """Extract one chunk into an ExtractionResult.

    `rules` defaults to `rules_for_language(doc.metadata['code_split_language'])`
    or `CSHARP_DEFAULT_RULES` if the language is missing. `chunk_index`
    defaults to `doc.metadata.get('chunk_index')` if present, else None.
    `now` defaults to `datetime.now(timezone.utc)`.
    """
    timestamp = now or datetime.now(timezone.utc)
    metadata = doc.metadata or {}
    page_content = doc.page_content or ''

    language = metadata.get('code_split_language')
    if language is None:
        log.debug('chunk missing code_split_language; falling back to CSHARP_DEFAULT_RULES')

    effective_rules = rules or (
        rules_for_language(language) if language else CSHARP_DEFAULT_RULES
    )

    resolved_chunk_index = chunk_index
    if resolved_chunk_index is None:
        raw_index = metadata.get('chunk_index')
        resolved_chunk_index = int(raw_index) if raw_index is not None else None

    artifact = Artifact(
        id=0,
        kind=ArtifactKind.CHUNK,
        path=str(metadata.get('source', '<unknown>')),
        chunk_index=resolved_chunk_index,
        language=str(language) if language is not None else None,
        byte_start=None,
        byte_end=None,
        last_modified_at=timestamp,
    )

    phrase_hits = glossary.match(page_content)
    phrase_spans = [hit.span for hit in phrase_hits]
    spans_consumed = sum(end - start for start, end in phrase_spans)

    masked_text = _mask_phrase_spans(page_content, phrase_spans)

    body_counts: Counter[str] = Counter(
        token
        for token in tokenize_text(masked_text, rules=effective_rules)
        if len(token) >= effective_rules.min_token_length or token.isdigit()
    )

    ast_symbol = metadata.get('ast_symbol') or ''
    if ast_symbol and not str(ast_symbol).strip():
        ast_symbol = ''

    symbol_tokens = (
        list(
            dict.fromkeys(
                token
                for token in tokenize(str(ast_symbol), rules=effective_rules)
                if len(token) >= effective_rules.min_token_length or token.isdigit()
            ),
        )
        if ast_symbol
        else []
    )
    defines_names = set(symbol_tokens)

    phrase_aggregates = _aggregate_phrase_hits(phrase_hits)
    atomic_names = sorted(set(body_counts.keys()) | defines_names)

    concepts: list[Concept] = []
    concept_id_by_key: dict[tuple[str, ConceptKind], int] = {}

    for phrase_name in sorted(phrase_aggregates.keys()):
        aggregate = phrase_aggregates[phrase_name]
        phrase = aggregate.phrase.phrase
        concept_id = len(concepts) + 1
        concept_id_by_key[(phrase_name, ConceptKind.PHRASE)] = concept_id
        concepts.append(
            Concept(
                id=concept_id,
                name=phrase.name,
                kind=ConceptKind.PHRASE,
                first_seen_at=timestamp,
                last_seen_at=timestamp,
                centrality_score=None,
                embedding=None,
                definition=phrase.definition,
                language_hint=str(language) if language is not None else None,
                original_tokens=phrase.surface_forms,
            ),
        )

    stopword_atomic_count = 0
    for atomic_name in atomic_names:
        concept_id = len(concepts) + 1
        concept_id_by_key[(atomic_name, ConceptKind.ATOMIC)] = concept_id
        if classify(atomic_name, language=language) != StopwordClass.NOT_STOPWORD:
            stopword_atomic_count += 1
        concepts.append(
            Concept(
                id=concept_id,
                name=atomic_name,
                kind=ConceptKind.ATOMIC,
                first_seen_at=timestamp,
                last_seen_at=timestamp,
                centrality_score=None,
                embedding=None,
                definition=None,
                language_hint=str(language) if language is not None else None,
                original_tokens=(atomic_name,),
            ),
        )

    edges, co_occurrence_pruned_count = _build_edges(
        artifact=artifact,
        concepts=concepts,
        concept_id_by_key=concept_id_by_key,
        defines_names=defines_names,
        body_counts=body_counts,
        phrase_aggregates=phrase_aggregates,
        timestamp=timestamp,
        language=language,
        options=options,
    )

    co_occurrence_edge_count = sum(
        1 for edge in edges if edge.type == EdgeType.CO_OCCURS_WITH
    )

    log.debug(
        'CO_OCCURS_WITH: emitted=%d pruned=%d (chunk_index=%s)',
        co_occurrence_edge_count,
        co_occurrence_pruned_count,
        resolved_chunk_index,
    )

    stats = ExtractionStats(
        chunk_text_length=len(page_content),
        distinct_atomic_count=sum(1 for concept in concepts if concept.kind == ConceptKind.ATOMIC),
        distinct_phrase_count=sum(1 for concept in concepts if concept.kind == ConceptKind.PHRASE),
        co_occurrence_edge_count=co_occurrence_edge_count,
        co_occurrence_pruned_count=co_occurrence_pruned_count,
        stopword_atomic_count=stopword_atomic_count,
        spans_consumed_by_phrases=spans_consumed,
    )

    return ExtractionResult(
        artifact=artifact,
        concepts=tuple(concepts),
        edges=tuple(edges),
        stats=stats,
    )


def extract_chunks(
    chunks: Sequence[Document],
    *,
    glossary: Glossary,
    rules: TokenRules | None = None,
    now: datetime | None = None,
    options: ExtractorOptions = DEFAULT_OPTIONS,
) -> list[ExtractionResult]:
    """Convenience helper for one whole file: auto-assigns
    `chunk_index` by enumerate(). Step 7 builder will use this."""
    return [
        extract(
            doc,
            glossary=glossary,
            rules=rules,
            chunk_index=index,
            now=now,
            options=options,
        )
        for index, doc in enumerate(chunks)
    ]


def _mask_phrase_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Replace matched phrase spans with spaces so identifier tokenization
    does not also emit the phrase's component tokens."""
    if not text or not spans:
        return text

    masked = list(text)
    for start, end in spans:
        for index in range(max(0, start), min(len(masked), end)):
            masked[index] = ' '
    return ''.join(masked)


def _aggregate_phrase_hits(hits: Sequence[PhraseHit]) -> dict[str, _PhraseAggregate]:
    aggregates: dict[str, _PhraseAggregate] = {}
    for hit in hits:
        name = hit.phrase.name
        if name not in aggregates:
            aggregates[name] = _PhraseAggregate(phrase=hit, hits=[hit])
        else:
            aggregates[name].hits.append(hit)
    return aggregates


def _occurrence_rank(
    concept: Concept,
    *,
    body_counts: Counter[str],
    phrase_aggregates: dict[str, _PhraseAggregate],
    defines_names: set[str],
) -> int:
    if concept.kind == ConceptKind.PHRASE:
        aggregate = phrase_aggregates.get(concept.name)
        return len(aggregate.hits) if aggregate is not None else 0
    body_count = body_counts.get(concept.name, 0)
    if concept.name in defines_names and body_count == 0:
        return 1
    return body_count


def _build_edges(
    *,
    artifact: Artifact,
    concepts: Sequence[Concept],
    concept_id_by_key: dict[tuple[str, ConceptKind], int],
    defines_names: set[str],
    body_counts: Counter[str],
    phrase_aggregates: dict[str, _PhraseAggregate],
    timestamp: datetime,
    language: str | None,
    options: ExtractorOptions,
) -> tuple[list[Edge], int]:
    edges: list[Edge] = []
    artifact_id = artifact.id

    for concept in concepts:
        if concept.kind == ConceptKind.ATOMIC and concept.name in defines_names:
            edges.append(
                edge_with_props(
                    src_id=artifact_id,
                    dst_id=concept.id,
                    props=DefinesProps(count=1),
                ),
            )
        elif concept.kind == ConceptKind.ATOMIC:
            count = body_counts.get(concept.name, 0)
            if count > 0:
                edges.append(
                    edge_with_props(
                        src_id=artifact_id,
                        dst_id=concept.id,
                        props=ReferencesProps(count=count),
                    ),
                )
        elif concept.kind == ConceptKind.PHRASE:
            aggregate = phrase_aggregates[concept.name]
            positions = tuple(hit.span[0] for hit in aggregate.hits)
            edges.append(
                edge_with_props(
                    src_id=artifact_id,
                    dst_id=concept.id,
                    props=ReferencesProps(
                        count=len(aggregate.hits),
                        positions=positions,
                    ),
                ),
            )

        edges.append(
            edge_with_props(
                src_id=concept.id,
                dst_id=artifact_id,
                props=IsNamedInProps(first_seen_at=timestamp),
            ),
        )

    co_occurrence_concepts = list(concepts)
    concept_cap = options.cooccurrence_concept_cap
    if len(co_occurrence_concepts) > concept_cap:
        log.warning(
            'chunk produced %d concepts for CO_OCCURS_WITH; capping to top %d '
            'by occurrence count',
            len(co_occurrence_concepts),
            concept_cap,
        )
        co_occurrence_concepts.sort(
            key=lambda concept: _occurrence_rank(
                concept,
                body_counts=body_counts,
                phrase_aggregates=phrase_aggregates,
                defines_names=defines_names,
            ),
            reverse=True,
        )
        co_occurrence_concepts = co_occurrence_concepts[:concept_cap]

    stopword_concept_ids: frozenset[int] = frozenset(
        concept.id
        for concept in concepts
        if concept.kind == ConceptKind.ATOMIC
        and classify(concept.name, language=language) != StopwordClass.NOT_STOPWORD
    )

    co_occurrence_pruned_count = 0
    for left, right in combinations(co_occurrence_concepts, 2):
        if options.prune_stopword_cooccurrences and (
            left.id in stopword_concept_ids or right.id in stopword_concept_ids
        ):
            co_occurrence_pruned_count += 2
            continue
        edges.append(
            edge_with_props(
                src_id=left.id,
                dst_id=right.id,
                props=CoOccursWithProps(weight=1.0, chunk_count=1),
            ),
        )
        edges.append(
            edge_with_props(
                src_id=right.id,
                dst_id=left.id,
                props=CoOccursWithProps(weight=1.0, chunk_count=1),
            ),
        )

    return edges, co_occurrence_pruned_count
