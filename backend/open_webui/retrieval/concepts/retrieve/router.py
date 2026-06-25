"""Rule-based query intent classifier and retriever dispatcher."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from open_webui.retrieval.concepts.extraction.glossary import Glossary
from open_webui.retrieval.concepts.extraction.identifiers import (
    TokenRules,
    rules_for_language,
    tokenize,
    tokenize_text,
)
from open_webui.retrieval.concepts.extraction.stopwords import (
    ENGLISH_STOPWORDS,
    StopwordClass,
    classify,
)
from open_webui.retrieval.concepts.retrieve.base import (
    RetrievalHit,
    RetrievalQuery,
)
from open_webui.retrieval.concepts.retrieve.hybrid import (
    HybridRetriever,
    HybridRetrieverConfig,
)
from open_webui.retrieval.concepts.retrieve.neighborhood import (
    NeighborhoodRetriever,
    NeighborhoodRetrieverConfig,
    SeedFilter,
)
from open_webui.retrieval.concepts.schema import ConceptKind, EdgeType
from open_webui.retrieval.concepts.store.protocol import GraphStore

log = logging.getLogger(__name__)

_IDENTIFIER_IN_TEXT_RE = re.compile(
    r'\[[^\]]+\]'
    r'|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'
    r'(?:<[^>]*>)?(?:\?)?',
)

_FIND_SYMBOL_KEYWORDS = (
    'where is',
    'where are',
    'defined',
    'declaration',
    'declared',
)

_WHERE_USED_KEYWORDS = (
    'where used',
    'callers of',
    'usages of',
    'references to',
    'who calls',
)

_EXPLAIN_REGION_KEYWORDS = (
    'what is',
    'what does',
    'how does',
    'how is',
    'explain',
    'describe',
    'show me',
    'walk me through',
    'understand',
)

_FIND_CONCEPT_KEYWORDS = (
    'pattern',
    'concept',
    'idea',
    'principle',
)

_GENERATE_CODE_KEYWORDS = (
    'write',
    'create',
    'generate',
    'implement',
    'code for',
    'class that',
    'function that',
    'method that',
)


class Intent(str, Enum):
    """High-level retrieval intent. Each maps to a retriever + parameters.

    Phase 1 supports 5 intents matching the acceptance question kinds. Unknown
    intent → defaults to EXPLAIN_REGION (the most general)."""

    FIND_SYMBOL = 'find_symbol'
    WHERE_USED = 'where_used'
    EXPLAIN_REGION = 'explain_region'
    FIND_CONCEPT = 'find_concept'
    GENERATE_CODE = 'generate_code'


@dataclass(frozen=True, slots=True)
class ClassifiedIntent:
    """Output of the intent classifier. ``intent`` drives dispatch; the
    other fields are extracted hints for the chosen retriever."""

    intent: Intent
    extracted_symbols: tuple[str, ...]
    extracted_phrases: tuple[str, ...]
    raw_text: str
    classifier_provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RouterResult:
    """Public output: the classified intent + the ranked retrieval hits."""

    intent: ClassifiedIntent
    hits: list[RetrievalHit]
    retriever_used: str
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Tunable wiring of the router. Defaults are tuned for the Phase 1
    Lollipop acceptance set; the acceptance harness can override."""

    language: str = 'csharp'
    glossary: Glossary | None = None
    embed_fn: Callable[[str], tuple[float, ...]] | None = None
    top_k_default: int = 10
    neighborhood_radius_default: int = 2
    decompose_unresolved_tokens: bool = True
    """When True, query tokens that don't resolve to an atomic concept are
    decomposed via greedy longest-prefix matching against the store's concept
    index. Recovers source-vs-query mismatches like ``sendinput`` →
    ``send`` + ``input`` (source ``SendInput`` splits to two atomics whose
    join the query merges back into one literal token). Bounded by
    ``decompose_min_token_length``."""

    decompose_min_token_length: int = 4
    """Minimum length for a query token to be considered for decomposition.
    Shorter tokens are too prone to spurious matches against arbitrary
    prefixes (``ui`` → ``u`` + ``i`` is worse than no decomposition)."""

    decompose_min_part_length: int = 2
    """Minimum length of a recovered sub-token during decomposition. Recovered
    parts shorter than this are dropped — matches the tokenizer's
    ``min_token_length`` default so we don't synthesize concepts the
    extractor would have rejected."""

    max_neighbors_per_seed: int | None = None
    """Override for the underlying retriever's ``max_neighbors_per_seed``.
    ``None`` leaves the retriever default in place. The Lollipop corpus is
    dense (avg degree ~84), so the default 50-per-seed cap can starve common
    answer concepts (``toolbar``, ``selection``) of the multiplicity signal
    they need to survive IDF-based tiebreaking. Acceptance tests bump this
    to ~400 to widen the per-seed frontier."""


def classify_intent(
    text: str,
    *,
    config: RouterConfig | None = None,
) -> ClassifiedIntent:
    """Lightweight rule-based classifier. NO LLM call."""
    cfg = config or RouterConfig()
    glossary = cfg.glossary or Glossary.default()
    rules = rules_for_language(cfg.language)
    lower = text.lower()

    extracted_phrases = tuple(hit.phrase.name for hit in glossary.match(text))
    extracted_symbols = _extract_symbols(text, language=cfg.language, rules=rules)

    intent, provenance = _classify_rules(lower, extracted_phrases)

    return ClassifiedIntent(
        intent=intent,
        extracted_symbols=extracted_symbols,
        extracted_phrases=extracted_phrases,
        raw_text=text,
        classifier_provenance=provenance,
    )


def route(
    query: RetrievalQuery,
    store: GraphStore,
    *,
    config: RouterConfig | None = None,
) -> RouterResult:
    """Classify + dispatch + retrieve."""
    cfg = config or RouterConfig()
    started = time.perf_counter()

    classified = classify_intent(query.text, config=cfg)
    hits, retriever_used = _dispatch(classified, query, store, cfg)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        'route intent=%s retriever=%s hits=%d elapsed_ms=%d',
        classified.intent.value,
        retriever_used,
        len(hits),
        elapsed_ms,
    )

    return RouterResult(
        intent=classified,
        hits=hits,
        retriever_used=retriever_used,
        elapsed_ms=elapsed_ms,
    )


def _classify_rules(
    lower: str,
    extracted_phrases: tuple[str, ...],
) -> tuple[Intent, dict[str, Any]]:
    if _matches_where_used(lower):
        return Intent.WHERE_USED, _provenance(2, lower, _WHERE_USED_KEYWORDS)

    if _matches_find_symbol(lower):
        return Intent.FIND_SYMBOL, _provenance(1, lower, _FIND_SYMBOL_KEYWORDS)

    # Glossary phrases are a *strong* find_concept signal — promote ahead of
    # explain_region. A keyword-only find_concept match (e.g. ``pattern`` in
    # "text pattern") is too weak: queries like "how does X get Y when Z
    # doesn't have a text pattern?" are explain_region first and only
    # incidentally mention "pattern". Demote keyword-only find_concept after
    # explain_region.
    if extracted_phrases:
        return Intent.FIND_CONCEPT, _provenance(
            4,
            lower,
            _FIND_CONCEPT_KEYWORDS,
            extra={
                'glossary_phrases': list(extracted_phrases),
                'matched_via': 'glossary',
            },
        )

    if _matches_keywords(lower, _EXPLAIN_REGION_KEYWORDS):
        return Intent.EXPLAIN_REGION, _provenance(3, lower, _EXPLAIN_REGION_KEYWORDS)

    if _matches_find_concept(lower, extracted_phrases):
        return Intent.FIND_CONCEPT, _provenance(
            4,
            lower,
            _FIND_CONCEPT_KEYWORDS,
            extra={
                'glossary_phrases': list(extracted_phrases),
                'matched_via': 'keyword',
            },
        )

    if _matches_keywords(lower, _GENERATE_CODE_KEYWORDS):
        return Intent.GENERATE_CODE, _provenance(5, lower, _GENERATE_CODE_KEYWORDS)

    log.debug('No classifier rule matched; defaulting to EXPLAIN_REGION')
    return Intent.EXPLAIN_REGION, {
        'rule': 'default',
        'matched_substrings': [],
        'keywords_found': [],
    }


def _matches_where_used(lower: str) -> bool:
    if _matches_keywords(lower, _WHERE_USED_KEYWORDS):
        return True
    if ' used' in lower and ('where is' in lower or 'where are' in lower):
        if not any(kw in lower for kw in ('defined', 'declaration', 'declared')):
            return True
    return False


def _matches_find_symbol(lower: str) -> bool:
    if not _matches_keywords(lower, _FIND_SYMBOL_KEYWORDS):
        return False
    if ' used' in lower and not any(
        kw in lower for kw in ('defined', 'declaration', 'declared')
    ):
        return False
    return True


def _matches_find_concept(
    lower: str,
    extracted_phrases: tuple[str, ...],
) -> bool:
    if extracted_phrases:
        return True
    return _matches_keywords(lower, _FIND_CONCEPT_KEYWORDS)


def _matches_keywords(lower: str, keywords: tuple[str, ...]) -> bool:
    matched = [kw for kw in keywords if kw in lower]
    return bool(matched)


def _provenance(
    rule: int | str,
    lower: str,
    keywords: tuple[str, ...],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    matched_substrings = [kw for kw in keywords if kw in lower]
    result: dict[str, Any] = {
        'rule': rule,
        'matched_substrings': matched_substrings,
        'keywords_found': matched_substrings,
    }
    if extra:
        result.update(extra)
    return result


def _extract_symbols(
    text: str,
    *,
    language: str,
    rules: TokenRules,
) -> tuple[str, ...]:
    tokens = tokenize_text(text, rules=rules)
    uppercase_derived = _tokens_from_uppercase_identifiers(text, rules)

    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if classify(token, language=language) != StopwordClass.NOT_STOPWORD:
            continue
        if token in seen:
            continue
        if not _looks_like_identifier_token(token, uppercase_derived):
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def _tokens_from_uppercase_identifiers(text: str, rules: TokenRules) -> set[str]:
    derived: set[str] = set()
    for match in _IDENTIFIER_IN_TEXT_RE.finditer(text):
        ident = match.group(0)
        if any(ch.isupper() for ch in ident):
            derived.update(tokenize(ident, rules=rules))
    return derived


def _looks_like_identifier_token(token: str, uppercase_derived: set[str]) -> bool:
    if token in uppercase_derived:
        return True
    if any(ch.isdigit() for ch in token):
        return True
    return len(token) >= 4 and token not in ENGLISH_STOPWORDS


def _dispatch(
    classified: ClassifiedIntent,
    query: RetrievalQuery,
    store: GraphStore,
    cfg: RouterConfig,
) -> tuple[list[RetrievalHit], str]:
    intent = classified.intent
    if intent == Intent.GENERATE_CODE:
        intent = Intent.EXPLAIN_REGION

    top_k = query.top_k or cfg.top_k_default

    if intent == Intent.FIND_SYMBOL:
        return _route_find_symbol(classified, query, store, cfg, top_k)
    if intent == Intent.WHERE_USED:
        return _route_where_used(classified, query, store, cfg, top_k)
    if intent == Intent.FIND_CONCEPT:
        return _route_find_concept(classified, query, store, cfg, top_k)
    return _route_explain_region(classified, query, store, cfg, top_k)


def _route_find_symbol(
    classified: ClassifiedIntent,
    query: RetrievalQuery,
    store: GraphStore,
    cfg: RouterConfig,
    top_k: int,
) -> tuple[list[RetrievalHit], str]:
    """Resolve symbol seeds and walk the defining neighborhood.

    PascalCase identifiers tokenize into multiple atomic names — e.g.
    ``ToolbarViewModel`` → ``toolbar``, ``view``, ``model``. Each token is
    looked up via ``find_concept_by_name``; multiple seeds from one identifier
    are expected and fine.
    """
    seeds = _resolve_symbol_seeds(classified, store, cfg)
    retrieval_query = RetrievalQuery(
        text=query.text,
        embedding=query.embedding,
        seed_concept_ids=seeds,
        top_k=top_k,
        edge_types_filter=(EdgeType.DEFINES, EdgeType.IS_NAMED_IN),
        kind_filter=query.kind_filter,
    )
    retriever = NeighborhoodRetriever(
        _neighborhood_config(
            cfg,
            radius=1,
            edge_types=(EdgeType.DEFINES, EdgeType.IS_NAMED_IN),
            seed_filter=SeedFilter.NONE,
        ),
    )
    return retriever.retrieve(retrieval_query, store), retriever.name


def _route_where_used(
    classified: ClassifiedIntent,
    query: RetrievalQuery,
    store: GraphStore,
    cfg: RouterConfig,
    top_k: int,
) -> tuple[list[RetrievalHit], str]:
    seeds = _resolve_symbol_seeds(classified, store, cfg)
    retrieval_query = RetrievalQuery(
        text=query.text,
        embedding=query.embedding,
        seed_concept_ids=seeds,
        top_k=top_k,
        edge_types_filter=(EdgeType.REFERENCES, EdgeType.CO_OCCURS_WITH),
        kind_filter=query.kind_filter,
    )
    retriever = NeighborhoodRetriever(
        _neighborhood_config(
            cfg,
            radius=1,
            edge_types=(EdgeType.REFERENCES, EdgeType.CO_OCCURS_WITH),
            seed_filter=SeedFilter.NONE,
        ),
    )
    return retriever.retrieve(retrieval_query, store), retriever.name


def _route_find_concept(
    classified: ClassifiedIntent,
    query: RetrievalQuery,
    store: GraphStore,
    cfg: RouterConfig,
    top_k: int,
) -> tuple[list[RetrievalHit], str]:
    seeds = _resolve_phrase_seeds(classified, store)
    if not seeds:
        return _route_find_symbol(classified, query, store, cfg, top_k)

    retrieval_query = RetrievalQuery(
        text=query.text,
        embedding=query.embedding,
        seed_concept_ids=seeds,
        top_k=top_k,
        kind_filter=query.kind_filter,
    )
    retriever = NeighborhoodRetriever(
        _neighborhood_config(
            cfg,
            radius=cfg.neighborhood_radius_default,
            seed_filter=SeedFilter.NONE,
        ),
    )
    return retriever.retrieve(retrieval_query, store), retriever.name


def _route_explain_region(
    classified: ClassifiedIntent,
    query: RetrievalQuery,
    store: GraphStore,
    cfg: RouterConfig,
    top_k: int,
) -> tuple[list[RetrievalHit], str]:
    seeds = _resolve_all_seeds(classified, store, cfg)
    if seeds:
        retrieval_query = RetrievalQuery(
            text=query.text,
            embedding=query.embedding,
            seed_concept_ids=seeds,
            top_k=top_k,
            kind_filter=query.kind_filter,
        )
        retriever = NeighborhoodRetriever(
            _neighborhood_config(
                cfg,
                radius=cfg.neighborhood_radius_default,
                seed_filter=SeedFilter.NONE,
            ),
        )
        return retriever.retrieve(retrieval_query, store), retriever.name

    if cfg.embed_fn is not None:
        embedding = cfg.embed_fn(query.text)
        retrieval_query = RetrievalQuery(
            text=query.text,
            embedding=embedding,
            top_k=top_k,
            kind_filter=query.kind_filter,
        )
        retriever = HybridRetriever(HybridRetrieverConfig())
        return retriever.retrieve(retrieval_query, store), retriever.name

    retrieval_query = RetrievalQuery(
        text=query.text,
        embedding=query.embedding,
        top_k=top_k,
        kind_filter=query.kind_filter,
    )
    retriever = NeighborhoodRetriever()
    return retriever.retrieve(retrieval_query, store), retriever.name


def _resolve_symbol_seeds(
    classified: ClassifiedIntent,
    store: GraphStore,
    cfg: RouterConfig | None = None,
) -> tuple[int, ...]:
    cfg = cfg or RouterConfig()
    seeds: set[int] = set()
    unresolved: list[str] = []
    for symbol in classified.extracted_symbols:
        concept_id = store.find_concept_by_name(symbol, ConceptKind.ATOMIC)
        if concept_id is not None:
            seeds.add(concept_id)
        else:
            unresolved.append(symbol)

    if cfg.decompose_unresolved_tokens and unresolved:
        for symbol in unresolved:
            recovered = _decompose_token(
                symbol,
                store,
                min_token_length=cfg.decompose_min_token_length,
                min_part_length=cfg.decompose_min_part_length,
            )
            seeds.update(recovered)

    return tuple(sorted(seeds))


def _resolve_phrase_seeds(
    classified: ClassifiedIntent,
    store: GraphStore,
) -> tuple[int, ...]:
    seeds: set[int] = set()
    for phrase in classified.extracted_phrases:
        concept_id = store.find_concept_by_name(phrase, ConceptKind.PHRASE)
        if concept_id is not None:
            seeds.add(concept_id)
    return tuple(sorted(seeds))


def _resolve_all_seeds(
    classified: ClassifiedIntent,
    store: GraphStore,
    cfg: RouterConfig | None = None,
) -> tuple[int, ...]:
    seeds = set(_resolve_phrase_seeds(classified, store))
    seeds.update(_resolve_symbol_seeds(classified, store, cfg))
    return tuple(sorted(seeds))


def _neighborhood_config(
    cfg: RouterConfig,
    *,
    radius: int,
    seed_filter: SeedFilter,
    edge_types: tuple[EdgeType, ...] | None = None,
) -> NeighborhoodRetrieverConfig:
    """Build a ``NeighborhoodRetrieverConfig`` honoring router-level overrides
    like ``max_neighbors_per_seed``. Centralizes the wiring so every route
    picks up the same defaults instead of constructing the config inline."""
    kwargs: dict[str, object] = {
        'radius': radius,
        'seed_filter': seed_filter,
    }
    if edge_types is not None:
        kwargs['edge_types'] = edge_types
    if cfg.max_neighbors_per_seed is not None:
        kwargs['max_neighbors_per_seed'] = cfg.max_neighbors_per_seed
    return NeighborhoodRetrieverConfig(**kwargs)


def _decompose_token(
    token: str,
    store: GraphStore,
    *,
    min_token_length: int,
    min_part_length: int,
) -> list[int]:
    """Greedy longest-prefix decomposition against the store's atomic index.

    Recovers a query token that the source tokenizer would have split
    differently. Examples (assuming the listed atomics exist):

    - ``sendinput`` → ``[send_id, input_id]`` (source ``SendInput``)
    - ``noop`` → ``[no_id, op_id]`` (when the tokenizer's short-PascalCase
      merge didn't fire, e.g. on older builds)
    - ``uiautomation`` → ``[ui_id, automation_id]``

    The greedy scan tries the longest viable prefix first at each position and
    skips characters that don't form a recognized atomic — partial decompositions
    are still useful as long as at least one real concept is recovered.
    """
    if len(token) < min_token_length:
        return []
    if not token.isalpha() or not token.islower():
        return []

    seeds: list[int] = []
    i = 0
    n = len(token)
    while i < n:
        matched = False
        for length in range(n - i, min_part_length - 1, -1):
            sub = token[i:i + length]
            if length == n and i == 0:
                continue
            concept_id = store.find_concept_by_name(sub, ConceptKind.ATOMIC)
            if concept_id is not None:
                seeds.append(int(concept_id))
                i += length
                matched = True
                break
        if matched:
            continue
        # No prefix match at the current cursor. If we have not matched
        # anything yet, reject the token entirely — picking up an arbitrary
        # substring deep in an unrelated token (``pops`` → ``ops``) is worse
        # than no decomposition at all. If we have matched something, treat
        # the trailing remainder as noise only when it is below the part-length
        # floor (typically plural ``s``); otherwise stop here with what we
        # have rather than jumping forward heuristically.
        if not seeds:
            return []
        if n - i < min_part_length:
            break
        break

    deduped: list[int] = []
    seen: set[int] = set()
    for seed_id in seeds:
        if seed_id in seen:
            continue
        seen.add(seed_id)
        deduped.append(seed_id)

    if deduped:
        log.debug(
            'decomposed unresolved query token %r -> seeds=%s',
            token,
            deduped,
        )
    return deduped
