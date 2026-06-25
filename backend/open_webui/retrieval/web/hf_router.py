"""Hugging Face Hub intent routing for the web-search pipeline.

Decides whether to dispatch a query to the dedicated :mod:`huggingface`
adapter alongside (or instead of) Kagi. Three trigger modes:

1. **Bang prefix** -- ``!hf rope scaling``, ``!huggingface qwen3``,
   ``!models``. Strips the bang from the query and routes exclusively to
   HF (no Kagi fanout).
2. **Explicit portal keyword** -- ``huggingface``, ``hf.co``,
   ``model card``, ``model hub``. The user clearly references HF; query
   left intact and Kagi runs in parallel.
3. **Open-weights family auto-route** -- the query contains a known
   open-weights model family as a whole word (``gemma``, ``llama``,
   ``qwen``, ``mistral``, ...). Forwards a canonical HF org as the
   ``author`` filter so we surface official releases (``google/gemma-3``,
   ``meta-llama/Llama-3.3``) rather than user-uploaded variants. Kagi
   runs in parallel.

Routing is fail-open: any unexpected error returns "no override" and the
search falls through to the configured engine.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)


class HfDecision(NamedTuple):
    """Result of HF intent routing.

    ``exclusive`` distinguishes the trigger modes:

    - **Bang match** -> ``exclusive=True``. User opted in to HF-only.
    - **Keyword / family match** -> ``exclusive=False``. Dispatcher fans
      out to Kagi in parallel for context.
    - **No match** -> ``matched=False``; other fields irrelevant.

    ``author`` is set on family auto-route matches (mapped via
    :data:`_FAMILY_TO_AUTHOR`) and on bang/keyword matches whose
    surrounding text mentions a known org by name (via
    :data:`_HF_KNOWN_ORGS`, e.g. "models from google on huggingface"
    -> ``google``). It scopes the HF API call to that org so results are
    pinned to the canonical model line -- this also collapses
    false-positive matches like ``granite countertops`` to near-zero
    HF hits.

    ``pipeline_tag`` is the HF Hub task tag we infer from intent words
    in the query (``embedding`` -> ``sentence-similarity``,
    ``rerank`` -> ``text-classification``, ``vlm`` ->
    ``image-text-to-text``, ...). When both ``author`` and
    ``pipeline_tag`` are set, ``query`` is reduced to the empty string:
    those two filters together already produce the precise listing the
    user asked for, and HF's substring-on-model-id ``search`` param
    would only narrow it back down to a long natural-language sentence
    that matches nothing.
    """

    matched: bool
    query: str
    author: Optional[str]
    exclusive: bool
    sort: Optional[str] = None
    pipeline_tag: Optional[str] = None


_HF_BANGS: frozenset[str] = frozenset(
    {
        '!hf',
        '!huggingface',
        '!hub',
        '!models',
        '!model',
    }
)


_HF_KEYWORDS: tuple[str, ...] = (
    'huggingface',
    'hf.co',
    'model card',
    'model hub',
)


# Curated map of open-weights model families -> canonical HF org/user.
#
# When the family appears as a whole word in the query, we route to HF and
# scope the search to that author. Names chosen to maximize signal vs.
# false-positive risk:
#
# - **Unambiguous in tech context** (gemma, qwen, mistral, deepseek, ...):
#   safe to auto-route. False-positive rate near zero.
# - **English collisions** (llama=animal, granite=rock, falcon=bird,
#   yi=common name): still routed, but the author filter (``meta-llama``,
#   ``ibm-granite``, ``tiiuae``, ``01-ai``) collapses unrelated queries
#   to ~0 HF results, and Kagi fanout handles the real answer.
#
# Add a family here when there's a clear canonical org and the name is
# either domain-specific or unambiguous enough that author-scoping
# defangs the collisions.
_FAMILY_TO_AUTHOR: dict[str, str] = {
    # Google
    'gemma': 'google',
    'gemma2': 'google',
    'gemma3': 'google',
    'gemma-2': 'google',
    'gemma-3': 'google',
    # Meta
    'llama': 'meta-llama',
    'llama2': 'meta-llama',
    'llama3': 'meta-llama',
    'llama4': 'meta-llama',
    'llama-2': 'meta-llama',
    'llama-3': 'meta-llama',
    'llama-4': 'meta-llama',
    'codellama': 'codellama',
    # Mistral AI
    'mistral': 'mistralai',
    'mixtral': 'mistralai',
    'codestral': 'mistralai',
    'pixtral': 'mistralai',
    # Alibaba / Qwen
    'qwen': 'Qwen',
    'qwen2': 'Qwen',
    'qwen3': 'Qwen',
    'qwen2.5': 'Qwen',
    'qwq': 'Qwen',
    # Microsoft
    'phi': 'microsoft',
    'phi-2': 'microsoft',
    'phi-3': 'microsoft',
    'phi-4': 'microsoft',
    'phi3': 'microsoft',
    'phi4': 'microsoft',
    # DeepSeek
    'deepseek': 'deepseek-ai',
    # IBM
    'granite': 'ibm-granite',
    # NVIDIA
    'nemotron': 'nvidia',
    # BigCode
    'starcoder': 'bigcode',
    'starcoder2': 'bigcode',
    # Databricks
    'dbrx': 'databricks',
    # Snowflake
    'arctic': 'Snowflake',
    # Cohere
    'aya': 'CohereLabs',
    'command-r': 'CohereLabs',
    'command-a': 'CohereLabs',
    # TII
    'falcon': 'tiiuae',
    # MosaicML
    'mpt': 'mosaicml',
    # InternLM
    'internlm': 'internlm',
    # Baichuan
    'baichuan': 'baichuan-inc',
    # Nous Research
    'hermes': 'NousResearch',
    'nous-hermes': 'NousResearch',
    # 01.AI
    'yi': '01-ai',
}


_BANG_RE = re.compile(r'^\s*(!\S+)\s+(.*)$', re.DOTALL)


# Recency cues that flip the HF sort to ``lastModified``. Inspected against
# the *original* user query (before we collapse to the family name for the
# search param), so "what is the latest gemma model" → sort by lastModified.
_RECENCY_WORDS: tuple[str, ...] = (
    'latest',
    'newest',
    'recent',
    'just released',
    'just dropped',
    'new release',
)


def _detect_sort(query: str) -> str:
    """Pick HF sort order from recency cues in the query."""
    q = query.lower()
    if any(w in q for w in _RECENCY_WORDS):
        return 'lastModified'
    return 'downloads'


# ── Author detection ────────────────────────────────────────────────────────
# Map of common human-readable org names -> canonical HF Hub org slug. Used
# to extract an explicit author from natural-language references like
# "models from google on huggingface", "anthropic's models", or the
# "<org>/<model>" reference form. Keys are lowercased aliases; values are
# the literal slug HF's ``?author=`` filter expects (which IS case-sensitive
# on the Hub for orgs like ``Qwen`` / ``CohereLabs`` / ``Snowflake``).
#
# Coverage criterion: include an alias when the alias is unambiguous in a
# tech context AND the org publishes enough models on HF that scoping
# ``author`` to it improves precision over a free-text search. False
# positives are bounded by requiring an HF trigger (bang, ``huggingface``
# keyword, or known model family) in the same query before author
# detection runs at all -- so a stray "google" in a generic web query
# does NOT route here.
_HF_KNOWN_ORGS: dict[str, str] = {
    'google': 'google',
    'google-bert': 'google-bert',
    'google-deepmind': 'google-deepmind',
    'deepmind': 'google-deepmind',
    'meta': 'meta-llama',
    'meta-llama': 'meta-llama',
    'facebook': 'facebook',
    'microsoft': 'microsoft',
    'mistral': 'mistralai',
    'mistralai': 'mistralai',
    'nvidia': 'nvidia',
    'ibm': 'ibm-granite',
    'ibm-granite': 'ibm-granite',
    'qwen': 'Qwen',
    'alibaba': 'Qwen',
    'deepseek': 'deepseek-ai',
    'deepseek-ai': 'deepseek-ai',
    'cohere': 'CohereLabs',
    'coherelabs': 'CohereLabs',
    'tii': 'tiiuae',
    'tiiuae': 'tiiuae',
    'databricks': 'databricks',
    'nous': 'NousResearch',
    'nousresearch': 'NousResearch',
    '01': '01-ai',
    '01-ai': '01-ai',
    '01.ai': '01-ai',
    'bigcode': 'bigcode',
    'snowflake': 'Snowflake',
    'baichuan': 'baichuan-inc',
    'internlm': 'internlm',
    'stability': 'stabilityai',
    'stabilityai': 'stabilityai',
    'mosaicml': 'mosaicml',
    'xai': 'xai-org',
    'amazon': 'amazon',
    'bytedance': 'ByteDance',
    'apple': 'apple',
    'openai': 'openai-community',
    'anthropic': 'Anthropic',
    # NOTE: do NOT alias bare "huggingface" / "hf" / "hf.co" to any org.
    # Those are our routing trigger keywords; a loose alias scan would
    # then pick them up as the author for any query that just mentions
    # the site (e.g. "best reranker on huggingface").
    'huggingfaceh4': 'HuggingFaceH4',
    'allen': 'allenai',
    'allenai': 'allenai',
    'eleuther': 'EleutherAI',
    'eleutherai': 'EleutherAI',
    'codellama': 'codellama',
    'sentence-transformers': 'sentence-transformers',
    'bge': 'BAAI',
    'baai': 'BAAI',
    'unsloth': 'unsloth',
}


# Match "<org>/<repo>" references anywhere in the query. Requires both
# halves to look like HF identifiers (alphanumerics + a small set of
# punctuation), which keeps unrelated slashes ("and/or", "yes/no") from
# tripping the author path. The actual alias check against
# :data:`_HF_KNOWN_ORGS` is what suppresses false positives.
_SLUG_PAIR_RE = re.compile(r'(?<![\w/])([A-Za-z0-9][\w.\-]*)/([A-Za-z0-9][\w.\-]*)')

# Natural-language author references. Each pattern captures the alias as
# group 1. Trailing context words ("organization", "team", "labs", "on
# huggingface") are encoded directly so the patterns don't overmatch.
# ``_ORG_TOKEN`` allows multi-token org names like "stability ai" and the
# hyphenated/dotted slug forms ("01-ai", "01.ai", "google-bert").
_ORG_TOKEN = r"[A-Za-z][A-Za-z0-9.\-]*(?:\s+ai)?"
_AUTHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\b(?:from|by|under(?:\s+the)?)\s+({_ORG_TOKEN})\b", re.IGNORECASE),
    re.compile(rf"\b({_ORG_TOKEN})['\u2019]s\b", re.IGNORECASE),
    re.compile(rf"\b({_ORG_TOKEN})\s+(?:organization|org|team|labs?|group)\b", re.IGNORECASE),
    re.compile(rf"\b({_ORG_TOKEN})\s+(?:on|at)\s+(?:hugging\s*face|hf\.co|the\s+hub)\b", re.IGNORECASE),
)


def _detect_author(query: str, *, loose: bool = False) -> Optional[str]:
    """Best-effort extraction of an HF org slug from a free-form query.

    Tries the ``<org>/<repo>`` slug-pair form first (strongest signal),
    then a small set of natural-language patterns. Returns the canonical
    HF org slug from :data:`_HF_KNOWN_ORGS`, or ``None`` if nothing
    matched. Unknown aliases fail closed: we'd rather miss an author
    than scope to a hallucinated org slug that returns zero results.

    When ``loose`` is True (used after a bang or HF keyword match, where
    the user has already opted into HF intent), we additionally accept
    any standalone token that matches a known alias. This catches
    "!hf google embedding models" style queries where the org is just
    listed as a bare word with no preceding preposition.
    """
    if not query:
        return None

    pair_match = _SLUG_PAIR_RE.search(query)
    if pair_match:
        alias = pair_match.group(1).lower()
        canonical = _HF_KNOWN_ORGS.get(alias)
        if canonical:
            return canonical

    for pattern in _AUTHOR_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        alias = match.group(1).lower().strip()
        canonical = _HF_KNOWN_ORGS.get(alias)
        if canonical:
            return canonical
        # Try collapsing whitespace -> "stability ai" -> "stabilityai"
        collapsed = re.sub(r'\s+', '', alias)
        canonical = _HF_KNOWN_ORGS.get(collapsed)
        if canonical:
            return canonical

    if loose:
        # Bare-token scan: walk the query and return the first token that
        # is a known org alias. Only enabled when the caller has already
        # confirmed HF intent (bang or keyword), so the false-positive
        # bar can be lower than in the strict path.
        for raw in re.findall(r"[A-Za-z0-9.\-_]+", query):
            canonical = _HF_KNOWN_ORGS.get(raw.lower())
            if canonical:
                return canonical
    return None


# ── Pipeline-tag detection ──────────────────────────────────────────────────
# Map intent phrases -> HF Hub ``pipeline_tag`` value. The HF API supports
# ``?pipeline_tag=<tag>`` as a server-side filter; passing it eliminates
# the "send a long sentence as ?search and match nothing" failure mode
# entirely when combined with ``author``.
#
# Order matters: the first matching entry wins. Specific multi-word
# phrases come before single-word ones so "image generation" routes to
# ``text-to-image`` before "image" alone (which we don't list because it's
# too ambiguous on its own).
_PIPELINE_INTENT: tuple[tuple[tuple[str, ...], str], ...] = (
    (('rerank', 'reranker', 'rerankers', 're-ranking'), 'text-classification'),
    (('embedding', 'embeddings', 'embed model', 'sentence-similarity',
      'sentence similarity', 'sentence transformer', 'sentence transformers',
      'text embedding'),
     'sentence-similarity'),
    (('feature extraction', 'feature-extraction'), 'feature-extraction'),
    (('image generation', 'text-to-image', 'text to image', 'diffusion model',
      'stable diffusion'),
     'text-to-image'),
    (('text-to-speech', 'text to speech', ' tts'), 'text-to-speech'),
    (('speech-to-text', 'speech to text', 'automatic speech recognition',
      ' asr', 'transcription model'),
     'automatic-speech-recognition'),
    (('image-text-to-text', 'vision language', 'vision-language',
      'multimodal', 'vlm '),
     'image-text-to-text'),
    (('translation model', 'machine translation'), 'translation'),
    (('summarization', 'summarizer'), 'summarization'),
    (('zero-shot classification', 'zero shot classification'),
     'zero-shot-classification'),
    (('fill-mask', 'fill mask', 'masked language', 'masked lm'), 'fill-mask'),
    (('question answering', 'qa model'), 'question-answering'),
    (('token classification', 'named entity recognition', ' ner '),
     'token-classification'),
    (('text classification', 'text classifier'), 'text-classification'),
    (('image classification', 'image classifier'), 'image-classification'),
    (('chat model', 'instruct model', 'instruction tuned',
      'instruction-tuned', 'llm', 'language model'),
     'text-generation'),
)


def _detect_pipeline_tag(query: str) -> Optional[str]:
    """Return an HF ``pipeline_tag`` value if the query expresses task intent.

    Padding with spaces lets us match short tokens like " tts" / " asr"
    without firing on inside-word occurrences ("tts" in "attestation").
    """
    if not query:
        return None
    haystack = f' {query.lower()} '
    for keywords, tag in _PIPELINE_INTENT:
        if any(kw in haystack for kw in keywords):
            return tag
    return None


# ── Query cleanup ───────────────────────────────────────────────────────────
# Stopwords to drop when reducing a natural-language question to the
# salient keywords HF's ``?search=`` substring filter actually understands.
# Includes generic English fillers, question words, and tokens we've
# already captured as ``author`` / ``pipeline_tag``.
_QUERY_STOPWORDS: frozenset[str] = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'available', 'be', 'best', 'by',
    'can', 'card', 'cards', 'checkpoint', 'checkpoints', 'do', 'does',
    'find', 'for', 'from',
    'good', 'group', 'has', 'have', 'how', 'huggingface', 'hugging', 'face',
    'hf', 'hf.co', 'hub', 'i', 'in', 'is', 'it', 'lab', 'labs', 'list',
    'me', 'model', 'models', 'new', 'newest', 'of', 'on', 'open', 'or',
    'org', 'organization', 'recent', 'release', 'released', 'releases',
    'search', 'show', 'team', 'that', 'the', 'their', 'there', 'these',
    'this', 'those', 'to', 'under', 'want', 'weights', 'what', 'whats',
    "what's", 'when', 'where', 'which', 'who', 'whom', 'whose', 'why',
    'with', 'you', 'your',
})


def _strip_to_keywords(
    query: str,
    *,
    author: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
) -> str:
    """Reduce ``query`` to keywords suitable for HF's ``?search=`` param.

    Behavior:

    - If both ``author`` and ``pipeline_tag`` are set, return ``''``.
      Those two filters alone already produce the precise listing the
      user asked for; adding the original sentence as a substring
      filter only narrows it back down to zero results.
    - Otherwise drop stopwords, the alias used for ``author`` (so
      "google embedding" doesn't become ``search=google%20embedding``
      after we already set ``author=google``), and the trigger phrases
      that produced ``pipeline_tag``.
    - Short tokens (<2 chars) are dropped unless purely numeric (model
      sizes like "7", "70" stay).
    - If the result is empty but the user clearly wanted *something*
      (i.e. neither author nor pipeline_tag is set), fall back to the
      original query rather than over-pruning to nothing.
    """
    if not query:
        return ''
    if author and pipeline_tag:
        return ''

    drop: set[str] = set()

    if author:
        for alias, canonical in _HF_KNOWN_ORGS.items():
            if canonical == author:
                drop.add(alias.lower())
                # Also drop any whitespace-split form so "stability ai" is
                # erased even though the alias map stores it collapsed.
                for part in re.split(r'[\s.\-]', alias):
                    if part:
                        drop.add(part.lower())

    if pipeline_tag:
        for keywords, tag in _PIPELINE_INTENT:
            if tag != pipeline_tag:
                continue
            for kw in keywords:
                for part in re.split(r'[\s.\-]', kw.strip().lower()):
                    if part:
                        drop.add(part)

    tokens = re.findall(r"[A-Za-z0-9.\-_]+", query)
    kept: list[str] = []
    for tok in tokens:
        lo = tok.lower()
        if lo in _QUERY_STOPWORDS or lo in drop:
            continue
        if len(lo) < 2 and not lo.isdigit():
            continue
        kept.append(tok)

    cleaned = ' '.join(kept).strip()
    if cleaned:
        return cleaned
    # Neither author nor pipeline_tag captured the intent and pruning ate
    # everything -- preserve the original query so the adapter still has
    # something to send.
    if not author and not pipeline_tag:
        return query.strip()
    return ''


def _build_family_pattern() -> re.Pattern[str]:
    """Compile a single regex matching any family name on word boundaries.

    Sort by length descending so multi-token aliases ("gemma-3",
    "llama-3", "nous-hermes") match before their bare-family prefix
    ("gemma", "llama", "hermes"). The first capture group identifies
    which family matched.

    Boundary semantics (looser than ``\\b``): the family name must NOT be
    flanked by letters or underscores -- but digits, hyphens, and dots are
    fine. This handles the common model-version pattern (``yi-34b``,
    ``falcon-7b``, ``phi-3.5``) while still preventing mid-word collisions
    (``apixtral``, ``gemmasomething``).
    """
    sorted_families = sorted(_FAMILY_TO_AUTHOR.keys(), key=len, reverse=True)
    alternation = '|'.join(re.escape(name) for name in sorted_families)
    return re.compile(rf'(?<![A-Za-z_])({alternation})(?![A-Za-z_])', re.IGNORECASE)


_FAMILY_RE = _build_family_pattern()


def route_query(query: str) -> HfDecision:
    """Detect Hugging Face / open-weights model intent.

    Returns an :class:`HfDecision`. Fails open -- any exception yields
    ``HfDecision(matched=False, query=query, author=None, exclusive=False)``.

    On a match, both ``author`` (from :data:`_HF_KNOWN_ORGS` /
    :data:`_FAMILY_TO_AUTHOR`) and ``pipeline_tag`` (from
    :data:`_PIPELINE_INTENT`) are extracted in addition to the trigger
    used. ``query`` is reduced to high-signal keywords for HF's
    substring-on-id ``?search=`` filter (or empty when ``author`` +
    ``pipeline_tag`` already pin the result set).
    """
    if not isinstance(query, str) or not query.strip():
        return HfDecision(False, query, None, False)

    try:
        bang_match = _BANG_RE.match(query)
        if bang_match:
            bang_token = bang_match.group(1).lower()
            if bang_token in _HF_BANGS:
                cleaned = bang_match.group(2).strip() or query
                author = _detect_author(cleaned, loose=True)
                pipeline_tag = _detect_pipeline_tag(cleaned)
                search_term = _strip_to_keywords(
                    cleaned, author=author, pipeline_tag=pipeline_tag
                )
                sort = _detect_sort(cleaned)
                log.debug(
                    'hf-router: bang %r -> huggingface; cleaned=%r author=%s '
                    'pipeline_tag=%s search=%r sort=%s',
                    bang_token,
                    cleaned,
                    author,
                    pipeline_tag,
                    search_term,
                    sort,
                )
                return HfDecision(
                    True, search_term, author, True, sort, pipeline_tag
                )

        haystack = query.lower()

        for kw in _HF_KEYWORDS:
            if kw in haystack:
                author = _detect_author(query, loose=True)
                pipeline_tag = _detect_pipeline_tag(query)
                search_term = _strip_to_keywords(
                    query, author=author, pipeline_tag=pipeline_tag
                )
                sort = _detect_sort(query)
                log.debug(
                    'hf-router: keyword %r matched in query=%r author=%s '
                    'pipeline_tag=%s search=%r sort=%s',
                    kw,
                    query,
                    author,
                    pipeline_tag,
                    search_term,
                    sort,
                )
                return HfDecision(
                    True, search_term, author, False, sort, pipeline_tag
                )

        family_match = _FAMILY_RE.search(query)
        if family_match:
            family = family_match.group(1).lower()
            author = _FAMILY_TO_AUTHOR.get(family)
            if author is None:
                # Try fallback to bare-family form (e.g. "gemma-3" -> "gemma")
                bare = re.split(r'[-.]', family, 1)[0]
                author = _FAMILY_TO_AUTHOR.get(bare)
            # Use the matched family as the search term, NOT the full query.
            # HF's ``search`` param is a substring match against model IDs --
            # "what is the latest gemma model" matches nothing, while
            # "gemma" + author=google returns the canonical lineup. The
            # ``author`` filter does the heavy lifting on relevance; sort
            # order (downloads vs lastModified) is decided by the adapter
            # from recency words in the cleaned query.
            search_term = family_match.group(1)
            pipeline_tag = _detect_pipeline_tag(query)
            # If the user added explicit task intent ("latest gemma
            # embedding model"), the pipeline_tag filter is more
            # precise than the family-name substring -- drop the
            # ``search`` term so we don't over-narrow.
            if pipeline_tag:
                search_term = ''
            sort = _detect_sort(query)
            log.debug(
                'hf-router: family %r -> huggingface author=%s pipeline_tag=%s '
                'search=%r sort=%s; original=%r',
                family,
                author,
                pipeline_tag,
                search_term,
                sort,
                query,
            )
            return HfDecision(
                True, search_term, author, False, sort, pipeline_tag
            )
    except Exception as e:
        log.debug('hf-router: routing error, falling back: %s', e)

    return HfDecision(False, query, None, False)
