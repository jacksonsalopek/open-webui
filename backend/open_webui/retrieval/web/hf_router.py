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

    ``author`` is set only on family auto-route matches (mapped from the
    matched family name via :data:`_FAMILY_TO_AUTHOR`). It scopes the HF
    API call to that org so results are pinned to the canonical model
    line -- this also collapses false-positive matches like
    ``granite countertops`` to near-zero HF hits.
    """

    matched: bool
    query: str
    author: Optional[str]
    exclusive: bool
    sort: Optional[str] = None


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
    """
    if not isinstance(query, str) or not query.strip():
        return HfDecision(False, query, None, False)

    try:
        bang_match = _BANG_RE.match(query)
        if bang_match:
            bang_token = bang_match.group(1).lower()
            if bang_token in _HF_BANGS:
                cleaned = bang_match.group(2).strip() or query
                sort = _detect_sort(cleaned)
                log.debug(
                    'hf-router: bang %r -> huggingface; cleaned=%r sort=%s',
                    bang_token,
                    cleaned,
                    sort,
                )
                return HfDecision(True, cleaned, None, True, sort)

        haystack = query.lower()

        for kw in _HF_KEYWORDS:
            if kw in haystack:
                sort = _detect_sort(query)
                log.debug(
                    'hf-router: keyword %r matched in query=%r sort=%s',
                    kw,
                    query,
                    sort,
                )
                return HfDecision(True, query, None, False, sort)

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
            sort = _detect_sort(query)
            log.debug(
                'hf-router: family %r -> huggingface author=%s search=%r sort=%s; original=%r',
                family,
                author,
                search_term,
                sort,
                query,
            )
            return HfDecision(True, search_term, author, False, sort)
    except Exception as e:
        log.debug('hf-router: routing error, falling back: %s', e)

    return HfDecision(False, query, None, False)
