"""Academic-intent routing for the web-search pipeline.

Decides whether to redirect a query from the default Kagi engine to the
dedicated :mod:`arxiv` adapter. Two trigger modes, mirroring
``kagi_lenses.route_query``:

1. **Bang prefix** — ``!arxiv quantum computing``. The bang is stripped from
   the query; routing fires unconditionally on match.
2. **Keyword scan** — substring match against the (lowercased) query. The
   query is left untouched since the keyword may carry intent (e.g.
   "preprint" is part of the search).

When a bang of the form ``!cs.LG``, ``!math.AP``, ``!stat.ML`` matches an
arXiv category we also forward that as a category filter so the adapter can
narrow to that subject area.

Routing is fail-open: any unexpected error returns "no override" and the
search falls through to the configured engine (kagi).
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)


class ArxivDecision(NamedTuple):
    """Result of arXiv intent routing.

    ``exclusive`` distinguishes the two trigger modes:

    - **Bang match** → ``exclusive=True``. The user explicitly opted in to
      arXiv-only results (``!arxiv quantum``), so the dispatcher should skip
      the Kagi fanout.
    - **Keyword match** → ``exclusive=False``. The user used generic
      academic vocabulary (``preprint``, ``research paper``); the dispatcher
      can fan out to Kagi in parallel for broader coverage.
    - **No match** → ``matched=False``, ``exclusive`` is irrelevant.
    """

    matched: bool
    query: str
    category: Optional[str]
    exclusive: bool


# Bangs that unambiguously mean "go to arXiv". Kept short and recognizable;
# do NOT add anything overloaded (e.g. ``!papers`` is ambiguous with Google
# Scholar / Semantic Scholar but those aren't wired up here so we own it).
_ARXIV_BANGS: frozenset[str] = frozenset(
    {
        '!arxiv',
        '!ax',
        '!papers',
        '!paper',
        '!preprint',
        '!preprints',
    }
)

# Keyword triggers. We're deliberately conservative — generic words like
# "study", "research", "thesis" cover too much non-arXiv ground (legal,
# medical, social-science) and would route too aggressively. The list below
# only fires when the user explicitly invokes arXiv or its surrounding
# vocabulary (preprint, peer-reviewed paper search, etc.).
_ARXIV_KEYWORDS: tuple[str, ...] = (
    'arxiv',
    'preprint',
    'research paper',
    'research papers',
    'academic paper',
    'academic papers',
    'peer reviewed paper',
    'peer-reviewed paper',
    'scientific paper',
    'scientific papers',
)


_BANG_RE = re.compile(r'^\s*(!\S+)\s+(.*)$', re.DOTALL)

# arXiv subject classifications follow ``<archive>.<subject>`` (e.g. ``cs.LG``,
# ``math.AP``, ``stat.ML``, ``q-bio.NC``). A bare ``!cs`` is too coarse to be
# useful (the whole computer-science archive) so we require the dotted form.
_CATEGORY_BANG_RE = re.compile(r'^!(?P<archive>[a-z][a-z\-]+)\.(?P<subject>[A-Z]{2})$')


def _bang_to_category(bang: str) -> Optional[str]:
    """Map ``!cs.LG`` → ``cs.LG``; return None for non-category bangs."""
    match = _CATEGORY_BANG_RE.match(bang)
    if match is None:
        return None
    return f"{match.group('archive')}.{match.group('subject')}"


def route_query(query: str) -> ArxivDecision:
    """Detect academic / arXiv intent.

    Returns an :class:`ArxivDecision`. Fails open — any exception yields
    ``ArxivDecision(matched=False, query=query, category=None, exclusive=False)``.
    """
    if not isinstance(query, str) or not query.strip():
        return ArxivDecision(False, query, None, False)

    try:
        bang_match = _BANG_RE.match(query)
        if bang_match:
            raw_bang = bang_match.group(1)
            bang_token = raw_bang.lower()
            # 1. Generic arXiv bangs.
            if bang_token in _ARXIV_BANGS:
                cleaned = bang_match.group(2).strip() or query
                log.debug('arxiv-router: bang %r → arxiv; cleaned=%r', bang_token, cleaned)
                return ArxivDecision(True, cleaned, None, True)
            # 2. Category bangs (case-sensitive on the subject half: !cs.LG,
            # not !cs.lg) — match against the unmodified token, not the
            # lowercased one.
            category = _bang_to_category(raw_bang)
            if category:
                cleaned = bang_match.group(2).strip() or query
                log.debug(
                    'arxiv-router: category bang %r → arxiv cat=%s; cleaned=%r',
                    raw_bang,
                    category,
                    cleaned,
                )
                return ArxivDecision(True, cleaned, category, True)

        haystack = query.lower()
        for kw in _ARXIV_KEYWORDS:
            if kw in haystack:
                log.debug('arxiv-router: keyword %r matched in query=%r', kw, query)
                return ArxivDecision(True, query, None, False)
    except Exception as e:  # defensive: never let routing break search
        log.debug('arxiv-router: routing error, falling back: %s', e)

    return ArxivDecision(False, query, None, False)
