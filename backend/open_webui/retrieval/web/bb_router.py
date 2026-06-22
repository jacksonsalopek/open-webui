"""Bitbucket intent routing for the web-search pipeline.

Decides whether to dispatch a query to the dedicated :mod:`bitbucket`
adapter alongside (or instead of) Kagi. Three trigger modes, mirroring
:mod:`hf_router`:

1. **Bang prefix** -- ``!bb foo``, ``!bitbucket bar``, ``!repo baz``,
   ``!code <something>``. Strips the bang from the query and routes
   exclusively to Bitbucket (no Kagi fanout).

   Special: ``!bb-pr <repo-slug> <query>`` and ``!pr <repo-slug> <query>``
   scope the PR-search leg to that specific repo. Required because
   Bitbucket Cloud has no workspace-wide PR search API; without a slug
   the PR leg is skipped entirely.

2. **Explicit portal keyword** -- ``bitbucket``, ``bb.org``, ``our
   repos``, ``our codebase``, ``internal repo``, ``company codebase``.
   The user clearly references the org's code; query left intact and
   Kagi runs in parallel.

3. **Inline repo-slug detection** -- the query contains a
   ``workspace/reposlug`` pattern matching the configured workspace.
   This both routes to Bitbucket AND scopes the PR-search leg to that
   slug. The slug is stripped from the query before search.

Routing is fail-open: any unexpected error returns ``BbDecision(False, ...)``
and the search falls through to the configured engine.

The configured workspace (passed in by the caller) is used only for the
inline-slug match -- the bang/keyword triggers route to Bitbucket
regardless of whether the user named the workspace.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)


class BbDecision(NamedTuple):
    """Result of Bitbucket intent routing.

    ``exclusive`` mirrors the HF router: True for bang matches (user opted
    in, no fanout), False for keyword / slug matches (fan out to Kagi too
    for broader context).

    ``repo_slug`` is the second half of a ``workspace/repo`` reference
    (extracted from either a ``!bb-pr <repo>`` / ``!pr <repo>`` bang or
    an inline ``workspace/reposlug`` token). When None, the PR-search leg
    is silently skipped — Bitbucket Cloud has no workspace-wide PR API.
    """

    matched: bool
    query: str
    repo_slug: Optional[str]
    exclusive: bool


# Generic bangs route to Bitbucket with all three legs enabled (PR leg only
# fires when a repo_slug is extracted some other way -- usually it won't).
_BB_BANGS: frozenset[str] = frozenset(
    {
        '!bb',
        '!bitbucket',
        '!repo',
        '!repos',
        '!code',
    }
)

# PR-targeted bangs require a repo slug as the next token. Without it the
# bang is treated as a generic Bitbucket route (so a typo doesn't black-
# hole the search).
_BB_PR_BANGS: frozenset[str] = frozenset(
    {
        '!bb-pr',
        '!pr',
        '!pull',
        '!pulls',
    }
)


# Keywords that imply internal-code intent. Kept conservative -- false
# positives here are expensive (every query mentioning "repo" would route
# to Bitbucket otherwise). The phrases here are deliberately
# possessive/internal-tone so they don't fire on neutral references like
# "the React repo on GitHub".
_BB_KEYWORDS: tuple[str, ...] = (
    'bitbucket',
    'bb.org',
    'bitbucket.org',
    'our codebase',
    'our repos',
    'our repo',
    'our code',
    'internal repo',
    'internal codebase',
    'company codebase',
    'company repo',
)


# Bang prefix + remainder. Same pattern as hf_router for consistency.
_BANG_RE = re.compile(r'^\s*(!\S+)\s+(.*)$', re.DOTALL)

# PR bang with explicit repo slug: ``!pr myrepo rest of query``.
# Repo slug = [a-z0-9._-]+ per Bitbucket Cloud slug rules.
_PR_BANG_RE = re.compile(
    r'^\s*(!\S+)\s+([a-z0-9._-]+)\s+(.*)$',
    re.DOTALL | re.IGNORECASE,
)


def _build_slug_re(workspace: str) -> Optional[re.Pattern[str]]:
    """Match ``workspace/reposlug`` as a whole token in the query.

    Returns None when ``workspace`` is empty so callers can short-circuit
    cheaply without a regex match attempt. Word-boundary on both sides
    keeps ``acme/platform`` from matching inside ``acme/platform-docs/v2``
    while still allowing ``acme/foo-bar`` (hyphens are valid in slugs).
    """
    if not workspace:
        return None
    return re.compile(
        rf'(?<![A-Za-z0-9_/-]){re.escape(workspace)}/([a-z0-9._-]+)(?![A-Za-z0-9_/-])',
        re.IGNORECASE,
    )


def route_query(query: str, workspace: Optional[str] = None) -> BbDecision:
    """Detect Bitbucket / internal-codebase intent.

    Returns a :class:`BbDecision`. Fails open -- any exception yields
    ``BbDecision(matched=False, query=query, repo_slug=None, exclusive=False)``.
    """
    if not isinstance(query, str) or not query.strip():
        return BbDecision(False, query, None, False)

    try:
        # Check PR-targeted bangs FIRST (longer pattern, so match before the
        # generic-bang fallthrough can swallow them as a single token).
        pr_match = _PR_BANG_RE.match(query)
        if pr_match:
            bang_token = pr_match.group(1).lower()
            if bang_token in _BB_PR_BANGS:
                repo_slug = pr_match.group(2).strip().lower()
                cleaned = pr_match.group(3).strip() or query
                log.debug(
                    'bb-router: pr-bang %r repo=%r cleaned=%r',
                    bang_token,
                    repo_slug,
                    cleaned,
                )
                return BbDecision(True, cleaned, repo_slug, True)

        bang_match = _BANG_RE.match(query)
        if bang_match:
            bang_token = bang_match.group(1).lower()
            # PR bang without a slug -> treat as generic Bitbucket route.
            # The PR leg will skip itself when repo_slug is None.
            if bang_token in _BB_BANGS or bang_token in _BB_PR_BANGS:
                cleaned = bang_match.group(2).strip() or query
                log.debug(
                    'bb-router: bang %r -> bitbucket; cleaned=%r',
                    bang_token,
                    cleaned,
                )
                return BbDecision(True, cleaned, None, True)

        haystack = query.lower()

        # Inline workspace/slug match takes priority over generic keywords
        # because it carries strictly more info (the slug scopes the PR leg).
        slug_re = _build_slug_re(workspace or '')
        if slug_re is not None:
            slug_match = slug_re.search(query)
            if slug_match:
                repo_slug = slug_match.group(1).lower()
                # Leave the slug in the query: code-search benefits from it
                # as a literal token (matches commit messages, README refs,
                # etc.) and the BBQL filters tolerate it gracefully.
                log.debug(
                    'bb-router: inline slug %s/%s matched in query=%r',
                    workspace,
                    repo_slug,
                    query,
                )
                return BbDecision(True, query, repo_slug, False)

        for kw in _BB_KEYWORDS:
            if kw in haystack:
                log.debug('bb-router: keyword %r matched in query=%r', kw, query)
                return BbDecision(True, query, None, False)
    except Exception as e:
        log.debug('bb-router: routing error, falling back: %s', e)

    return BbDecision(False, query, None, False)
