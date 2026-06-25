"""Doc-portal routing for the web-search pipeline.

Decides whether to redirect a query from the default Kagi engine to a
specific documentation portal — currently MDN (web platform), Microsoft
Learn (Windows / .NET / Azure / etc.), or Godot Engine (game-dev docs
hosted on Read the Docs). Two trigger modes, mirroring :mod:`arxiv_router`
and :mod:`kagi_lenses`:

1. **Bang prefix** — ``!mdn fetch api`` / ``!winui navigationview`` /
   ``!godot signals``. The bang is stripped before dispatch; routing fires
   unconditionally on match.
2. **Keyword scan** — fires only when the user *explicitly names* the
   portal (``mdn``, ``developer.mozilla``, ``learn.microsoft``, ``msdn``,
   ``godotengine``, ``gdscript``, ``docs.godotengine``). Topic-only queries
   (e.g. plain "winui xaml") stay on Kagi where
   :func:`nl_filter._detect_developer_topic_domains` already routes them
   to the right doc domain via Kagi's ``sites_included``.

This intentional split — bangs / portal-name keywords route to the
portal's *own* API, generic doc-intent keywords stay on Kagi — means users
who explicitly invoke a portal get authoritative-only results, while users
who just ask topic questions still get Kagi's broader mix of docs + blog
posts + Stack Overflow.

A few high-signal product bangs (``!winui``, ``!dotnet``, ``!wpf``,
``!winapi``, ``!azure``) opt into Microsoft Learn with the corresponding
``product`` scope for tighter results. The Godot bangs ``!godot3`` /
``!godot4`` / ``!godotlatest`` pin the docs branch (3.6 / stable / master).

Routing is fail-open: any unexpected error returns "no override" and the
search falls through to the configured engine (Kagi).
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)


class DocsDecision(NamedTuple):
    """Result of doc-portal intent routing.

    ``exclusive`` distinguishes the two trigger modes:

    - **Bang match** → ``exclusive=True``. The user explicitly named the
      portal (``!mdn``, ``!winui``, ``!dotnet``, ``!godot``), so the
      dispatcher should skip the Kagi fanout and surface authoritative
      results only.
    - **Keyword match** → ``exclusive=False``. The user named the portal
      in prose (``mdn``, ``learn.microsoft``, ``msdn``, ``gdscript``); the
      dispatcher can fan out to Kagi in parallel for broader coverage
      (Stack Overflow, blog posts, etc.).
    - **No match** → ``engine=None``, ``exclusive`` is irrelevant.

    ``product`` and ``version`` are engine-specific narrowing knobs and
    only one is meaningful at a time:

    - ``product`` — Microsoft Learn ``products`` slug (e.g. ``dotnet``,
      ``windows``, ``azure``). Set by product bangs like ``!winui`` /
      ``!dotnet``. Always ``None`` for non-mslearn engines.
    - ``version`` — Godot docs branch slug (``stable`` / ``latest`` /
      ``3.6``). Set by ``!godot3`` / ``!godotlatest`` etc. Always ``None``
      for non-godot engines.
    """

    engine: Optional[str]  # 'mdn', 'mslearn', 'godot', or None
    query: str
    product: Optional[str]  # MS Learn product slug, if a product bang fired
    exclusive: bool
    version: Optional[str] = None  # Godot docs branch, if a version bang fired


# Bangs that route to MDN. Web-platform shorthand only; avoid
# topic-overloaded bangs like ``!js`` (could be Node-server vs. browser-JS).
_MDN_BANGS: frozenset[str] = frozenset(
    {
        '!mdn',
        '!webdocs',
        '!webdev',
        '!webplatform',
    }
)

# Bangs that route to Microsoft Learn without a product scope.
_MSLEARN_BANGS: frozenset[str] = frozenset(
    {
        '!mslearn',
        '!msdocs',
        '!msdn',
        '!learn',
    }
)

# Bangs that route to the Godot Engine docs (Read the Docs project
# ``godot``). ``!godot3`` / ``!godot4`` / ``!godotlatest`` additionally
# pin the docs branch via :data:`_GODOT_VERSION_BANGS`.
_GODOT_BANGS: frozenset[str] = frozenset(
    {
        '!godot',
        '!gd',
        '!gdscript',
        '!godotengine',
        '!godotdocs',
    }
)

# Godot bangs that additionally pin a docs branch. Values match the Read
# the Docs version slugs the Godot project actually builds (``stable`` ≈
# the current 4.x release line; ``latest`` ≈ the master/dev branch; ``3.6``
# is the legacy 3.x line). The adapter normalizes a few aliases on top of
# these (``master`` → ``latest``, ``4.x`` → ``stable``, etc.) so the bang
# table can stay tight.
_GODOT_VERSION_BANGS: dict[str, str] = {
    '!godot3': '3.6',
    '!godot4': 'stable',
    '!godotstable': 'stable',
    '!godotlatest': 'latest',
    '!godotmaster': 'latest',
    '!godotdev': 'latest',
}

# Bangs that route to Microsoft Learn WITH a product scope. The mapping
# value is the Learn ``products`` slug (verified against
# ``learn.microsoft.com/api/search`` facet output).
_MSLEARN_PRODUCT_BANGS: dict[str, str] = {
    '!winui': 'windows',
    '!winapp': 'windows',
    '!winsdk': 'windows',
    '!winappsdk': 'windows',
    '!winapi': 'windows',
    '!win32': 'windows',
    '!wpf': 'dotnet',
    '!wf': 'dotnet',
    '!winforms': 'dotnet',
    '!dotnet': 'dotnet',
    '!netcore': 'dotnet',
    '!azure': 'azure',
    '!entra': 'entra',
    '!powershell': 'powershell',
    '!aspnet': 'aspnet-core',
    '!efcore': 'ef-core',
}

# Keyword triggers — only fire when the user explicitly *names* the portal.
# Topic-only queries stay on Kagi (see module docstring).
_MDN_KEYWORDS: tuple[str, ...] = (
    'mdn',
    'developer.mozilla',
    'mozilla developer',
)
_MSLEARN_KEYWORDS: tuple[str, ...] = (
    'learn.microsoft',
    'msdn',
    'microsoft learn',
    'microsoft docs',
)
# Godot portal keywords. Kept narrow and engine-specific — "godot" alone is
# a common reference (Beckett's play, Godot Tequila, etc.), but the fanout
# layer still merges with Kagi when ``exclusive=False`` so a false-positive
# match just adds one cheap leg instead of replacing Kagi entirely. The
# strict variants (``gdscript``, ``godotengine``, ``godot engine``,
# ``docs.godotengine``) are unambiguous primary-source pins.
_GODOT_KEYWORDS: tuple[str, ...] = (
    'godotengine',
    'godot engine',
    'docs.godotengine',
    'gdscript',
)


_BANG_RE = re.compile(r'^\s*(!\S+)\s+(.*)$', re.DOTALL)


def route_query(query: str) -> DocsDecision:
    """Detect doc-portal intent.

    Returns a :class:`DocsDecision`. Fails open — any exception yields
    ``DocsDecision(engine=None, query=query, product=None, exclusive=False)``.
    """
    if not isinstance(query, str) or not query.strip():
        return DocsDecision(None, query, None, False)

    try:
        bang_match = _BANG_RE.match(query)
        if bang_match:
            bang_token = bang_match.group(1).lower()
            cleaned = bang_match.group(2).strip() or query

            if bang_token in _MDN_BANGS:
                log.debug('docs-router: bang %r → mdn; cleaned=%r', bang_token, cleaned)
                return DocsDecision('mdn', cleaned, None, True)
            if bang_token in _MSLEARN_BANGS:
                log.debug(
                    'docs-router: bang %r → mslearn; cleaned=%r', bang_token, cleaned
                )
                return DocsDecision('mslearn', cleaned, None, True)
            product = _MSLEARN_PRODUCT_BANGS.get(bang_token)
            if product is not None:
                log.debug(
                    'docs-router: product bang %r → mslearn (product=%s); cleaned=%r',
                    bang_token,
                    product,
                    cleaned,
                )
                return DocsDecision('mslearn', cleaned, product, True)
            if bang_token in _GODOT_BANGS:
                log.debug(
                    'docs-router: bang %r → godot; cleaned=%r', bang_token, cleaned
                )
                return DocsDecision('godot', cleaned, None, True, None)
            godot_version = _GODOT_VERSION_BANGS.get(bang_token)
            if godot_version is not None:
                log.debug(
                    'docs-router: version bang %r → godot (version=%s); cleaned=%r',
                    bang_token,
                    godot_version,
                    cleaned,
                )
                return DocsDecision('godot', cleaned, None, True, godot_version)

        haystack = query.lower()
        # Keyword scan — first match wins. MDN keyword pool is checked
        # before Learn because the only overlap risk ("microsoft mdn"...)
        # doesn't appear in real queries, and MDN's pool is narrower.
        # Godot keywords come last because they're the most specialized
        # corpus; a generic doc query that happens to mention "godot"
        # alongside MDN/Learn should still resolve to the web/MS pool.
        for kw in _MDN_KEYWORDS:
            if kw in haystack:
                log.debug('docs-router: keyword %r → mdn; query=%r', kw, query)
                return DocsDecision('mdn', query, None, False)
        for kw in _MSLEARN_KEYWORDS:
            if kw in haystack:
                log.debug('docs-router: keyword %r → mslearn; query=%r', kw, query)
                return DocsDecision('mslearn', query, None, False)
        for kw in _GODOT_KEYWORDS:
            if kw in haystack:
                log.debug('docs-router: keyword %r → godot; query=%r', kw, query)
                return DocsDecision('godot', query, None, False, None)
    except Exception as e:  # defensive: never let routing break search
        log.debug('docs-router: routing error, falling back: %s', e)

    return DocsDecision(None, query, None, False)
