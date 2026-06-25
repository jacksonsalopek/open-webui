"""Godot Engine docs adapter for the web-search pipeline.

Godot's documentation is hosted on Read the Docs (project slug ``godot``).
We hit the Read the Docs Server-Side Search API v3 at
``https://app.readthedocs.org/api/v3/search/`` with the documented fielded
syntax (``project:godot/<version>``) and map the response into the same
:class:`SearchResult` shape every other engine returns. No API key required.

Response shape (Dec 2025):

.. code-block:: json

    {
      "count": 173,
      "results": [
        {"type": "page",
         "project": {"slug": "godot"},
         "version": {"slug": "stable"},
         "title": "Signal",
         "domain": "https://docs.godotengine.org",
         "path": "/en/stable/classes/class_signal.html",
         "highlights": {"title": ["<span>Signal</span>"]},
         "blocks": [{"type": "section",
                     "id": "signal",
                     "title": "Signal",
                     "content": "A built-in type ...",
                     "highlights": {"content": ["A built-in type ..."],
                                    "title": ["<span>Signal</span>"]}},
                    ...]},
        ...
      ]
    }

We surface one :class:`SearchResult` per page. ``link`` = ``domain + path``.
``snippet`` is built from the highest-quality block:

1. The first block whose ``highlights.content`` is non-empty (so the snippet
   shows the matched span, not just the lede).
2. Failing that, the first block's ``content``.
3. Failing that, the page-level title highlight or just the page title.

References:
- API docs:  https://docs.readthedocs.com/platform/stable/server-side-search/api.html
- Syntax:    https://docs.readthedocs.com/platform/stable/server-side-search/syntax.html
"""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Iterable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.nl_filter import WebSearchFilter

log = logging.getLogger(__name__)

GODOT_ENDPOINT = 'https://app.readthedocs.org/api/v3/search/'
GODOT_PROJECT_SLUG = 'godot'
REQUEST_TIMEOUT = 10

# Default Godot docs branch. ``stable`` tracks the current release series
# (4.x at time of writing); ``latest`` is the development docs (master). The
# legacy 3.x line lives on ``3.6``. Users opt into non-stable via
# :mod:`docs_router` bangs (``!godot4``, ``!godot3``, ``!godotlatest``).
_DEFAULT_VERSION = 'stable'

# Versions Read the Docs actively builds for the Godot project. Anything not
# in this set falls back to ``stable`` — RtD returns an empty result list for
# unknown versions rather than transparently falling back, which would look
# like an outage from the user's POV.
_SUPPORTED_VERSIONS: frozenset[str] = frozenset({'stable', 'latest', '3.6'})

# RtD highlight fragments are wrapped in ``<span>...</span>`` to mark the
# matched terms. We strip the tags but keep the inner text. Span attributes
# are never set on these fragments so a plain literal regex is enough; the
# HTML-escaped content is unescaped after stripping so e.g. ``&#x27;`` →
# ``'`` in the rendered snippet.
_SPAN_RE = re.compile(r'</?span(?:\s[^>]*)?>', re.IGNORECASE)


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({'GET'}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update(
        {'User-Agent': 'open-webui/godot-adapter (+https://openwebui.com)'}
    )
    return session


_session = _build_session()


def _resolve_version(version: Optional[str]) -> str:
    """Pin the Godot version slug to a value Read the Docs builds for.

    Accepts the bang-router's free-form ``version`` and falls back to
    ``stable`` for anything we don't recognize. Normalizes a few common
    aliases (``master`` → ``latest``, ``4`` / ``4.x`` → ``stable``,
    ``3`` / ``3.x`` → ``3.6``) so the router can stay loose without
    poisoning the upstream call with an unknown slug.
    """
    if not version:
        return _DEFAULT_VERSION
    code = version.strip().lower()
    if not code:
        return _DEFAULT_VERSION
    aliases = {
        'master': 'latest',
        'main': 'latest',
        'dev': 'latest',
        'development': 'latest',
        '4': 'stable',
        '4.x': 'stable',
        '4.0': 'stable',
        '3': '3.6',
        '3.x': '3.6',
        '3.5': '3.6',
    }
    code = aliases.get(code, code)
    if code in _SUPPORTED_VERSIONS:
        return code
    return _DEFAULT_VERSION


def _strip_highlight(text: str) -> str:
    """Drop ``<span>`` markup and decode HTML entities in a snippet fragment."""
    if not text:
        return ''
    return unescape(_SPAN_RE.sub('', text)).strip()


def _first_content_highlight(blocks: Iterable[dict]) -> Optional[str]:
    """Return the first block's content highlight, joined and cleaned.

    Each block's ``highlights.content`` is a list of fragments containing
    the matched term wrapped in ``<span>``. We join with ``" ... "`` (the
    same elision marker Kagi uses) so the snippet reads as a coherent
    excerpt even when RtD returns multiple non-contiguous spans.
    """
    for block in blocks:
        if not isinstance(block, dict):
            continue
        highlights = block.get('highlights') or {}
        fragments = highlights.get('content') or []
        cleaned = [_strip_highlight(f) for f in fragments if isinstance(f, str)]
        cleaned = [f for f in cleaned if f]
        if cleaned:
            return ' ... '.join(cleaned)
    return None


def _first_block_content(blocks: Iterable[dict]) -> Optional[str]:
    """Fallback snippet: the raw ``content`` of the first usable block."""
    for block in blocks:
        if not isinstance(block, dict):
            continue
        content = (block.get('content') or '').strip()
        if content:
            return content
    return None


def search_godot(
    query: str,
    count: int,
    search_filter: Optional[WebSearchFilter] = None,
    version: Optional[str] = None,
) -> list[SearchResult]:
    """Search the Godot Engine docs and return up to ``count`` results.

    Args:
        query: Free-text user query. Sent verbatim after the
            ``project:godot/<version>`` fielded prefix so the RtD index
            only returns hits inside the Godot docs project.
        count: Caller-requested result count; final list trimmed to this.
        search_filter: Optional NL filter. RtD search has no native
            language scope (the Godot project is en-only on RtD), so the
            filter falls through to the generic post-filter in
            ``apply_to_results``.
        version: Optional Godot docs version slug (e.g. ``stable``,
            ``latest``, ``3.6``). Usually supplied by the upstream router
            from a bang like ``!godot3``. Falls back to ``stable``.
    """
    pinned = _resolve_version(version)
    fielded = f'project:{GODOT_PROJECT_SLUG}/{pinned} {query.strip()}'.strip()

    params: dict[str, str | int] = {
        'q': fielded,
        # RtD's default page_size is 50 — request only what we need to keep
        # the response small. Cap at 50 to match the upstream limit.
        'page_size': min(max(count, 1), 50),
    }
    log.debug('godot: GET %s params=%s', GODOT_ENDPOINT, params)

    response = _session.get(GODOT_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)

    log.debug(
        'godot: response status=%s len=%d',
        response.status_code,
        len(response.content),
    )

    if not response.ok:
        log.error(
            'godot: search failed status=%s params=%s body=%s',
            response.status_code,
            params,
            response.text[:500],
        )
        response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as e:
        log.error('godot: failed to parse JSON response: %s', e)
        return []

    raw_results = payload.get('results') or []
    if not isinstance(raw_results, list):
        log.warning('godot: unexpected `results` shape: %r', type(raw_results))
        return []

    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = (item.get('title') or '').strip()
        domain = (item.get('domain') or '').rstrip('/')
        path = item.get('path') or ''
        if not title or not domain or not path:
            continue
        if not path.startswith('/'):
            path = '/' + path
        link = f'{domain}{path}'

        blocks = item.get('blocks') or []
        if not isinstance(blocks, list):
            blocks = []

        snippet: Optional[str] = (
            _first_content_highlight(blocks)
            or _first_block_content(blocks)
            or _strip_highlight(
                (item.get('highlights') or {}).get('title', [''])[0]
                if isinstance((item.get('highlights') or {}).get('title'), list)
                else ''
            )
            or None
        )
        if snippet:
            # 800-char cap matches MDN/MS Learn; Godot tutorial pages have
            # very long first blocks (whole tutorial sections inlined) so
            # without a cap a single result could swallow the LLM context.
            snippet = snippet[:800]

        results.append(SearchResult(link=link, title=title, snippet=snippet))
        if len(results) >= count:
            break

    return results[:count]
