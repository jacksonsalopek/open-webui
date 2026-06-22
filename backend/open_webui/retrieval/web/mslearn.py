"""Microsoft Learn (formerly MSDN) search API adapter.

Hits the public Microsoft Learn search endpoint at
``https://learn.microsoft.com/api/search`` and maps the response into the
same :class:`SearchResult` shape every other engine returns. No API key
required.

Response shape (Dec 2025):

.. code-block:: json

    {
      "results": [
        {"title": "WinUI 3 - Windows apps",
         "url": "https://learn.microsoft.com/en-us/windows/apps/winui/winui3/",
         "displayUrl": {...},
         "description": "WinUI 3 ... In this article ...",
         "lastUpdatedDate": "2025-11-12",
         "breadcrumbs": [...],
         "category": "..."},
        ...
      ],
      "count": 1234,
      "nextLink": "...",
      ...
    }

Notable scopes for narrowing:

- ``products`` — comma-separated product IDs (e.g. ``dotnet``, ``windows``,
  ``azure``). We forward an optional ``product`` arg as a single-entry list.
- ``locale`` — BCP-47 lower-cased (``en-us``, ``ja-jp``, ``de-de``).
- ``$top`` — page size, capped at 50 by the upstream service.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.nl_filter import WebSearchFilter

log = logging.getLogger(__name__)

MSLEARN_ENDPOINT = 'https://learn.microsoft.com/api/search'
REQUEST_TIMEOUT = 10

_DEFAULT_LOCALE = 'en-us'
_TOP_HARD_CAP = 50

# Boilerplate strings the Learn frontend injects into the description text
# (driven by their "Summarize this article" CTA + reading-aid blurbs). Strip
# them so embedded summaries don't leak into the result snippet and confuse
# the downstream model.
_DESCRIPTION_NOISE = (
    re.compile(r'\bSummarize this article for me\b', re.IGNORECASE),
    re.compile(r'\bIn this article\b', re.IGNORECASE),
)


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
        {'User-Agent': 'open-webui/mslearn-adapter (+https://openwebui.com)'}
    )
    return session


_session = _build_session()


def _clean_description(text: str) -> str:
    """Strip UI boilerplate and collapse whitespace.

    Learn's API returns the visible page text including widget labels like
    "Summarize this article for me" / "In this article", which would
    otherwise show up at the top of every snippet.
    """
    if not text:
        return ''
    cleaned = text
    for pattern in _DESCRIPTION_NOISE:
        cleaned = pattern.sub('', cleaned)
    # Collapse the multi-space gaps left by the substitutions above.
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _resolve_locale(language: Optional[str]) -> str:
    """Map a ``WebSearchFilter.language`` (ISO 639-1) to a Learn locale tag.

    Learn locales follow BCP-47 lower-cased (``en-us``, ``fr-fr``,
    ``ja-jp``). For bare language codes we expand the common ones; anything
    else falls back to en-us since Learn returns an empty result list for
    unknown locales rather than transparently falling back.
    """
    if not language:
        return _DEFAULT_LOCALE
    code = language.strip().lower()
    if not code:
        return _DEFAULT_LOCALE
    # Common bare-language → Learn-locale expansions. Keep narrow — Learn
    # supports dozens but most are partial machine translations and the
    # user almost certainly wants the canonical en-us docs.
    expansions = {
        'en': 'en-us',
        'fr': 'fr-fr',
        'de': 'de-de',
        'ja': 'ja-jp',
        'ko': 'ko-kr',
        'es': 'es-es',
        'pt': 'pt-br',
        'zh': 'zh-cn',
        'ru': 'ru-ru',
        'it': 'it-it',
    }
    if code in expansions:
        return expansions[code]
    # Already a BCP-47 tag like "en-us"
    if '-' in code:
        return code
    return _DEFAULT_LOCALE


def search_mslearn(
    query: str,
    count: int,
    search_filter: Optional[WebSearchFilter] = None,
    product: Optional[str] = None,
) -> list[SearchResult]:
    """Search Microsoft Learn and return up to ``count`` results.

    Args:
        query: Free-text user query; sent as the ``search`` param.
        count: Caller-requested result count; final list trimmed to this.
        search_filter: Optional NL filter. ``language`` maps to Learn's
            ``locale``; other dimensions fall through to the generic
            post-filter.
        product: Optional Learn product slug (e.g. ``dotnet``, ``windows``,
            ``azure``) for narrowing. Usually supplied by the upstream
            router from a bang like ``!dotnet``.
    """
    locale = _resolve_locale(
        search_filter.language if search_filter is not None else None
    )

    params: dict[str, str | int] = {
        'search': query,
        'locale': locale,
        '$top': min(max(count, 1), _TOP_HARD_CAP),
        'expandScope': 'true',
    }
    if product:
        # The Learn search API expects a comma-separated list under
        # ``products``. We only forward a single bang-derived value, but
        # leave the comma-join shape so multi-product scoping is trivial
        # to add later.
        params['products'] = product

    log.debug('mslearn: GET %s params=%s', MSLEARN_ENDPOINT, params)

    response = _session.get(MSLEARN_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)

    log.debug(
        'mslearn: response status=%s len=%d',
        response.status_code,
        len(response.content),
    )

    if not response.ok:
        log.error(
            'mslearn: search failed status=%s params=%s body=%s',
            response.status_code,
            params,
            response.text[:500],
        )
        response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as e:
        log.error('mslearn: failed to parse JSON response: %s', e)
        return []

    raw_results = payload.get('results') or []
    if not isinstance(raw_results, list):
        log.warning('mslearn: unexpected `results` shape: %r', type(raw_results))
        return []

    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = (item.get('title') or '').strip()
        link = (item.get('url') or '').strip()
        if not title or not link:
            continue
        description = _clean_description(item.get('description') or '')
        last_updated = (item.get('lastUpdatedDate') or '').strip()
        prefix = f'(updated {last_updated[:10]}) ' if last_updated else ''
        snippet = (prefix + description)[:800] if description else None
        results.append(SearchResult(link=link, title=title, snippet=snippet))
        if len(results) >= count:
            break

    return results[:count]
