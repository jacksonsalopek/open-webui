"""MDN Web Docs API adapter for the web-search pipeline.

Hits the public MDN search endpoint at
``https://developer.mozilla.org/api/v1/search`` and maps the response into
the same :class:`SearchResult` shape every other engine returns. No API key
required.

The endpoint is undocumented but stable — it's the same JSON the live MDN
site search consumes (Mozilla's Yari frontend). Response shape (Dec 2025):

.. code-block:: json

    {
      "documents": [
        {"mdn_url": "/en-US/docs/Web/API/Fetch_API",
         "title": "Fetch API",
         "summary": "...",
         "score": 218.4,
         "popularity": 0.14},
        ...
      ],
      "metadata": {...},
      "suggestions": [...]
    }

The `mdn_url` is a server-relative path; we prefix the canonical origin so
the downstream loader gets a fetchable URL.
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.nl_filter import WebSearchFilter

log = logging.getLogger(__name__)

MDN_ENDPOINT = 'https://developer.mozilla.org/api/v1/search'
MDN_ORIGIN = 'https://developer.mozilla.org'
REQUEST_TIMEOUT = 10

# MDN's search supports `locale` (e.g. ``en-US``, ``fr``, ``de``). We default
# to en-US — the docs are mostly translated from English and the en-US
# corpus is the most complete, especially for new web platform features.
_DEFAULT_LOCALE = 'en-US'

# Locales MDN actively translates. Anything else falls back to en-US since
# MDN returns an empty document list for unsupported locales rather than
# transparently falling back. Source: developer.mozilla.org footer.
_SUPPORTED_LOCALES: frozenset[str] = frozenset(
    {'en-US', 'de', 'es', 'fr', 'ja', 'ko', 'pt-BR', 'ru', 'zh-CN', 'zh-TW'}
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
        {'User-Agent': 'open-webui/mdn-adapter (+https://openwebui.com)'}
    )
    return session


_session = _build_session()


def _resolve_locale(language: Optional[str]) -> str:
    """Map a ``WebSearchFilter.language`` (ISO 639-1) to an MDN locale tag.

    MDN locales mostly follow BCP-47 (``en-US``, ``pt-BR``). For bare
    language codes we keep the simple ones (``de``, ``fr``, ``ja``, ``ko``,
    ``ru``) and upgrade ``en`` to ``en-US``. Unsupported codes fall back to
    the default so we don't quietly return zero results.
    """
    if not language:
        return _DEFAULT_LOCALE
    code = language.strip()
    if not code:
        return _DEFAULT_LOCALE
    if code.lower() == 'en':
        return 'en-US'
    if code in _SUPPORTED_LOCALES:
        return code
    # case-insensitive retry
    for supported in _SUPPORTED_LOCALES:
        if supported.lower() == code.lower():
            return supported
    return _DEFAULT_LOCALE


def search_mdn(
    query: str,
    count: int,
    search_filter: Optional[WebSearchFilter] = None,
) -> list[SearchResult]:
    """Search MDN and return up to ``count`` results.

    Args:
        query: Free-text user query; passed through verbatim as the ``q`` param.
        count: Caller-requested result count; final list trimmed to this.
        search_filter: Optional NL filter. Only ``language`` is honored
            natively (mapped to MDN's ``locale``). Other dimensions
            (``include_keywords``, etc.) fall through to the generic
            post-filter in ``apply_to_results``.
    """
    locale = _resolve_locale(
        search_filter.language if search_filter is not None else None
    )

    params = {'q': query, 'locale': locale}
    log.debug('mdn: GET %s params=%s', MDN_ENDPOINT, params)

    response = _session.get(MDN_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)

    log.debug(
        'mdn: response status=%s len=%d', response.status_code, len(response.content)
    )

    if not response.ok:
        log.error(
            'mdn: search failed status=%s params=%s body=%s',
            response.status_code,
            params,
            response.text[:500],
        )
        response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as e:
        log.error('mdn: failed to parse JSON response: %s', e)
        return []

    documents = payload.get('documents') or []
    if not isinstance(documents, list):
        log.warning('mdn: unexpected `documents` shape: %r', type(documents))
        return []

    results: list[SearchResult] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        title = (doc.get('title') or '').strip()
        relative = (doc.get('mdn_url') or '').strip()
        if not title or not relative:
            continue
        link = urljoin(MDN_ORIGIN, relative)
        summary = (doc.get('summary') or '').strip()
        # MDN summaries are typically 1-2 sentences and already well-formed;
        # cap at 800 chars to be consistent with the other adapters without
        # truncating useful context mid-paragraph.
        snippet = summary[:800] if summary else None
        results.append(SearchResult(link=link, title=title, snippet=snippet))
        if len(results) >= count:
            break

    return results[:count]
