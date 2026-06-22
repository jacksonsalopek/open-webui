"""arXiv API adapter for the web-search pipeline.

Hits arXiv's public Atom API (``http://export.arxiv.org/api/query``) and maps
the response into the same :class:`SearchResult` shape every other engine
returns, so the rest of the pipeline (loader, embed, retrieval) is engine-
agnostic. No API key required.

Design notes:

- arXiv asks API clients to keep to roughly one request every 3 seconds per
  IP. We enforce this in-process via a module-level lock so concurrent
  sub-queries under ``asyncio.gather`` serialize cleanly. Recommended by
  https://info.arxiv.org/help/api/user-manual.html#paging .
- The Atom response is parsed with stdlib ``xml.etree.ElementTree`` to keep
  the dependency footprint zero — ``feedparser`` would be nicer but isn't
  worth a transitive dep for one endpoint.
- ``WebSearchFilter`` integration: ``after``/``before`` flip ``sortBy`` to
  ``submittedDate`` and post-filter entries by their published date, since
  arXiv's API has no native date param. ``include_keywords`` are appended to
  the query as additional ``all:`` clauses; ``exclude_keywords``/domains are
  left to the generic post-filter in ``apply_to_results``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Optional
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.nl_filter import WebSearchFilter

log = logging.getLogger(__name__)

ARXIV_ENDPOINT = 'http://export.arxiv.org/api/query'
REQUEST_TIMEOUT = 15

# arXiv's published guidance: keep to ~1 request per 3 seconds. We enforce a
# slightly tighter floor than the 3s in the docs because real-world usage
# alternates with other latency (LLM, embed) so back-to-back hammering is rare.
_MIN_INTERVAL_SECONDS = 3.0
_rate_lock = threading.Lock()
_last_request_ts: float = 0.0

# Atom + arXiv namespaces used in the response payload.
_NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom',
}

# Hard caps from arXiv: ``max_results`` accepts up to 30000 but anything past
# a few dozen rapidly degrades latency. Cap to a sane page size; the caller's
# ``count`` still slices the final list.
_MAX_RESULTS_HARD_CAP = 50


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({'GET'}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    # arXiv API docs ask clients to identify themselves so they can contact
    # heavy users; a stable UA also helps if they ever need to block abuse
    # without taking the whole IP offline.
    session.headers.update({'User-Agent': 'open-webui/arxiv-adapter (+https://openwebui.com)'})
    return session


_session = _build_session()


def _respect_rate_limit() -> None:
    """Block until at least ``_MIN_INTERVAL_SECONDS`` has elapsed since the
    last completed arXiv call. Module-global so concurrent searches serialize.
    """
    global _last_request_ts
    with _rate_lock:
        elapsed = time.monotonic() - _last_request_ts
        if elapsed < _MIN_INTERVAL_SECONDS:
            sleep_for = _MIN_INTERVAL_SECONDS - elapsed
            log.debug('arxiv: rate-limit sleep %.2fs', sleep_for)
            time.sleep(sleep_for)
        _last_request_ts = time.monotonic()


def _build_search_query(
    query: str,
    *,
    category: Optional[str],
    include_keywords: list[str],
) -> str:
    """Compose the ``search_query`` URL param.

    The whole query goes under ``all:`` (title + abstract + author). Extra
    include_keywords get AND-joined; a category constraint is wrapped in
    parens so it scopes the whole expression.
    """
    base = f'all:{query}'
    for kw in include_keywords:
        kw = kw.strip()
        if kw:
            base += f' AND all:{kw}'
    if category:
        base = f'({base}) AND cat:{category}'
    return base


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def _entry_to_result(
    entry: ET.Element,
    *,
    after: Optional[date],
    before: Optional[date],
) -> Optional[SearchResult]:
    """Convert one Atom ``<entry>`` to a SearchResult, or None if filtered out.

    Date filtering is applied here because arXiv's API has no native
    ``filters.after``/``before`` and SearchResult doesn't carry a publish-date
    field for the generic post-filter to act on.
    """
    title_el = entry.find('atom:title', _NS)
    summary_el = entry.find('atom:summary', _NS)
    published_el = entry.find('atom:published', _NS)

    title = (title_el.text or '').strip() if title_el is not None else ''
    summary = (summary_el.text or '').strip() if summary_el is not None else ''
    if not title:
        return None

    published = _parse_date(published_el.text if published_el is not None else None)
    if after and published and published < after:
        return None
    if before and published and published > before:
        return None

    # Prefer the PDF link over the abstract HTML page — the loader can extract
    # PDF text directly and embed it. Fall back to the abstract page link if
    # arXiv ever omits the PDF link from the entry.
    pdf_link: Optional[str] = None
    abs_link: Optional[str] = None
    for link in entry.findall('atom:link', _NS):
        if link.get('type') == 'application/pdf':
            pdf_link = link.get('href')
        elif link.get('rel') == 'alternate':
            abs_link = link.get('href')
    final_link = pdf_link or abs_link
    if not final_link:
        return None

    authors = [
        (a.findtext('atom:name', default='', namespaces=_NS) or '').strip()
        for a in entry.findall('atom:author', _NS)
    ]
    authors = [a for a in authors if a]
    author_str = ', '.join(authors[:5])
    if len(authors) > 5:
        author_str += f' (+{len(authors) - 5} more)'

    primary_cat_el = entry.find('arxiv:primary_category', _NS)
    primary_cat = primary_cat_el.get('term') if primary_cat_el is not None else None

    prefix_bits: list[str] = []
    if author_str:
        prefix_bits.append(author_str)
    if primary_cat:
        prefix_bits.append(f'[{primary_cat}]')
    if published:
        prefix_bits.append(published.isoformat())
    prefix = ' — '.join(prefix_bits)
    snippet = f'{prefix}\n{summary}' if prefix else summary
    # Snippet caps mirror what other engines return — the loader will fetch
    # the PDF and produce the real chunked content downstream.
    snippet = snippet[:1000]

    # Normalize the title (arXiv embeds linebreaks + double-spaces in titles).
    title = ' '.join(title.split())

    return SearchResult(link=final_link, title=title, snippet=snippet)


def search_arxiv(
    query: str,
    count: int,
    search_filter: Optional[WebSearchFilter] = None,
    category: Optional[str] = None,
) -> list[SearchResult]:
    """Search arXiv and return up to ``count`` results.

    Args:
        query: Free-text user query; sent under ``all:``.
        count: Caller-requested result count; final list is trimmed to this.
        search_filter: Parsed NL filter. ``after``/``before`` drive sort order
            and post-filter; ``include_keywords`` are ANDed into the query.
        category: Optional arXiv category constraint (e.g. ``cs.LG``,
            ``math.AP``). Usually supplied by the upstream router.
    """
    after = search_filter.after if search_filter is not None else None
    before = search_filter.before if search_filter is not None else None
    include_keywords = (
        list(search_filter.include_keywords) if search_filter is not None else []
    )

    search_query = _build_search_query(
        query, category=category, include_keywords=include_keywords
    )

    # Recency intent flips us to newest-first. Otherwise let arXiv's relevance
    # ranking pick the order — for non-time-sensitive queries (e.g. "attention
    # is all you need") it surfaces the seminal paper, which recency wouldn't.
    sort_by = 'submittedDate' if (after or before) else 'relevance'

    # Pull a larger page than the caller asked for when we have a date filter
    # so post-filtering still has enough survivors. With no date filter we ask
    # for exactly ``count`` to minimize bandwidth.
    page_size = min(
        _MAX_RESULTS_HARD_CAP,
        count * 3 if (after or before) else max(count, 1),
    )

    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': page_size,
        'sortBy': sort_by,
        'sortOrder': 'descending',
    }

    log.debug('arxiv: GET %s params=%s', ARXIV_ENDPOINT, params)

    _respect_rate_limit()
    response = _session.get(ARXIV_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)

    log.debug(
        'arxiv: response status=%s len=%d',
        response.status_code,
        len(response.content),
    )

    if not response.ok:
        log.error(
            'arxiv: search failed status=%s params=%s body=%s',
            response.status_code,
            params,
            response.text[:500],
        )
        response.raise_for_status()

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        log.error('arxiv: failed to parse Atom response: %s', e)
        return []

    results: list[SearchResult] = []
    for entry in root.findall('atom:entry', _NS):
        result = _entry_to_result(entry, after=after, before=before)
        if result is not None:
            results.append(result)
        if len(results) >= count:
            break

    return results[:count]
