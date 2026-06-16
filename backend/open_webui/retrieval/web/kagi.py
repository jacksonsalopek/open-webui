import logging
import socket
import threading
from contextlib import contextmanager
from typing import Optional
from datetime import datetime, timedelta

import requests
import urllib3.util.connection as urllib3_connection
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from open_webui.retrieval.web.main import SearchResult, get_filtered_results
from open_webui.retrieval.web.nl_filter import WebSearchFilter

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10

# Some deployments (e.g. Docker hosts without an IPv6 route) resolve kagi.com to
# an AAAA record they cannot reach, which surfaces as "[Errno 101] Network is
# unreachable". urllib3 picks the address family via a process-global hook, so
# we temporarily pin it to IPv4 for the Kagi request and serialize the swap.
_gai_lock = threading.Lock()


@contextmanager
def _force_ipv4():
    with _gai_lock:
        original_gai_family = urllib3_connection.allowed_gai_family
        urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
        try:
            yield
        finally:
            urllib3_connection.allowed_gai_family = original_gai_family


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({'POST'}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


_session = _build_session()

RECENCY_KEYWORDS = ("latest", "news")
BREAKING_KEYWORDS = (
    "breaking",
    "breaking news",
    "just in",
    "happening now",
    "right now",
    "developing story",
    "last 24 hours",
    "past 24 hours",
    "last 48 hours",
    "past 48 hours",
)
# Kagi's filters.after is date-granular, so a 24-48h "breaking" window maps to
# the last 2 calendar days.
BREAKING_WINDOW_DAYS = 2
RECENCY_WINDOW_DAYS = 30


def search_kagi(
    api_key: str,
    query: str,
    count: int,
    filter_list: Optional[list[str]] = None,
    search_filter: Optional[WebSearchFilter] = None,
) -> list[SearchResult]:
    """Search using Kagi's Search API and return the results as a list of SearchResult objects.

    The Search API will inherit the settings in your account, including results personalization and snippet length.

    Recency handling:
    - If a structured ``search_filter`` is supplied, its native params (date range,
      region, safe-search, domain/keyword lens) are mapped onto the Kagi request
      via ``to_provider_params``.
    - As a deterministic fallback, queries containing "latest"/"news" apply a
      30-day recency window, while "breaking news" style queries (e.g. "breaking",
      "last 24 hours") apply a tighter 24-48h window. Both pull from search and
      news for source variety and skip the domain ``filter_list`` so fresh results
      aren't dropped.

    Args:
        api_key (str): A Kagi Search API key
        query (str): The query to search for
        count (int): The number of results to return
        filter_list (list[str] | None): Domain allow-list
        search_filter (WebSearchFilter | None): Parsed natural-language filter
    """
    query_lower = query.lower()
    is_breaking_query = any(keyword in query_lower for keyword in BREAKING_KEYWORDS)
    is_recency_query = is_breaking_query or any(keyword in query_lower for keyword in RECENCY_KEYWORDS)

    url = 'https://kagi.com/api/v1/search'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    # Kagi rejects extract.count outside [1, 10] (search.extract_count_invalid).
    # The downstream `results[:count]` slice still honors the caller's full
    # requested count if Kagi returns more raw results than the extract cap.
    KAGI_EXTRACT_COUNT_MAX = 10
    extract_count = max(1, min(count, KAGI_EXTRACT_COUNT_MAX))

    payload = {
        'query': query,
        'extract': {'count': extract_count},
    }

    filters = {'safe_search': False}
    if is_breaking_query:
        filters['after'] = (datetime.now() - timedelta(days=BREAKING_WINDOW_DAYS)).strftime('%Y-%m-%d')
    elif is_recency_query:
        filters['after'] = (datetime.now() - timedelta(days=RECENCY_WINDOW_DAYS)).strftime('%Y-%m-%d')

    # Structured filter takes precedence over the keyword heuristic and maps onto
    # Kagi's native params: filters.after/before/region/safe_search, plus an
    # inline lens for domain/keyword include/exclude.
    if search_filter is not None:
        provider_params = search_filter.to_provider_params('kagi')
        filters.update(provider_params.get('filters', {}))
        if 'safe_search' in provider_params:
            filters['safe_search'] = provider_params['safe_search']
        if 'lens' in provider_params:
            payload['lens'] = provider_params['lens']

    payload['filters'] = filters

    log.debug(
        "Kagi search request: url=%s query=%r count=%s is_breaking=%s is_recency=%s "
        "has_filter=%s payload=%s",
        url,
        query,
        count,
        is_breaking_query,
        is_recency_query,
        search_filter is not None,
        payload,
    )

    with _force_ipv4():
        response = _session.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

    log.debug(
        "Kagi search response: status=%s headers=%s body=%s",
        response.status_code,
        dict(response.headers),
        response.text[:2000],
    )

    if not response.ok:
        log.error(
            "Kagi search failed: status=%s url=%s payload=%s response_body=%s",
            response.status_code,
            url,
            payload,
            response.text[:2000],
        )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Re-raise with the response body appended so callers/log-only sites get
        # actionable detail (Kagi returns the validation error in the body).
        body_excerpt = response.text[:500].replace('\n', ' ')
        raise requests.exceptions.HTTPError(
            f"{e} | response_body={body_excerpt}", response=response
        ) from e

    json_response = response.json()
    data = json_response.get('data', {})

    if isinstance(json_response, dict):
        api_errors = json_response.get('error') or json_response.get('errors')
        if api_errors:
            log.warning("Kagi search returned API errors: %s", api_errors)

    # Recency-oriented requests pull from both buckets for a variety of sources.
    wants_variety = is_recency_query or (search_filter is not None and search_filter.after is not None)
    if wants_variety:
        search_results = data.get('search', []) + data.get('news', [])
    else:
        search_results = data.get('search', [])

    results = [
        SearchResult(link=result['url'], title=result['title'], snippet=result.get('snippet'))
        for result in search_results
        if result.get('url') and result.get('title')
    ]

    # Skip the domain filter_list for recency queries so fresh results aren't dropped.
    if filter_list and not is_recency_query:
        results = get_filtered_results(results, filter_list)

    # Kagi's API ignores `extract.count` as a hard cap and returns the full result
    # page (and for recency queries we merge `search` + `news`), so enforce the
    # caller-requested limit here to match every other engine adapter.
    return results[:count]
