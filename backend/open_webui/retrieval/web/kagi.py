import logging
from typing import Optional
from datetime import datetime, timedelta

import requests
from open_webui.retrieval.web.main import SearchResult, get_filtered_results

log = logging.getLogger(__name__)

RECENCY_KEYWORDS = ("latest", "news")


def search_kagi(api_key: str, query: str, count: int, filter_list: Optional[list[str]] = None) -> list[SearchResult]:
    """Search using Kagi's Search API and return the results as a list of SearchResult objects.

    The Search API will inherit the settings in your account, including results personalization and snippet length.

    A recency filter is only applied when the query is asking for recent information (i.e. it
    contains the word "latest" or "news"). In that case a variety of sources are returned and the
    filter_list is intentionally ignored so that fresh results are not removed.

    Args:
        api_key (str): A Kagi Search API key
        query (str): The query to search for
        count (int): The number of results to return
    """
    is_recency_query = any(keyword in query.lower() for keyword in RECENCY_KEYWORDS)

    url = 'https://kagi.com/api/v1/search'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    filters = {'safe_search': False}
    if is_recency_query:
        filters['after'] = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    payload = {
        'query': query,
        'extract': {'count': count},
        'filters': filters,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    json_response = response.json()
    data = json_response.get('data', {})

    if is_recency_query:
        # Pull from both search and news to ensure a variety of sources.
        search_results = data.get('search', []) + data.get('news', [])
    else:
        search_results = data.get('search', [])

    results = [
        SearchResult(link=result['url'], title=result['title'], snippet=result.get('snippet'))
        for result in search_results
        if result.get('url') and result.get('title')
    ]

    # Skip the filter_list for recency queries so fresh results aren't dropped.
    if filter_list and not is_recency_query:
        results = get_filtered_results(results, filter_list)

    return results
