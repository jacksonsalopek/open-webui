"""Natural-language web-search result filtering.

This module turns a free-form natural-language instruction (typically the user's
search query) into a canonical, provider-agnostic :class:`WebSearchFilter`. That
container can then be:

1. Translated into a given provider's *native* filtering query params via
   :meth:`WebSearchFilter.to_provider_params` (preferred — filtering happens at
   the source), and/or
2. Applied to the returned results via :meth:`WebSearchFilter.apply_to_results`
   for providers that lack native support for a given dimension (e.g. domain or
   keyword filtering).

The parser calls an OpenAI-compatible chat endpoint (e.g. the local litellm
gateway). It is deliberately *fail-open*: if the feature is disabled, the model
is unreachable, or the response can't be parsed, an empty filter is returned and
search behaves exactly as it did before.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Callable, Optional
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


class WebSearchFilter(BaseModel):
    """Provider-agnostic representation of web-search result filters."""

    after: Optional[date] = None  # keep results published on/after this date
    before: Optional[date] = None  # keep results published on/before this date
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    safe_search: Optional[bool] = None
    language: Optional[str] = None  # ISO 639-1 code, e.g. "en"
    region: Optional[str] = None  # ISO 3166-1 alpha-2 country code, e.g. "US"

    def is_empty(self) -> bool:
        return not any(
            [
                self.after,
                self.before,
                self.include_domains,
                self.exclude_domains,
                self.include_keywords,
                self.exclude_keywords,
                self.safe_search is not None,
                self.language,
                self.region,
            ]
        )

    def to_provider_params(self, provider: str) -> dict:
        """Translate this filter into ``provider``'s native query params.

        Returns an empty dict for providers without a registered translator
        (filtering for those falls back to :meth:`apply_to_results`).
        """
        translator = _PROVIDER_TRANSLATORS.get(provider)
        if translator is None:
            return {}
        return translator(self)

    def apply_to_results(self, results: list) -> list:
        """Best-effort post-hoc filtering of returned ``SearchResult`` objects.

        Only dimensions that can be evaluated from the result metadata are
        applied here: include/exclude domains and include/exclude keywords.
        Date and language constraints depend on data we don't reliably have on
        the result and are expected to be handled by the provider natively.
        """
        if self.is_empty():
            return results

        include_domains = [d.lower().lstrip('.') for d in self.include_domains]
        exclude_domains = [d.lower().lstrip('.') for d in self.exclude_domains]
        include_keywords = [k.lower() for k in self.include_keywords]
        exclude_keywords = [k.lower() for k in self.exclude_keywords]

        filtered = []
        for result in results:
            link = getattr(result, 'link', None) or ''
            domain = urlparse(link).netloc.lower()
            haystack = ' '.join(
                str(part or '')
                for part in (getattr(result, 'title', ''), getattr(result, 'snippet', ''))
            ).lower()

            if include_domains and not any(domain == d or domain.endswith('.' + d) for d in include_domains):
                continue
            if exclude_domains and any(domain == d or domain.endswith('.' + d) for d in exclude_domains):
                continue
            if include_keywords and not any(k in haystack for k in include_keywords):
                continue
            if exclude_keywords and any(k in haystack for k in exclude_keywords):
                continue

            filtered.append(result)

        return filtered


# ── Provider translators ────────────────────────────────────────────────────
# Map a canonical filter into a provider's native query params. Each translator
# returns a dict whose shape matches what the corresponding ``search_*`` helper
# expects to receive (see the provider module / its call site).

def _to_kagi(f: WebSearchFilter) -> dict:
    """Map onto Kagi's native params.

    See https://kagi.com/api/docs/openapi/search/search. Dates go in ``filters``,
    ``safe_search`` is top-level, and domain/keyword constraints use an inline
    ``lens``. Kagi covers every dimension we model, so no post-filtering is needed.
    """
    params: dict = {}

    filters: dict = {}
    if f.after:
        filters['after'] = f.after.isoformat()
    if f.before:
        filters['before'] = f.before.isoformat()
    if f.region:
        filters['region'] = f.region.upper()
    if filters:
        params['filters'] = filters

    if f.safe_search is not None:
        params['safe_search'] = f.safe_search

    lens: dict = {}
    if f.include_domains:
        lens['sites_included'] = f.include_domains
    if f.exclude_domains:
        lens['sites_excluded'] = f.exclude_domains
    if f.include_keywords:
        lens['keywords_included'] = f.include_keywords
    if f.exclude_keywords:
        lens['keywords_excluded'] = f.exclude_keywords
    if lens:
        params['lens'] = lens

    return params


_PROVIDER_TRANSLATORS: dict[str, Callable[[WebSearchFilter], dict]] = {
    'kagi': _to_kagi,
}

# Providers whose translator covers every filter dimension we model; the
# generic post-filter (``apply_to_results``) is skipped for these to avoid
# dropping valid results the provider already vetted (e.g. a keyword match in
# page content that isn't present in the returned snippet).
NATIVE_FULL_SUPPORT: frozenset[str] = frozenset({'kagi'})


def has_full_native_support(provider: str) -> bool:
    return provider in NATIVE_FULL_SUPPORT


# ── Natural-language parsing ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """You extract structured web-search filters from a user's search query.

Return ONLY a JSON object with these optional fields (omit a field or use null/[] when the query does not clearly ask for it):
- "after": ISO date (YYYY-MM-DD) lower bound for recency. Use this when the query asks for recent/latest/current information or "news".
- "before": ISO date (YYYY-MM-DD) upper bound.
- "include_domains": array of bare domains the user wants results restricted to (e.g. ["arxiv.org"]).
- "exclude_domains": array of bare domains to exclude.
- "include_keywords": array of terms results should mention.
- "exclude_keywords": array of terms results should NOT mention.
- "language": ISO 639-1 code if the user explicitly requests a language.
- "region": ISO 3166-1 alpha-2 country code (e.g. "US", "GB") if the user wants results localized to a specific country/region.

Rules:
- Do NOT invent filters. If the query is a plain informational search with no filtering intent, return {}.
- Be conservative: only populate a field when the query clearly implies it.
- Today's date is {today}. For "recent"/"latest"/"news" style queries, set "after" to roughly 30 days before today.
- For "breaking news" style queries (e.g. "breaking", "happening now", "last 24/48 hours", "developing story", or anything implying the past day or two), set "after" to 1-2 days before today instead of 30.
"""


def parse_nl_filter(
    instruction: str,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> WebSearchFilter:
    """Parse a natural-language ``instruction`` into a :class:`WebSearchFilter`.

    Fails open: returns an empty filter on any error.
    """
    instruction = (instruction or '').strip()
    if not instruction:
        return WebSearchFilter()

    base_url = base_url or os.getenv('OPENAI_API_BASE_URL') or os.getenv('OPENAI_API_BASE_URLS', '').split(';')[0]
    api_key = api_key or os.getenv('OPENAI_API_KEY', 'sk-anything')
    model = model or os.getenv('WEB_SEARCH_NL_FILTER_MODEL', 'granite4.1:8b')

    if not base_url:
        log.debug('nl_filter: no OPENAI_API_BASE_URL configured; skipping')
        return WebSearchFilter()

    today = date.today().isoformat()
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': _SYSTEM_PROMPT.format(today=today)},
            {'role': 'user', 'content': instruction},
        ],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
    }

    try:
        response = requests.post(
            f'{base_url.rstrip("/")}/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        data = json.loads(content)
    except Exception as e:
        log.debug(f'nl_filter: parse failed, returning empty filter ({e})')
        return WebSearchFilter()

    if not isinstance(data, dict):
        return WebSearchFilter()

    # Only keep keys we recognize so unexpected model output can't break validation.
    allowed = set(WebSearchFilter.model_fields.keys())
    data = {k: v for k, v in data.items() if k in allowed and v not in (None, '')}

    try:
        return WebSearchFilter(**data)
    except ValidationError as e:
        log.debug(f'nl_filter: validation failed, returning empty filter ({e})')
        return WebSearchFilter()


def extract_filter_from_query(query: str) -> WebSearchFilter:
    """Convenience wrapper used by the search pipeline.

    Honors the ``ENABLE_WEB_SEARCH_NL_FILTER`` env flag (default on) and always
    fails open so search keeps working if parsing is unavailable.
    """
    if not _env_flag('ENABLE_WEB_SEARCH_NL_FILTER', True):
        return WebSearchFilter()
    try:
        return parse_nl_filter(query)
    except Exception as e:  # defensive: never let filtering break search
        log.debug(f'nl_filter: extraction failed ({e})')
        return WebSearchFilter()
