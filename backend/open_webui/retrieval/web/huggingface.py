"""Hugging Face Hub API adapter for the web-search pipeline.

Hits the public ``https://huggingface.co/api/models`` endpoint and maps each
returned model into a :class:`SearchResult`. No API key required for public
models.

Response shape (Dec 2025):

.. code-block:: json

    [
      {"_id": "...", "id": "google/gemma-2-9b", "modelId": "google/gemma-2-9b",
       "lastModified": "2025-12-03T10:11:22.000Z",
       "createdAt": "2024-06-27T12:34:56.000Z",
       "pipeline_tag": "text-generation",
       "library_name": "transformers",
       "downloads": 12345, "likes": 678,
       "tags": ["transformers", "safetensors", "gemma2", "text-generation",
                "license:gemma", "en"],
       "private": false},
      ...
    ]

Sort policy: decided upstream by :mod:`hf_router` (which inspects the
*original* query for recency cues like ``latest`` / ``newest`` / ``recent``)
and forwarded here as the ``sort`` arg. Defaults to ``downloads`` -- the
strongest signal for *canonical* releases (Google's gemma-3 vs random
user-uploaded variants).

Author filter: when the upstream :mod:`hf_router` matched a known
open-weights family (``gemma`` -> ``google``, ``llama`` -> ``meta-llama``,
etc.), it forwards the canonical HF org as ``author``. That keeps the
results pinned to the model line the user actually meant -- and serves as
a safety net for false-positive auto-routes (``granite countertops``,
``best llama for trekking``) by collapsing them to ~zero HF hits while
Kagi handles the real answer via fanout.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.nl_filter import WebSearchFilter

log = logging.getLogger(__name__)

HF_ENDPOINT = 'https://huggingface.co/api/models'
HF_ORIGIN = 'https://huggingface.co'
REQUEST_TIMEOUT = 10

_LIMIT_HARD_CAP = 50

_VALID_SORTS: frozenset[str] = frozenset(
    {'downloads', 'likes', 'lastModified', 'createdAt'}
)
_DEFAULT_SORT = 'downloads'


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
        {'User-Agent': 'open-webui/huggingface-adapter (+https://openwebui.com)'}
    )
    return session


_session = _build_session()


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def _format_count(n: int) -> str:
    """Compact human-readable count: 1234 -> "1.2K", 1234567 -> "1.2M"."""
    if not isinstance(n, (int, float)):
        return str(n)
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.1f}K'
    return str(n)


_TAG_NOISE_PREFIXES: tuple[str, ...] = (
    'license:',
    'arxiv:',
    'doi:',
    'region:',
)
_TAG_NOISE_EXACT: frozenset[str] = frozenset(
    {
        'safetensors',
        'pytorch',
        'tf',
        'jax',
        'onnx',
        'gguf',
        'autotrain_compatible',
        'endpoints_compatible',
        'text-generation-inference',
        'has_space',
        'conversational',
    }
)


def _select_signal_tags(tags: list[str], *, limit: int = 6) -> list[str]:
    """Filter HF tags to the high-signal subset for snippet display.

    HF surfaces ~20-30 tags per model, most of which are infrastructure
    (``safetensors``, ``pytorch``, ``endpoints_compatible``) or noisy
    metadata (``license:apache-2.0``, ``arxiv:2401.04088``). The useful
    ones for a search snippet are language codes, model family names, and
    domain tags (``code``, ``reasoning``, ``vision``, etc.).
    """
    out: list[str] = []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        if tag in _TAG_NOISE_EXACT:
            continue
        if any(tag.startswith(p) for p in _TAG_NOISE_PREFIXES):
            continue
        out.append(tag)
        if len(out) >= limit:
            break
    return out


def search_huggingface(
    query: str,
    count: int,
    search_filter: Optional[WebSearchFilter] = None,
    author: Optional[str] = None,
    sort: Optional[str] = None,
) -> list[SearchResult]:
    """Search Hugging Face Hub and return up to ``count`` model results.

    Args:
        query: Free-text search; sent as the ``search`` param to HF.
        count: Caller-requested result count; final list is trimmed.
        search_filter: Optional NL filter. ``after``/``before`` are applied
            as a local post-filter against each model's ``lastModified``;
            HF's API has no native date param.
        author: Optional HF org/user slug (e.g. ``google``, ``meta-llama``,
            ``mistralai``). When supplied, scopes the search to that
            author's models -- used by the upstream router when a known
            open-weights family triggered the route.
        sort: Optional sort key (``downloads``, ``likes``, ``lastModified``,
            ``createdAt``). Decided upstream by :mod:`hf_router` from the
            *original* user query. Defaults to ``downloads``.
    """
    after = search_filter.after if search_filter is not None else None
    before = search_filter.before if search_filter is not None else None

    sort_by = sort if sort in _VALID_SORTS else _DEFAULT_SORT

    params: dict[str, str | int] = {
        'search': query,
        'sort': sort_by,
        'direction': '-1',
        'limit': min(
            _LIMIT_HARD_CAP, max(count * 3 if (after or before) else count, 1)
        ),
    }
    if author:
        params['author'] = author

    log.debug('hf: GET %s params=%s', HF_ENDPOINT, params)

    response = _session.get(HF_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
    log.debug(
        'hf: response status=%s len=%d', response.status_code, len(response.content)
    )

    if not response.ok:
        log.error(
            'hf: search failed status=%s params=%s body=%s',
            response.status_code,
            params,
            response.text[:500],
        )
        response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as e:
        log.error('hf: failed to parse JSON response: %s', e)
        return []

    if not isinstance(payload, list):
        log.warning('hf: unexpected response shape: %r', type(payload))
        return []

    results: list[SearchResult] = []
    for model in payload:
        if not isinstance(model, dict):
            continue
        model_id = (model.get('id') or model.get('modelId') or '').strip()
        if not model_id:
            continue
        if model.get('private') is True:
            continue

        last_modified = _parse_iso_date(model.get('lastModified'))
        if after and last_modified and last_modified < after:
            continue
        if before and last_modified and last_modified > before:
            continue

        pipeline = (model.get('pipeline_tag') or '').strip()
        library = (model.get('library_name') or '').strip()
        downloads = model.get('downloads') or 0
        likes = model.get('likes') or 0

        meta_parts: list[str] = []
        if pipeline:
            meta_parts.append(f'[{pipeline}]')
        if library:
            meta_parts.append(library)
        if last_modified:
            meta_parts.append(f'updated {last_modified.isoformat()}')
        if downloads:
            meta_parts.append(f'{_format_count(int(downloads))} downloads')
        if likes:
            meta_parts.append(f'{_format_count(int(likes))} likes')

        tags = _select_signal_tags(model.get('tags', []), limit=6)
        tag_line = ', '.join(tags) if tags else ''

        meta_line = ' · '.join(meta_parts)
        snippet_parts = [p for p in (meta_line, tag_line) if p]
        snippet = '\n'.join(snippet_parts)[:800] if snippet_parts else None

        link = f'{HF_ORIGIN}/{model_id}'
        results.append(SearchResult(link=link, title=model_id, snippet=snippet))
        if len(results) >= count:
            break

    return results[:count]
