"""Per-extract LLM compression of web-search docs.

After trafilatura (and the heading-aware trim in ``SafeTrafilaturaLoader``)
returns docs, this module fans out to ``gemma-3-1b`` -- the always-on
sub-second task model routed through LiteLLM to the Spark -- to compress
each doc to an adaptive target length. The model is asked to preserve
code blocks verbatim, drop prose chrome, and keep exact identifiers /
version numbers / CVE ids intact.

Why per-doc compression instead of a single bigger summarizer call?

- Token budget. command-a-plus + cline have a 200K context window, but the
  chat KV cache scales linearly with the input. Pasting 8 URLs * ~4K
  chars/extract = ~32K chars (~10K tokens) on every web-search turn
  burns ~3-4× more context than ~8 compressed snippets at 500 tokens
  each.
- Citation fidelity. We can preserve the source URL / title metadata on
  each compressed doc, so Open WebUI's citation UI still links back to
  the original page. A single bulk summary would collapse all sources
  into one blob and break the citation surface.
- Concurrency. gemma-3-1b runs on the Spark with no llama-swap eviction
  (it's a dedicated always-on service), so we can fan out 6 parallel
  compressions per batch for negligible wall-clock cost (each call is
  sub-second once the model is warm).
- Graceful degradation. If one compress call fails (timeout, 5xx), we
  keep the ORIGINAL doc rather than dropping it. The whole pipeline
  never raises -- the worst-case outcome is that we paid the latency
  but ended up with the same context the chat would have seen anyway.

Configuration (all env-driven, defaults in ``config.py``):

- ``WEB_SEARCH_COMPRESS_ENABLED`` (default True)
- ``WEB_SEARCH_COMPRESS_MODEL``   (default ``gemma-3-1b``)
- ``WEB_SEARCH_COMPRESS_BASE_URL`` (default ``http://litellm:4000/v1``)
- ``WEB_SEARCH_COMPRESS_CONCURRENCY`` (default 6)
- ``WEB_SEARCH_COMPRESS_TIMEOUT_SECONDS`` (default 30)

Adaptive target lengths -- the prompt asks the model to land near these
based on input shape:

- Short page (< 2000 chars): ~300 tokens (~1200 chars)
- Long / code-heavy (> 5000 chars OR > 5 code blocks OR contains source
  markers like ``function`` / ``class`` / ``def`` / ``fn`` / ``import`` /
  ``#include``): ~800 tokens (~3200 chars)
- Otherwise: ~500 tokens (~2000 chars)

The heuristic deliberately favors keeping more bytes on code-heavy docs
because that's where chat models lose accuracy under aggressive
summarization (an aggressive compress pass tends to paraphrase variable
names and API signatures, defeating the entire point of citing the
source).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import Request
from langchain_core.documents import Document

log = logging.getLogger(__name__)


# Default knobs. ``getattr(request.app.state.config, NAME, default)``
# at the call site honors ConfigVar overrides; these are the floor.
_DEFAULT_MODEL = 'gemma-3-1b'
_DEFAULT_BASE_URL = 'http://litellm:4000/v1'
_DEFAULT_CONCURRENCY = 6
_DEFAULT_TIMEOUT_SECONDS = 30

# Code-bearing markers used to bias the target length upward.
# Order-insensitive substring match (lowercased) -- the goal is to
# detect "this doc has source code or API definitions in it" with as
# few false positives as possible. Markdown fenced blocks (``` blocks)
# are detected separately because they're the clearest signal.
_CODE_MARKERS = (
    'function ',
    'class ',
    'def ',
    'fn ',
    'import ',
    '#include',
)

# Short pages: ~300 tokens (~1200 chars). Medium: ~500 tokens (~2000
# chars). Long / code-heavy: ~800 tokens (~3200 chars). The chars-to-
# tokens conversion uses the rough 4:1 ratio that holds for English
# prose + code at qwen / gemma tokenizers.
_TARGET_SHORT_WORDS = 220   # ~300 tokens
_TARGET_MEDIUM_WORDS = 380  # ~500 tokens
_TARGET_LONG_WORDS = 600    # ~800 tokens

_SHORT_INPUT_THRESHOLD = 2000
_LONG_INPUT_THRESHOLD = 5000
_CODE_HEAVY_FENCE_THRESHOLD = 5


def _env(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def _pick_target_words(content: str) -> int:
    """Return the target compressed length in words for a given input doc.

    See module docstring for the bands. Word count is a stable proxy for
    token count across English prose + code; the model is told to aim
    near the target in the system prompt and we don't enforce a hard
    cap (the model + provider's max_tokens handle that).
    """
    length = len(content)
    fence_count = content.count('```')
    has_code_markers = any(m in content.lower() for m in _CODE_MARKERS)

    code_heavy = (
        # Fenced-block count: each fence open + close = 2, so ``> N``
        # fences means more than N/2 fenced blocks.
        fence_count >= 2 * _CODE_HEAVY_FENCE_THRESHOLD
        or has_code_markers
    )

    if length > _LONG_INPUT_THRESHOLD or code_heavy:
        return _TARGET_LONG_WORDS
    if length < _SHORT_INPUT_THRESHOLD:
        return _TARGET_SHORT_WORDS
    return _TARGET_MEDIUM_WORDS


# Static system prompt. Variable bits (target word count, query) get
# substituted via ``str.format`` -- keep the curly braces in literal
# markdown by doubling them.
_SYSTEM_PROMPT_TEMPLATE = (
    'You are a faithful technical summarizer. Compress the user-provided '
    'web page extract to roughly {target_words} words, preserving the '
    'parts that answer the user\'s original query.\n\n'
    'HARD RULES:\n'
    '- Preserve every fenced code block (```...```) VERBATIM. Do not '
    'paraphrase, reformat, or abbreviate code.\n'
    '- Preserve every function/class/method signature, identifier, '
    'version number, CVE id, command-line flag, file path, URL, and '
    'API endpoint EXACTLY as written.\n'
    '- Drop prose chrome: "In this article we will explore...", '
    '"Stay tuned for...", marketing copy, repeated headings, "Click '
    'here to learn more", cookie / GDPR notices that survived '
    'extraction.\n'
    '- Drop sections that are clearly off-topic relative to the query.\n'
    '- Do NOT invent new facts. If something is missing from the source, '
    'leave it missing -- do not fill in plausible-sounding details.\n'
    '- Output Markdown. Use the same heading structure as the source '
    'when it helps, but feel free to flatten headings the user does not '
    'need.\n'
    '- Do not include any meta-commentary like "Here is the compressed '
    'version" or "Summary:" -- just emit the compressed content.\n\n'
    'USER QUERY (for relevance, do NOT answer it -- just keep what is '
    'relevant to it):\n{query}'
)


def _build_messages(doc: Document, query: str, target_words: int) -> List[Dict[str, str]]:
    """Build the chat-completions messages array for one compress call."""
    title = doc.metadata.get('title') or ''
    source = doc.metadata.get('source') or ''
    user_payload_parts: List[str] = []
    if title:
        user_payload_parts.append(f'Title: {title}')
    if source:
        user_payload_parts.append(f'Source: {source}')
    user_payload_parts.append('')
    user_payload_parts.append(doc.page_content)
    user_payload = '\n'.join(user_payload_parts)
    return [
        {
            'role': 'system',
            'content': _SYSTEM_PROMPT_TEMPLATE.format(
                target_words=target_words,
                query=query or '(no specific query provided; preserve full informational content)',
            ),
        },
        {'role': 'user', 'content': user_payload},
    ]


def _forward_traceparent(request: Request) -> Dict[str, str]:
    """Return the W3C trace headers from ``request`` ready to splat into aiohttp.

    Open WebUI's OTel auto-instrumentation will inject these headers on
    its own outbound aiohttp calls when a span is active, but doing it
    manually too is cheap and keeps the trace chain intact even if the
    auto-instrumentation is disabled or the call escapes its context
    (e.g. when ``asyncio.gather`` schedules workers on a different
    event-loop task). We forward both ``traceparent`` and ``tracestate``
    -- the latter carries vendor-specific span context.
    """
    headers: Dict[str, str] = {}
    try:
        tp = request.headers.get('traceparent') if request is not None else None
        ts = request.headers.get('tracestate') if request is not None else None
        if tp:
            headers['traceparent'] = tp
        if ts:
            headers['tracestate'] = ts
    except Exception:  # pragma: no cover -- request may be a mock in tests
        return headers
    return headers


async def _compress_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    doc: Document,
    query: str,
    *,
    model: str,
    base_url: str,
    timeout: int,
    trace_headers: Dict[str, str],
) -> Document:
    """Compress a single doc; return the compressed doc, or the original on failure.

    The original doc's metadata is preserved verbatim on the returned
    doc (source/title/sitename etc.), so the citation surface is
    invariant under compression. We tag the compressed doc with
    ``metadata['compressed'] = True`` so downstream consumers can tell
    them apart from un-compressed ones (e.g. cache hits that bypassed
    the trafilatura pipeline entirely).
    """
    target_words = _pick_target_words(doc.page_content)
    messages = _build_messages(doc, query, target_words)

    # ``max_tokens`` floor leaves room for the model to overshoot the
    # word target slightly without truncation; floor at 256 prevents
    # the model from emitting an empty body when the target is short.
    max_tokens = max(256, int(target_words * 1.6))

    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0.2,
        'max_tokens': max_tokens,
        'stream': False,
    }

    url = base_url.rstrip('/') + '/chat/completions'

    async with semaphore:
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers=trace_headers or None,
            ) as response:
                response.raise_for_status()
                data: Any = await response.json()
        except Exception as e:
            log.warning(
                'llm_compress: keeping original doc for %s due to error: %s',
                doc.metadata.get('source', '<no-source>'),
                e,
            )
            return doc

    try:
        compressed = data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as e:
        log.warning(
            'llm_compress: bad response shape for %s, keeping original: %s',
            doc.metadata.get('source', '<no-source>'),
            e,
        )
        return doc

    if not compressed or not compressed.strip():
        log.warning(
            'llm_compress: empty completion for %s, keeping original',
            doc.metadata.get('source', '<no-source>'),
        )
        return doc

    new_metadata = dict(doc.metadata)
    new_metadata['compressed'] = True
    new_metadata['compressed_target_words'] = target_words
    return Document(page_content=compressed, metadata=new_metadata)


async def compress_docs(
    docs: List[Document],
    query: str,
    *,
    request: Optional[Request] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    concurrency: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> List[Document]:
    """Fan out per-doc compress calls; return docs in the same input order.

    NEVER raises -- on any pipeline-level failure (no docs, all calls
    timing out, etc.) the original docs are returned. The chat surface
    stays correct even when the Spark is briefly unreachable; we just
    pay a small upper-bound wait (``timeout_seconds``) before returning
    the un-compressed fallback.
    """
    if not docs:
        return docs

    cfg = getattr(request, 'app', None)
    cfg = getattr(cfg, 'state', None) if cfg is not None else None
    cfg = getattr(cfg, 'config', None) if cfg is not None else None

    def _cfg(name: str, fallback: Any) -> Any:
        if cfg is None:
            return fallback
        return getattr(cfg, name, fallback)

    resolved_model = model or _cfg('WEB_SEARCH_COMPRESS_MODEL', _env('WEB_SEARCH_COMPRESS_MODEL', _DEFAULT_MODEL))
    resolved_base_url = base_url or _cfg('WEB_SEARCH_COMPRESS_BASE_URL', _env('WEB_SEARCH_COMPRESS_BASE_URL', _DEFAULT_BASE_URL))
    resolved_concurrency = int(
        concurrency
        or _cfg('WEB_SEARCH_COMPRESS_CONCURRENCY', _env_int('WEB_SEARCH_COMPRESS_CONCURRENCY', _DEFAULT_CONCURRENCY))
    )
    resolved_timeout = int(
        timeout_seconds
        or _cfg(
            'WEB_SEARCH_COMPRESS_TIMEOUT_SECONDS',
            _env_int('WEB_SEARCH_COMPRESS_TIMEOUT_SECONDS', _DEFAULT_TIMEOUT_SECONDS),
        )
    )

    semaphore = asyncio.Semaphore(max(1, resolved_concurrency))
    trace_headers = _forward_traceparent(request) if request is not None else {}

    # One aiohttp session for the whole batch -- connection reuse across
    # the 6+ concurrent calls keeps Spark-side conn churn low.
    timeout_total = aiohttp.ClientTimeout(total=resolved_timeout * max(1, len(docs)))
    try:
        async with aiohttp.ClientSession(timeout=timeout_total) as session:
            tasks = [
                _compress_one(
                    session,
                    semaphore,
                    doc,
                    query,
                    model=resolved_model,
                    base_url=resolved_base_url,
                    timeout=resolved_timeout,
                    trace_headers=trace_headers,
                )
                for doc in docs
            ]
            # gather without return_exceptions -- _compress_one is
            # designed to never raise. If something escapes anyway, we
            # don't want the whole compress pass to crash the search
            # turn; the outer try/except is the safety net.
            results = await asyncio.gather(*tasks)
    except Exception as e:
        log.warning(
            'llm_compress: pipeline failed (%s); returning original docs',
            e,
        )
        return docs

    compressed_count = sum(1 for d in results if d.metadata.get('compressed'))
    if compressed_count:
        log.info(
            'llm_compress: %d/%d docs compressed via %s',
            compressed_count,
            len(results),
            resolved_model,
        )
    return results


__all__ = ['compress_docs']
