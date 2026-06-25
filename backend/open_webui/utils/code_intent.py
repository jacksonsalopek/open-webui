"""Stage 0 intent classifier — cheap routing for code-shaped queries.

See ``docs/CODE_SAFETY_PIPELINE.md#stage-0--intent-classifier-planned-prerequisite-for-energy-wins``
for the full routing table and the rationale. The summary:

A non-trivial fraction of "coding ability" turns don't need the full
retrieve + embed + rerank + generate funnel. "Find symbol ``foo``" is a
tag lookup; "where is ``foo`` used" is a ripgrep over the Stage 1 symbol
index; "explain this region" is a single embed retrieve. Spending a full
RAG pass on each of those is exactly the inefficiency the safety pipeline
is meant to eliminate. This module is the routing primitive that lets
upstream code dispatch on intent before paying the embedding tax.

Taxonomy
--------

Seven labels; callers pattern-match on a :data:`CodeIntent` literal:

- ``find_symbol`` -- "where is ``X`` defined" / "find the
  ``UserService`` class". Route to the Stage 1 symbol index.
- ``where_used`` -- "where is ``foo`` called" / "all usages of ``X``".
  Route to a ripgrep over the symbol index.
- ``explain_region`` -- "what does this function do" / "explain
  ``UserService.login``". Route to single-chunk retrieval.
- ``generate_code`` -- "write me a function that ...". Route to the
  generation path; Stages 1-3 run on the output.
- ``generate_and_run`` -- "write and run a function that ...". Same as
  ``generate_code`` plus Stage 5 (sandboxed execution).
- ``refactor`` -- "rename ``foo`` to ``bar``" / "extract this into a
  method". Stages 1-3 on input AND output.
- ``unknown`` -- the model returned something we couldn't parse, or
  every guard (timeout, network error, empty input) tripped. Caller
  should fall through to its default routing.

Implementation notes
--------------------

- Uses the small task model (``TASK_MODEL``, currently routed to
  ``gemma-3-1b`` via the Spark backend per ``docs/FEATURES.md``).
  ~50 ms per classification on warm cache.
- LiteLLM is reached via the existing OpenAI-compatible chat-completions
  HTTP endpoint -- same shape as :mod:`open_webui.retrieval.web.nl_filter`,
  same fail-open contract.
- Strictly fail-open: any timeout, transport error, or unparseable
  response returns ``'unknown'`` rather than raising. The classifier is
  a router, not a gate -- a missed classification only forfeits the
  energy win, it doesn't break the request.
- Empty / whitespace input short-circuits to ``'unknown'`` without
  hitting the network at all.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal, Optional, get_args

log = logging.getLogger(__name__)


CodeIntent = Literal[
    'find_symbol',
    'where_used',
    'explain_region',
    'generate_code',
    'generate_and_run',
    'refactor',
    'unknown',
]


# Cached at module load so the per-call dispatch is a frozenset lookup
# rather than re-introspecting the Literal each time.
_VALID_LABELS: frozenset[str] = frozenset(get_args(CodeIntent))


_SYSTEM_PROMPT = """You classify a user's coding-related query into exactly one routing label.

Output ONLY one of these literal strings, with no quotes, no punctuation, no commentary, no explanation:

find_symbol
where_used
explain_region
generate_code
generate_and_run
refactor
unknown

Definitions:
- find_symbol: user wants to locate where a named symbol (function, class, type, constant) is defined.
- where_used: user wants every call site / usage of a named symbol.
- explain_region: user wants a description of an existing piece of code (a function, a region, a file).
- generate_code: user wants new code written. They will read/copy it; they did not ask to execute it.
- generate_and_run: user wants new code written AND executed (mentions "run", "execute", asks for the result).
- refactor: user wants existing code rewritten in place (rename, extract, restructure, modernize).
- unknown: query is not about code, or doesn't fit any of the above.

Respond with the label only.
"""


_DECORATION_CHARS = ' \t\r\n`"\'.,;:!?()[]{}'


def _normalize_label(raw: str) -> CodeIntent:
    """Coerce a model output string into a known label or ``'unknown'``.

    Models occasionally wrap the label in quotes, backticks, trailing
    punctuation, leading whitespace, etc. ("``find_symbol``",
    "find_symbol.", "  refactor  ", '"refactor".'). We strip a fixed
    set of decoration characters from both ends in a single ``strip``
    pass -- multi-char trailing decoration would otherwise require an
    iterative loop.
    """
    if not raw:
        return 'unknown'
    cleaned = raw.strip(_DECORATION_CHARS).lower()
    if cleaned in _VALID_LABELS:
        return cleaned  # type: ignore[return-value]
    return 'unknown'


async def _call_litellm(
    *,
    query: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> Optional[str]:
    """Single OpenAI-compatible chat-completions request, returns raw content.

    Returns ``None`` on any failure. Kept as an awaitable so the caller
    can wrap the whole thing in :func:`asyncio.wait_for`; the underlying
    HTTP call uses ``requests`` (sync) inside ``asyncio.to_thread`` so we
    don't pull in an extra HTTP dependency just for this module -- this
    matches the pattern in :mod:`open_webui.retrieval.web.nl_filter`.
    """
    import requests  # local import: keeps module import cheap

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': query},
        ],
        'temperature': 0,
        # Tiny budget -- the label is at most 16 characters, anything
        # the model emits beyond that is decoration we'll strip anyway.
        'max_tokens': 16,
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    url = f'{base_url.rstrip("/")}/chat/completions'

    def _do_request() -> Optional[str]:
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:  # noqa: BLE001 -- fail-open by contract
            log.debug('code_intent: classifier call failed: %s', e)
            return None

    return await asyncio.to_thread(_do_request)


async def classify_code_intent(
    query: str,
    *,
    model: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> CodeIntent:
    """Classify a query into one of the seven :data:`CodeIntent` labels.

    Returns ``'unknown'`` on any failure (empty input, missing
    classifier endpoint, timeout, network error, unparseable label).
    Never raises -- callers can wire this into hot paths without
    needing a defensive try/except around it.

    Parameters
    ----------
    query:
        The natural-language coding query to classify.
    model:
        Override the model name. Defaults to the
        ``CODE_INTENT_CLASSIFIER_MODEL`` env var, then ``TASK_MODEL``,
        then a hard fallback string so the classifier still works on
        bootstrap configurations that haven't set ``TASK_MODEL`` yet.
    timeout_seconds:
        Wall-clock budget for the entire call (HTTP timeout + parsing).
        Defaults to 5 s; the classifier is a fast-path router, so going
        much over a few seconds defeats the purpose of having it.
    """
    if not query or not query.strip():
        return 'unknown'

    chosen_model = (
        model
        or os.getenv('CODE_INTENT_CLASSIFIER_MODEL')
        or os.getenv('TASK_MODEL')
        or 'gemma-3-1b'
    )
    base_url = (
        os.getenv('OPENAI_API_BASE_URL')
        or os.getenv('OPENAI_API_BASE_URLS', '').split(';')[0]
    )
    api_key = os.getenv('OPENAI_API_KEY', 'sk-anything')

    if not base_url:
        log.debug('code_intent: no OPENAI_API_BASE_URL configured; returning unknown')
        return 'unknown'

    try:
        raw = await asyncio.wait_for(
            _call_litellm(
                query=query.strip(),
                model=chosen_model,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        log.debug('code_intent: classifier timed out after %ss', timeout_seconds)
        return 'unknown'
    except Exception as e:  # noqa: BLE001 -- fail-open by contract
        log.debug('code_intent: classifier raised: %s', e)
        return 'unknown'

    if raw is None:
        return 'unknown'

    return _normalize_label(raw)


__all__ = ['CodeIntent', 'classify_code_intent']
