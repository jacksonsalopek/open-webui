"""Kagi lens auto-routing for the web-search pipeline.

Maps user query intent → a Kagi ``lens_id`` so that search calls automatically
scope to the right curated set of domains (a "lens" in Kagi parlance). Two
trigger modes, in priority order:

1. **Bang prefix** — ``!reddit best mechanical keyboard``. The bang is stripped
   from the query before it goes to Kagi; the matching lens is applied.
2. **Keyword scan** — ``best mechanical keyboard reddit``. Substring matches
   against any configured ``keywords`` route to that lens. The query is *not*
   modified, since the keyword may be intentional.

Lens IDs come from one of two sources:

- A built-in lens identifier exposed by Kagi (e.g. their stable "Programming",
  "Forums", or "News" lenses).
- A shareable user lens — visible at https://kagi.com/settings/lenses once you
  flip "Shareable" on the lens you want to expose. The ID is the URL slug
  (``https://kagi.com/lenses/<ID>``).

Configuration lives in a YAML file pointed at by the
``KAGI_LENSES_CONFIG_PATH`` env var (default
``/app/backend/data/kagi_lenses.yaml``). Schema:

.. code-block:: yaml

    lenses:
      - id: "ABC123XYZ"           # required: Kagi lens_id
        name: "Reddit"            # optional: display name for logs
        bangs: ["!reddit", "!r"]  # optional: prefix bangs (case-insensitive)
        keywords: ["on reddit", "subreddit"]  # optional: substring triggers
        priority: 10              # optional: higher wins on keyword ties

Routing is fail-open: malformed config, missing file, or a YAML parse error
all degrade to "no lens routed" and search behaves exactly as it did before.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import yaml

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = '/app/backend/data/kagi_lenses.yaml'


@dataclass(frozen=True)
class LensRule:
    """A single configured lens with its routing triggers."""

    id: str  # the lens_id sent to Kagi
    name: str  # display name for logging
    bangs: tuple[str, ...]  # already lowercased, leading '!' enforced
    keywords: tuple[str, ...]  # already lowercased
    priority: int = 0


@dataclass
class _Cache:
    path: Optional[str] = None
    mtime: float = 0.0
    rules: tuple[LensRule, ...] = field(default_factory=tuple)


_cache = _Cache()
_cache_lock = Lock()


def _resolve_path(explicit: Optional[str]) -> str:
    return (
        explicit
        or os.getenv('KAGI_LENSES_CONFIG_PATH')
        or DEFAULT_CONFIG_PATH
    )


def _normalise_bang(bang: str) -> Optional[str]:
    bang = (bang or '').strip().lower()
    if not bang:
        return None
    if not bang.startswith('!'):
        bang = '!' + bang
    # Bangs must be a single token; otherwise the prefix-matching logic
    # would be ambiguous when the user types e.g. "!my lens query".
    if any(c.isspace() for c in bang):
        log.warning('kagi-lenses: dropping bang with whitespace: %r', bang)
        return None
    return bang


def _build_rule(raw: dict, idx: int) -> Optional[LensRule]:
    if not isinstance(raw, dict):
        log.warning('kagi-lenses: entry #%d is not a mapping, skipping', idx)
        return None

    lens_id = (raw.get('id') or '').strip()
    if not lens_id:
        log.warning('kagi-lenses: entry #%d missing required `id`, skipping', idx)
        return None

    name = (raw.get('name') or lens_id).strip()

    bangs_raw = raw.get('bangs') or []
    if not isinstance(bangs_raw, list):
        bangs_raw = []
    bangs: list[str] = []
    for b in bangs_raw:
        if isinstance(b, str):
            normalised = _normalise_bang(b)
            if normalised is not None:
                bangs.append(normalised)

    keywords_raw = raw.get('keywords') or []
    if not isinstance(keywords_raw, list):
        keywords_raw = []
    keywords = tuple(
        k.lower().strip()
        for k in keywords_raw
        if isinstance(k, str) and k.strip()
    )

    try:
        priority = int(raw.get('priority') or 0)
    except (TypeError, ValueError):
        priority = 0

    if not bangs and not keywords:
        log.warning(
            'kagi-lenses: entry %r (id=%s) has no bangs or keywords, skipping',
            name,
            lens_id,
        )
        return None

    return LensRule(
        id=lens_id,
        name=name,
        bangs=tuple(bangs),
        keywords=keywords,
        priority=priority,
    )


def _load_rules(path: str) -> tuple[LensRule, ...]:
    """Read ``path`` and return the parsed rule list, or () on any failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError:
        log.debug('kagi-lenses: config file not found at %s; routing disabled', path)
        return ()
    except (OSError, yaml.YAMLError) as e:
        log.error('kagi-lenses: failed to read %s: %s', path, e)
        return ()

    if not isinstance(doc, dict):
        log.warning('kagi-lenses: %s does not contain a mapping at the top level', path)
        return ()

    raw_lenses = doc.get('lenses') or []
    if not isinstance(raw_lenses, list):
        log.warning('kagi-lenses: %s `lenses` is not a list', path)
        return ()

    rules: list[LensRule] = []
    for idx, entry in enumerate(raw_lenses):
        rule = _build_rule(entry, idx)
        if rule is not None:
            rules.append(rule)

    log.info('kagi-lenses: loaded %d rule(s) from %s', len(rules), path)
    return tuple(rules)


def _refresh_cache(explicit_path: Optional[str]) -> tuple[LensRule, ...]:
    """Return the active rule set, reloading from disk if mtime advanced.

    Lets users edit the YAML and pick up changes without a container
    restart, while keeping the hot path a single ``stat`` syscall when
    nothing has changed.
    """
    path = _resolve_path(explicit_path)

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    with _cache_lock:
        if _cache.path == path and _cache.mtime == mtime and _cache.rules:
            return _cache.rules

        rules = _load_rules(path) if mtime else ()
        _cache.path = path
        _cache.mtime = mtime
        _cache.rules = rules
        return rules


_BANG_RE = re.compile(r'^\s*(!\S+)\s+(.*)$', re.DOTALL)


def route_query(
    query: str,
    *,
    config_path: Optional[str] = None,
) -> tuple[Optional[str], str, Optional[str]]:
    """Route ``query`` through configured lens rules.

    Returns ``(lens_id, cleaned_query, lens_name)``.

    - If a bang prefix matched, ``cleaned_query`` has the bang stripped.
    - If only a keyword matched, ``cleaned_query`` is the original query.
    - If nothing matched, returns ``(None, query, None)``.

    Always fails open: any unexpected error returns ``(None, query, None)``.
    """
    if not isinstance(query, str) or not query.strip():
        return None, query, None

    try:
        rules = _refresh_cache(config_path)
    except Exception as e:  # defensive
        log.debug('kagi-lenses: cache refresh failed: %s', e)
        return None, query, None

    if not rules:
        return None, query, None

    # Layer 1 — bang prefix wins outright.
    bang_match = _BANG_RE.match(query)
    if bang_match:
        bang_token = bang_match.group(1).lower()
        for rule in rules:
            if bang_token in rule.bangs:
                cleaned = bang_match.group(2).strip()
                log.debug(
                    'kagi-lenses: bang %r → lens %s (%s); cleaned query=%r',
                    bang_token,
                    rule.id,
                    rule.name,
                    cleaned,
                )
                return rule.id, cleaned or query, rule.name

    # Layer 2 — keyword scan, highest-priority match wins, ties broken by
    # config order (sort is stable). We scan once per rule rather than
    # building a unified pattern so we can preserve the priority weighting.
    haystack = query.lower()
    candidates: list[LensRule] = [
        r for r in rules if r.keywords and any(k in haystack for k in r.keywords)
    ]
    if candidates:
        candidates.sort(key=lambda r: r.priority, reverse=True)
        winner = candidates[0]
        log.debug(
            'kagi-lenses: keyword match for query=%r → lens %s (%s)',
            query,
            winner.id,
            winner.name,
        )
        return winner.id, query, winner.name

    return None, query, None


def reset_cache_for_tests() -> None:
    """Test hook: forget the cached rules so the next call re-reads disk."""
    with _cache_lock:
        _cache.path = None
        _cache.mtime = 0.0
        _cache.rules = ()
