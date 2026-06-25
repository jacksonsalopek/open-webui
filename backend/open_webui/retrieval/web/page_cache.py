"""SQLite-backed page-content cache for web loaders.

Keeps fetched page bodies in a tiny on-disk store keyed by url hash so
follow-up queries that revisit the same URL within the TTL skip the upstream
fetcher (Playwright, in our deployment) entirely. The DB lives under
``DATA_DIR`` which is on the persistent open-webui volume, so the cache
survives container restarts.

Configuration (all via env, fail-open):

- ``WEB_PAGE_CACHE_ENABLED`` -- default ``true``.
- ``WEB_PAGE_CACHE_TTL_SECONDS`` -- default 43200 (12h). The default was
  bumped from 6h to 12h when conditional-GET support landed: even when a
  cached entry's body expires, we still keep its ``ETag`` /
  ``Last-Modified`` headers and the next fetch sends an
  ``If-None-Match`` / ``If-Modified-Since`` request. Most servers reply
  with a 23-byte ``304 Not Modified`` instead of a full re-render, so
  "expired" entries can re-validate at essentially zero cost. A longer
  default TTL is therefore safe -- the worst case for a stale-but-
  unchanged page is one extra ~50ms round trip, not a re-fetch of the
  full body.
- ``WEB_PAGE_CACHE_RECENCY_TTL_SECONDS`` -- default 1800 (30 min). Used when
  the caller passes a TTL override matching a recency-intent query (the NL
  filter parsed an ``after=`` date close to today).

API surface is intentionally tiny: :func:`get`, :func:`put`,
:func:`get_with_validators`, :func:`touch`, and the helper
:func:`recency_ttl_seconds` / :func:`default_ttl_seconds` accessors. All
errors are swallowed and logged at DEBUG so a corrupt or unwritable DB
never breaks a search.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Optional, Tuple

from open_webui.env import DATA_DIR

log = logging.getLogger(__name__)

# Bumped from 6h to 12h once conditional GETs landed -- stale entries can
# re-validate via If-None-Match / If-Modified-Since for the cost of a
# single 304 round trip, so a longer default TTL is safe.
_DEFAULT_TTL_SECONDS = 12 * 60 * 60
_DEFAULT_RECENCY_TTL_SECONDS = 30 * 60

_DB_PATH = DATA_DIR / 'web_page_cache.db'
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


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


def is_enabled() -> bool:
    return _env_flag('WEB_PAGE_CACHE_ENABLED', True)


def default_ttl_seconds() -> int:
    return _env_int('WEB_PAGE_CACHE_TTL_SECONDS', _DEFAULT_TTL_SECONDS)


def recency_ttl_seconds() -> int:
    return _env_int('WEB_PAGE_CACHE_RECENCY_TTL_SECONDS', _DEFAULT_RECENCY_TTL_SECONDS)


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


@contextmanager
def _conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh connection per call: sqlite3 connections are NOT safe to share
    # across threads without extra locking, and the alazy_load path runs
    # inside asyncio with the cache touched from multiple workers.
    c = sqlite3.connect(str(_DB_PATH), timeout=5.0)
    try:
        yield c
    finally:
        c.close()


def _ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        with _conn() as c:
            # WAL lets readers and a single writer coexist without blocking
            # each other; matters because cache writes happen from multiple
            # coroutines in the same alazy_load batch.
            c.execute('PRAGMA journal_mode=WAL;')
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS pages (
                    url_hash      TEXT PRIMARY KEY,
                    url           TEXT NOT NULL,
                    content       TEXT NOT NULL,
                    fetched_at    INTEGER NOT NULL,
                    etag          TEXT,
                    last_modified TEXT
                )
                """
            )
            # Migration: add ``last_modified`` to pre-existing DBs that
            # were created before conditional-GET support landed. SQLite
            # doesn't support ``ADD COLUMN IF NOT EXISTS`` (3.35+ has it
            # for some clauses but not portably across the bundled
            # versions), so we just try/except the duplicate-column
            # error. On a fresh DB the column is already part of the
            # CREATE below; on an old one this catches up.
            try:
                c.execute('ALTER TABLE pages ADD COLUMN last_modified TEXT')
            except sqlite3.OperationalError as e:
                if 'duplicate column name' not in str(e).lower():
                    # Not the expected "column already exists" case --
                    # log so we notice a real migration failure but
                    # keep going since the cache is fail-open anyway.
                    log.debug('page_cache: ALTER TABLE last_modified: %s', e)
            c.commit()
        _INITIALIZED = True


def get(url: str, ttl_seconds: Optional[int] = None) -> Optional[str]:
    """Return cached page content for ``url`` if fresh, else ``None``.

    ``ttl_seconds`` overrides the env-configured default for this lookup.
    A non-positive TTL disables the cache for this lookup (always miss).
    ``ttl_seconds=None`` and ``ttl_seconds=0`` are distinct -- pass a
    huge value or use the dedicated ``_force_fresh=True`` plumbing in
    callers to ignore TTL entirely.
    """
    if not is_enabled():
        return None
    ttl = ttl_seconds if ttl_seconds is not None else default_ttl_seconds()
    if ttl <= 0:
        return None
    try:
        _ensure_initialized()
        cutoff = int(time.time()) - ttl
        with _conn() as c:
            row = c.execute(
                'SELECT content FROM pages WHERE url_hash = ? AND fetched_at >= ?',
                (_hash_url(url), cutoff),
            ).fetchone()
        if row is None:
            return None
        return row[0]
    except Exception as e:
        log.debug('page_cache: get failed for %s: %s', url, e)
        return None


def get_with_validators(
    url: str, ttl_seconds: Optional[int] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(content, etag, last_modified)`` for ``url``.

    Three logical states:

    1. Cache miss: ``(None, None, None)``. No entry, or cache disabled.
    2. Fresh hit: ``(content, etag, last_modified)`` with content
       populated. The caller can use ``content`` directly and skip
       the fetch entirely.
    3. Stale hit with validators: ``(None, etag, last_modified)``.
       The entry exists but ``fetched_at`` is older than ``ttl``; the
       caller should send a conditional GET (``If-None-Match`` /
       ``If-Modified-Since``) and, on ``304 Not Modified``, fall back
       to :func:`get` with no TTL to retrieve the body and then call
       :func:`touch` to refresh the freshness window.

    A stale entry with no recorded validators returns ``(None, None,
    None)`` -- there's nothing to validate against, so the caller has
    to do a full re-fetch anyway.
    """
    if not is_enabled():
        return (None, None, None)
    ttl = ttl_seconds if ttl_seconds is not None else default_ttl_seconds()
    if ttl <= 0:
        return (None, None, None)
    try:
        _ensure_initialized()
        with _conn() as c:
            row = c.execute(
                'SELECT content, fetched_at, etag, last_modified '
                'FROM pages WHERE url_hash = ?',
                (_hash_url(url),),
            ).fetchone()
        if row is None:
            return (None, None, None)
        content, fetched_at, etag, last_modified = row
        cutoff = int(time.time()) - ttl
        if fetched_at >= cutoff:
            # Fresh -- return the body alongside validators in case the
            # caller wants to opportunistically refresh anyway.
            return (content, etag, last_modified)
        # Stale: only useful if we have at least one validator. Without
        # one the caller has no choice but to do a full re-fetch, which
        # they would have done anyway on a plain cache miss.
        if etag or last_modified:
            return (None, etag, last_modified)
        return (None, None, None)
    except Exception as e:
        log.debug('page_cache: get_with_validators failed for %s: %s', url, e)
        return (None, None, None)


def get_force(url: str) -> Optional[str]:
    """Return cached content ignoring TTL. Used to back conditional-GET 304s.

    On ``304 Not Modified`` the server explicitly told us the previously-
    cached body is still valid, so the TTL check would be a misleading
    no-op. ``touch(url)`` should be called alongside this to refresh
    ``fetched_at`` -- otherwise every conditional GET re-validates on
    every subsequent search.
    """
    if not is_enabled():
        return None
    try:
        _ensure_initialized()
        with _conn() as c:
            row = c.execute(
                'SELECT content FROM pages WHERE url_hash = ?',
                (_hash_url(url),),
            ).fetchone()
        return row[0] if row else None
    except Exception as e:
        log.debug('page_cache: get_force failed for %s: %s', url, e)
        return None


def touch(url: str) -> None:
    """Refresh an entry's ``fetched_at`` without rewriting the body.

    Called after a successful ``304 Not Modified`` response so the
    entry stays "fresh" until the next TTL boundary instead of forcing
    a conditional GET on every search.
    """
    if not is_enabled():
        return
    try:
        _ensure_initialized()
        with _conn() as c:
            c.execute(
                'UPDATE pages SET fetched_at = ? WHERE url_hash = ?',
                (int(time.time()), _hash_url(url)),
            )
            c.commit()
    except Exception as e:
        log.debug('page_cache: touch failed for %s: %s', url, e)


def put(
    url: str,
    content: str,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> None:
    """Store / refresh a page's cached content. No-op for empty bodies.

    ``etag`` and ``last_modified`` are captured from the HTTP response
    so future searches can do conditional GETs (``If-None-Match`` /
    ``If-Modified-Since``) and re-validate via cheap 304s instead of
    re-downloading the full body.
    """
    if not is_enabled():
        return
    if not content:
        return
    try:
        _ensure_initialized()
        with _conn() as c:
            c.execute(
                """
                INSERT INTO pages (url_hash, url, content, fetched_at, etag, last_modified)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    url           = excluded.url,
                    content       = excluded.content,
                    fetched_at    = excluded.fetched_at,
                    etag          = excluded.etag,
                    last_modified = excluded.last_modified
                """,
                (
                    _hash_url(url),
                    url,
                    content,
                    int(time.time()),
                    etag,
                    last_modified,
                ),
            )
            c.commit()
    except Exception as e:
        log.debug('page_cache: put failed for %s: %s', url, e)
