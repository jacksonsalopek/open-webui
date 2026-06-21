"""SQLite-backed page-content cache for web loaders.

Keeps fetched page bodies in a tiny on-disk store keyed by url hash so
follow-up queries that revisit the same URL within the TTL skip the upstream
fetcher (Playwright, in our deployment) entirely. The DB lives under
``DATA_DIR`` which is on the persistent open-webui volume, so the cache
survives container restarts.

Configuration (all via env, fail-open):

- ``WEB_PAGE_CACHE_ENABLED`` -- default ``true``.
- ``WEB_PAGE_CACHE_TTL_SECONDS`` -- default 21600 (6h).
- ``WEB_PAGE_CACHE_RECENCY_TTL_SECONDS`` -- default 1800 (30 min). Used when
  the caller passes a TTL override matching a recency-intent query (the NL
  filter parsed an ``after=`` date close to today).

API surface is intentionally tiny: :func:`get`, :func:`put`, and the helper
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
from typing import Optional

from open_webui.env import DATA_DIR

log = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 6 * 60 * 60
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
                    url_hash   TEXT PRIMARY KEY,
                    url        TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    etag       TEXT
                )
                """
            )
            c.commit()
        _INITIALIZED = True


def get(url: str, ttl_seconds: Optional[int] = None) -> Optional[str]:
    """Return cached page content for ``url`` if fresh, else ``None``.

    ``ttl_seconds`` overrides the env-configured default for this lookup.
    A non-positive TTL disables the cache for this lookup (always miss).
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


def put(url: str, content: str, etag: Optional[str] = None) -> None:
    """Store / refresh a page's cached content. No-op for empty bodies."""
    if not is_enabled():
        return
    if not content:
        return
    try:
        _ensure_initialized()
        with _conn() as c:
            c.execute(
                """
                INSERT INTO pages (url_hash, url, content, fetched_at, etag)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    url        = excluded.url,
                    content    = excluded.content,
                    fetched_at = excluded.fetched_at,
                    etag       = excluded.etag
                """,
                (_hash_url(url), url, content, int(time.time()), etag),
            )
            c.commit()
    except Exception as e:
        log.debug('page_cache: put failed for %s: %s', url, e)
