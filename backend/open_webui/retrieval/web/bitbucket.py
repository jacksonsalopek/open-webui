"""Bitbucket Cloud API adapter for the web-search pipeline.

Single-workspace, single-token model. Configured via the env vars
``BITBUCKET_ACCESS_TOKEN`` (Workspace/Repository Access Token recommended;
user-scoped API tokens also work — both are Bearer-auth strings) and
``BITBUCKET_WORKSPACE`` (the slug, e.g. ``acmecorp``). No per-user OAuth,
no callback endpoints, no token storage: every user's searches share the
same service token, which makes the read scope of that token the access-
control boundary. Scope the token to "Repositories: Read" only.

Three search modes, fanned out in parallel and merged by the upstream
search dispatcher:

1. **Code Search** — ``GET /2.0/workspaces/{ws}/search/code``. Full-text
   match against file contents across all repos in the workspace. Returns
   file path + line snippets. This is the real workhorse; it's what users
   actually want when they ask "where do we do X in our codebase". A
   paid-tier feature on Bitbucket Cloud — falls through cleanly with a
   logged 403 on free plans rather than crashing the search.

2. **Repository Search** — ``GET /2.0/repositories/{ws}?q={BBQL}``. Not
   really "search" — BBQL is a property-filter DSL. We build a query that
   matches ``name`` and ``description`` substrings. Useful for "is there
   a repo named X?" / "what repos do we have for Y?" intents that code
   search misses.

3. **Pull Request Search** — ``GET /2.0/repositories/{ws}/{repo}/pullrequests``
   ``?q=title~"query"``. Bitbucket Cloud has NO workspace-wide PR endpoint,
   so this only runs when the caller passes a specific ``repo_slug``.
   The upstream router extracts a repo slug from queries that contain
   ``workspace/reposlug`` patterns or explicit ``!bb-pr <repo>`` bangs.
   Without a repo target, the PR leg is silently skipped.

Failure mode: every API call individually catches network / auth / 4xx
errors and returns ``[]`` for that leg. A partial outage (e.g. PR endpoint
down but code search working) still produces useful results.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_webui.retrieval.web.main import SearchResult

log = logging.getLogger(__name__)

BB_API = 'https://api.bitbucket.org/2.0'
BB_ORIGIN = 'https://bitbucket.org'
REQUEST_TIMEOUT = 10

# Cap per-mode results; the upstream merge caps the final union via
# WEB_SEARCH_RESULT_COUNT anyway, but we cap here too so a runaway page
# count doesn't pull thousands of records just to throw them away.
_PER_MODE_HARD_CAP = 25


def _build_session(token: str) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        # 401 / 403 are auth/permission errors — retrying won't help, fail
        # fast. 429 means we hit Bitbucket's rate limit; backoff helps there.
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({'GET'}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update(
        {
            'User-Agent': 'open-webui/bitbucket-adapter (+https://openwebui.com)',
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
        }
    )
    return session


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + '…'


def _bbql_escape(value: str) -> str:
    """Escape a value for a BBQL substring filter ('name ~ "X"').

    BBQL strings are double-quoted; embedded double quotes need backslash-
    escaping. Newlines / backslashes get scrubbed entirely because Bitbucket
    rejects the filter outright when they're present.
    """
    return value.replace('\\', '').replace('"', '\\"').replace('\n', ' ').strip()


def search_bitbucket_code(
    query: str,
    workspace: str,
    token: str,
    count: int,
) -> list[SearchResult]:
    """Search code contents across all repos in ``workspace``.

    Endpoint docs: https://developer.atlassian.com/cloud/bitbucket/rest/api-group-search/

    Each hit becomes one SearchResult with:
      - link: file URL on bitbucket.org (with the matched line anchor when
        the API returns one)
      - title: ``{repo_slug} · {file_path}``
      - snippet: line-numbered content_match excerpts (truncated)
    """
    if not workspace or not token:
        return []

    session = _build_session(token)
    url = f'{BB_API}/workspaces/{workspace}/search/code'
    params = {
        'search_query': query,
        'pagelen': min(_PER_MODE_HARD_CAP, max(count, 1)),
    }

    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning('bitbucket-code: request failed: %s', e)
        return []

    if response.status_code == 403:
        # Code Search is a paid-tier feature on some Bitbucket Cloud plans.
        # Don't spam errors — log once at debug and skip the leg.
        log.debug(
            'bitbucket-code: 403 (search may be a paid-tier feature for workspace %r); '
            'skipping code-search leg',
            workspace,
        )
        return []
    if not response.ok:
        log.warning(
            'bitbucket-code: status=%s body=%s',
            response.status_code,
            response.text[:300],
        )
        return []

    try:
        payload = response.json()
    except ValueError as e:
        log.warning('bitbucket-code: bad JSON: %s', e)
        return []

    results: list[SearchResult] = []
    for hit in payload.get('values') or []:
        if not isinstance(hit, dict):
            continue
        file_info = hit.get('file') or {}
        repo = (file_info.get('commit') or {}).get('repository') or {}
        repo_slug = repo.get('full_name') or repo.get('name') or ''
        file_path = file_info.get('path') or ''
        if not (repo_slug and file_path):
            continue

        # ``links.self.href`` points at the API; we want the html URL.
        html_link = (file_info.get('links') or {}).get('self', {}).get('href')
        if html_link:
            # API href -> https://bitbucket.org/{repo}/src/{commit}/{path}
            html_link = html_link.replace(
                'api.bitbucket.org/2.0/repositories', 'bitbucket.org'
            ).replace('/src/', '/src/', 1)
        else:
            html_link = f'{BB_ORIGIN}/{repo_slug}/src/HEAD/{file_path}'

        # Build a snippet from the content_matches (lines + line numbers).
        # The API returns each match as a list of lines with optional
        # match-segment offsets within each line; we only need the line text.
        snippet_lines: list[str] = []
        for match in hit.get('content_matches') or []:
            for line in match.get('lines') or []:
                ln_no = line.get('line')
                segments = line.get('segments') or []
                line_text = ''.join(s.get('text', '') for s in segments)
                if not line_text.strip():
                    continue
                prefix = f'{ln_no:>4}: ' if isinstance(ln_no, int) else '    : '
                snippet_lines.append(prefix + line_text.rstrip())
                if len(snippet_lines) >= 8:
                    break
            if len(snippet_lines) >= 8:
                break

        snippet = _truncate('\n'.join(snippet_lines), 1200)
        results.append(
            SearchResult(
                link=html_link,
                title=f'{repo_slug} · {file_path}',
                snippet=snippet,
            )
        )
        if len(results) >= count:
            break

    return results[:count]


def search_bitbucket_repos(
    query: str,
    workspace: str,
    token: str,
    count: int,
) -> list[SearchResult]:
    """Filter workspace repos by name/description substring.

    Endpoint docs: https://developer.atlassian.com/cloud/bitbucket/rest/api-group-repositories/

    Not a true search — Bitbucket exposes a BBQL ``q`` parameter that filters
    repository properties. We build ``name ~ "X" OR description ~ "X"`` so
    the result set covers both the slug and the human description. Sort by
    most-recently-updated so active repos surface first.
    """
    if not workspace or not token or not query.strip():
        return []

    session = _build_session(token)
    url = f'{BB_API}/repositories/{workspace}'
    escaped = _bbql_escape(query)
    bbql = f'name ~ "{escaped}" OR description ~ "{escaped}"'
    params = {
        'q': bbql,
        'sort': '-updated_on',
        'pagelen': min(_PER_MODE_HARD_CAP, max(count, 1)),
    }

    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning('bitbucket-repos: request failed: %s', e)
        return []

    if not response.ok:
        log.warning(
            'bitbucket-repos: status=%s body=%s',
            response.status_code,
            response.text[:300],
        )
        return []

    try:
        payload = response.json()
    except ValueError as e:
        log.warning('bitbucket-repos: bad JSON: %s', e)
        return []

    results: list[SearchResult] = []
    for repo in payload.get('values') or []:
        if not isinstance(repo, dict):
            continue
        full_name = repo.get('full_name') or repo.get('name') or ''
        description = (repo.get('description') or '').strip()
        updated = (repo.get('updated_on') or '')[:10]
        language = repo.get('language') or ''
        is_private = repo.get('is_private')

        if not full_name:
            continue

        meta_parts: list[str] = []
        if language:
            meta_parts.append(f'[{language}]')
        if updated:
            meta_parts.append(f'updated {updated}')
        if is_private is True:
            meta_parts.append('private')
        meta_line = ' · '.join(meta_parts)

        snippet_parts = [p for p in (meta_line, description) if p]
        snippet = _truncate('\n'.join(snippet_parts), 800)

        results.append(
            SearchResult(
                link=f'{BB_ORIGIN}/{full_name}',
                title=full_name,
                snippet=snippet,
            )
        )
        if len(results) >= count:
            break

    return results[:count]


def search_bitbucket_prs(
    query: str,
    workspace: str,
    repo_slug: str,
    token: str,
    count: int,
) -> list[SearchResult]:
    """Search PR titles within ``workspace/repo_slug`` for ``query``.

    Endpoint docs: https://developer.atlassian.com/cloud/bitbucket/rest/api-group-pullrequests/

    Skipped unless a specific ``repo_slug`` was resolved upstream — Bitbucket
    Cloud has no workspace-wide PR search and iterating every repo for every
    user query is a non-starter at API cost.

    Searches both OPEN and MERGED PRs (excludes DECLINED) on the assumption
    that historical context is at least as useful as in-flight work.
    """
    if not workspace or not token or not repo_slug:
        return []

    # Reject obviously-malformed slugs to avoid hitting Bitbucket with junk.
    if '/' in repo_slug or ' ' in repo_slug:
        log.debug('bitbucket-prs: invalid repo_slug %r; skipping', repo_slug)
        return []

    session = _build_session(token)
    url = f'{BB_API}/repositories/{workspace}/{repo_slug}/pullrequests'
    escaped = _bbql_escape(query)
    bbql = f'title ~ "{escaped}" AND (state = "OPEN" OR state = "MERGED")'
    params = {
        'q': bbql,
        'sort': '-updated_on',
        'pagelen': min(_PER_MODE_HARD_CAP, max(count, 1)),
    }

    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning('bitbucket-prs: request failed: %s', e)
        return []

    if not response.ok:
        log.debug(
            'bitbucket-prs: status=%s for %s/%s (likely not found or no perm)',
            response.status_code,
            workspace,
            repo_slug,
        )
        return []

    try:
        payload = response.json()
    except ValueError as e:
        log.warning('bitbucket-prs: bad JSON: %s', e)
        return []

    results: list[SearchResult] = []
    for pr in payload.get('values') or []:
        if not isinstance(pr, dict):
            continue
        pr_id = pr.get('id')
        title = (pr.get('title') or '').strip()
        state = pr.get('state') or ''
        author = ((pr.get('author') or {}).get('display_name') or '').strip()
        updated = (pr.get('updated_on') or '')[:10]
        description = (pr.get('summary') or {}).get('raw') or pr.get('description') or ''
        description = description.strip()

        if pr_id is None or not title:
            continue

        link = f'{BB_ORIGIN}/{workspace}/{repo_slug}/pull-requests/{pr_id}'

        meta_parts = [f'#{pr_id}', state.lower()]
        if author:
            meta_parts.append(f'by {author}')
        if updated:
            meta_parts.append(f'updated {updated}')
        meta_line = ' · '.join(meta_parts)

        snippet_parts = [meta_line]
        if description:
            snippet_parts.append(description)
        snippet = _truncate('\n'.join(snippet_parts), 800)

        results.append(
            SearchResult(
                link=link,
                title=f'{repo_slug} PR: {title}',
                snippet=snippet,
            )
        )
        if len(results) >= count:
            break

    return results[:count]


def search_bitbucket(
    query: str,
    workspace: str,
    token: str,
    count: int,
    *,
    repo_slug: Optional[str] = None,
    include_code: bool = True,
    include_repos: bool = True,
    include_prs: bool = True,
) -> list[SearchResult]:
    """Fan out to code / repo / PR search and merge results round-robin.

    Order of legs in the merge matters because the round-robin interleave
    preserves the top hit from each leg first: code first (highest-signal
    for "where do we do X"), then repos (useful for "what repos exist for
    Y"), then PRs (lowest-signal in the general case, also frequently
    skipped without a repo_slug). Per-leg cap = ``count`` so a single
    overflowing leg can't crowd out a thinner one in the merge.

    Failure tolerance: each leg's API call is wrapped in its own try; a
    downed PR endpoint doesn't kill the code-search results.
    """
    if not workspace or not token or not query.strip():
        return []

    bundles: list[list[SearchResult]] = []
    if include_code:
        bundles.append(search_bitbucket_code(query, workspace, token, count))
    if include_repos:
        bundles.append(search_bitbucket_repos(query, workspace, token, count))
    if include_prs and repo_slug:
        bundles.append(
            search_bitbucket_prs(query, workspace, repo_slug, token, count)
        )

    # Round-robin interleave so each leg's top result lands in the merged
    # top-N. Dedup by link to avoid identical hits (rare but possible when a
    # repo's README matches both code search and repo description).
    seen: set[str] = set()
    merged: list[SearchResult] = []
    indices = [0] * len(bundles)
    while len(merged) < count and any(
        indices[i] < len(bundles[i]) for i in range(len(bundles))
    ):
        for i, bundle in enumerate(bundles):
            if indices[i] >= len(bundle):
                continue
            result = bundle[indices[i]]
            indices[i] += 1
            if result.link in seen:
                continue
            seen.add(result.link)
            merged.append(result)
            if len(merged) >= count:
                break

    log.debug(
        'bitbucket: merged %s -> %d result(s)',
        [len(b) for b in bundles],
        len(merged),
    )
    return merged
