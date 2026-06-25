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
import re
from datetime import date, datetime, timedelta
from typing import Callable, Optional
from urllib.parse import urlparse

try:  # py>=3.9
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover -- backstop for older runtimes
    ZoneInfo = None  # type: ignore[assignment]

import requests
from pydantic import BaseModel, Field, ValidationError, model_validator

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

    @model_validator(mode='after')
    def _drop_inverted_date_range(self) -> 'WebSearchFilter':
        # The LLM occasionally hallucinates a ``before`` that predates ``after``
        # (observed: query="What's today's date?" -> after=today, before=today-30d).
        # An inverted range is an empty interval, which Kagi (and every other
        # provider) honors verbatim -- returning zero results and bubbling up as
        # "No results found from web search". We can't tell which bound the
        # model meant, so drop both and let the search run unfiltered. This is
        # consistent with the module-level fail-open contract.
        if self.after and self.before and self.after > self.before:
            log.debug(
                'nl_filter: dropping inverted date range after=%s > before=%s',
                self.after,
                self.before,
            )
            self.after = None
            self.before = None
        return self

    @model_validator(mode='after')
    def _drop_future_after(self) -> 'WebSearchFilter':
        # ``after > today`` is nonsensical — no provider has content published
        # in the future. Observed failure mode: granite4.1:8b sometimes
        # emits today's date (or later) as ``after`` for queries with zero
        # recency intent ("constraining LLM context windows"), which then
        # filters out every real result. Drop the bound and let the search
        # run unfiltered. We keep ``after == today`` intact because some
        # legitimate breaking-news queries land there.
        today = date.today()
        if self.after and self.after > today:
            log.debug(
                'nl_filter: dropping future after=%s (today=%s)',
                self.after,
                today,
            )
            self.after = None
        if self.before and self.before > today:
            # ``before`` in the future is a no-op for the date filter — every
            # past document is "before the future" — so drop it to keep the
            # provider params clean.
            log.debug(
                'nl_filter: dropping future before=%s (today=%s)',
                self.before,
                today,
            )
            self.before = None
        return self

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
    ``lens``. Kagi covers every dimension we model except region, so the
    generic post-filter is still skipped for kagi.

    Region is intentionally NOT forwarded. Kagi's ``/api/v1/search``
    rejects ISO 3166-1 alpha-2 codes that aren't in its own narrower
    region list with ``search.filters_region_invalid`` HTTP 400 (observed
    on a "Kentucky" prompt where the NL filter inferred ``region: US`` →
    Kagi 400 → entire chat turn aborted). The NL filter's region field
    is still useful for other providers that take real ISO codes, but
    for Kagi we just drop it on the floor.
    """
    params: dict = {}

    filters: dict = {}
    if f.after:
        filters['after'] = f.after.isoformat()
    if f.before:
        filters['before'] = f.before.isoformat()
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
- "language": ISO 639-1 code ONLY if the user explicitly asks for results in a specific language (e.g. "en español", "in French", "auf Deutsch"). A country mention alone is NOT a language request.
- "region": ISO 3166-1 alpha-2 country code (e.g. "US", "GB", "HU") whenever the query is about events/news/topics in a specific country, even if the user did not ask for localization explicitly. Examples: "Hungary news" -> "HU"; "elections in Brazil" -> "BR"; "UK weather" -> "GB". Do NOT set "language" in these cases.

Rules:
- Do NOT invent filters. If the query is a plain informational search with no filtering intent, return {}.
- Be conservative: only populate a field when the query clearly implies it.
- Today's date is {today}. For "recent"/"latest"/"news" style queries, set "after" to roughly 30 days before today.
- For "breaking news" style queries (e.g. "breaking", "happening now", "last 24/48 hours", "developing story", or anything implying the past day or two), set "after" to 1-2 days before today instead of 30.
- Do NOT add the country name itself to "include_keywords" when you have already set "region"; the region filter handles that.
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
    # NOTE: ``_SYSTEM_PROMPT`` contains literal ``{}`` and ``{ "queries": ... }``
    # JSON examples that ``str.format`` interprets as positional placeholders and
    # raises ``IndexError`` on. We only need a single named substitution, so use
    # ``str.replace`` and skip the format minilanguage entirely. (Previously this
    # path crashed every call and silently fell through to an empty filter.)
    system_prompt = _SYSTEM_PROMPT.replace('{today}', today)
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
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


# ── Developer-doc topic routing ───────────────────────────────────────────────
# When a query expresses explicit primary-source intent (e.g. "docs", "api
# reference") AND mentions a known developer topic, we narrow Kagi's lens to
# that ecosystem's canonical doc domains. Kagi's ``sites_included`` is a hard
# allow-list, so we deliberately *only* trigger on explicit doc intent — casual
# queries like "best WinUI alternatives" stay unrestricted.
#
# The topic table is seeded from the ``.claude/agents`` and ``.claude/commands``
# files in repos like ventana/lollipop, where we routinely chase Microsoft Learn
# pages for WinUI 3, ONNX Runtime, MEAI, ML.NET, Whisper.net, Microsoft Agent
# Framework, Fluent Design, MSIX/Store, MSBuild, accessibility, and the
# .NET template engine. To extend, add a ``(keywords, domains)`` entry below;
# the FIRST topic whose keywords match wins, so list specific topics before
# broader ones (e.g. "WinUI" before plain ".NET").

_DEV_DOCS_INTENT_KEYWORDS: tuple[str, ...] = (
    'docs',
    'documentation',
    'api reference',
    'api docs',
    'reference docs',
    'official docs',
    # Microsoft-side intent shortcuts.
    'msdn',
    'learn.microsoft',
    # MDN-side intent shortcuts so "mdn fetch api" / "developer.mozilla addEventListener"
    # trigger routing without the literal word "docs".
    'mdn',
    'developer.mozilla',
)

_DEVELOPER_TOPICS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # WinUI 3 / Windows App SDK -- .claude/agents/winui3-expert.md
    (
        ('winui', 'windows app sdk', 'winappsdk', 'windowsappsdk', 'xaml islands'),
        ('learn.microsoft.com', 'github.com', 'devblogs.microsoft.com'),
    ),
    # ONNX Runtime / ONNX models -- .claude/commands/dotnet-ai.md
    (
        ('onnxruntime', 'onnx runtime', 'onnx'),
        ('onnxruntime.ai', 'learn.microsoft.com', 'github.com'),
    ),
    # Microsoft.Extensions.AI (MEAI) -- .claude/commands/dotnet-ai.md
    (
        ('microsoft.extensions.ai', 'extensions.ai', 'meai', 'ichatclient'),
        ('learn.microsoft.com', 'devblogs.microsoft.com', 'github.com'),
    ),
    # Microsoft Agent Framework -- .claude/commands/dotnet-ai.md
    (
        ('microsoft agent framework', 'agent framework'),
        ('learn.microsoft.com', 'devblogs.microsoft.com', 'github.com'),
    ),
    # ML.NET -- .claude/commands/dotnet-ai.md
    (
        ('ml.net', 'mlnet'),
        ('learn.microsoft.com', 'github.com'),
    ),
    # Whisper.net -- .claude/commands/dotnet-ai.md
    (
        ('whisper.net',),
        ('github.com', 'learn.microsoft.com'),
    ),
    # CommunityToolkit -- .claude/agents/winui3-expert.md
    (
        ('communitytoolkit', 'community toolkit', 'winuiex'),
        ('learn.microsoft.com', 'github.com'),
    ),
    # Fluent Design -- .claude/agents/fluent-design.md
    (
        ('fluent design', 'fluent 2', 'fluent ui', 'mica backdrop', 'segoe fluent icons'),
        ('learn.microsoft.com', 'fluent2.microsoft.design', 'github.com'),
    ),
    # MSIX / Microsoft Store packaging -- .claude/agents/store-packaging.md
    (
        ('msix', 'msixupload', 'appxmanifest', '.appinstaller', 'microsoft store packaging', 'partner center'),
        ('learn.microsoft.com', 'partner.microsoft.com', 'github.com'),
    ),
    # MSBuild -- .claude/agents/msbuild-expert.md
    (
        ('msbuild', '.csproj', '.props', '.targets', 'directory.build.props'),
        ('learn.microsoft.com', 'github.com', 'devblogs.microsoft.com'),
    ),
    # Windows accessibility (UIA, Narrator) -- .claude/agents/accessibility.md
    (
        ('uia', 'narrator', 'automationproperties', 'accessibility insights'),
        ('learn.microsoft.com', 'accessibilityinsights.io', 'w3.org'),
    ),
    # .NET template engine -- .claude/agents/template-engine.md
    (
        ('dotnet new', 'template.json', 'dotnet template engine'),
        ('learn.microsoft.com', 'github.com'),
    ),
    # .NET / C# (catchall, MUST be last among .NET topics)
    (
        ('.net', 'dotnet', 'c#', 'csharp', 'nuget', 'roslyn', 'asp.net'),
        ('learn.microsoft.com', 'devblogs.microsoft.com', 'github.com'),
    ),
    # ── Game development ─────────────────────────────────────────────────────
    # Godot Engine. Placed before the web catchall because GDScript and the
    # engine's class reference are unambiguously primary-source-on-RtD. Plain
    # "godot " (with trailing space, to avoid matching unrelated words like
    # "ungodot") covers most queries; specific Godot APIs are exhaustively
    # routed via :mod:`docs_router` when the user invokes the portal pins.
    (
        ('godot ', 'godotengine', 'gdscript', 'godot engine', 'godot 4', 'godot 3'),
        ('docs.godotengine.org', 'github.com'),
    ),

    # ── Web development ──────────────────────────────────────────────────────
    # Specific frameworks/runtimes come first; the web-platform catchall (MDN /
    # web.dev / WHATWG / W3C) is the very last entry. "react native" / "next"
    # / "remix" / "nuxt" / "solidstart" precede their parent framework so the
    # more specific match wins.

    # React Native (must come before React)
    (
        ('react native', 'reactnative'),
        ('reactnative.dev', 'developer.mozilla.org', 'github.com'),
    ),
    # Next.js (React-based, must come before React)
    (
        ('next.js', 'nextjs', 'next js', 'next 14', 'next 15'),
        ('nextjs.org', 'react.dev', 'developer.mozilla.org'),
    ),
    # Remix / React Router (must come before React)
    (
        ('remix.run', 'react router', 'reactrouter'),
        ('remix.run', 'reactrouter.com', 'developer.mozilla.org'),
    ),
    # React
    (
        ('reactjs', 'react.js', 'react.dev', 'jsx', 'usestate', 'useeffect',
         'react hook', 'react component', 'react server component', 'react '),
        ('react.dev', 'developer.mozilla.org', 'github.com'),
    ),
    # Nuxt (Vue-based, must come before Vue)
    (
        ('nuxt.js', 'nuxtjs', 'nuxt 3', 'nuxt 4', 'nuxt '),
        ('nuxt.com', 'vuejs.org', 'developer.mozilla.org'),
    ),
    # Vue
    (
        ('vue.js', 'vuejs', 'vue 3', 'vue composition', 'vue '),
        ('vuejs.org', 'developer.mozilla.org', 'github.com'),
    ),
    # SolidStart (Solid-based, must come before Solid)
    (
        ('solidstart', 'solid-start', 'solid start'),
        ('docs.solidjs.com', 'developer.mozilla.org'),
    ),
    # SolidJS
    (
        ('solidjs', 'solid.js', 'solid-js'),
        ('docs.solidjs.com', 'developer.mozilla.org', 'github.com'),
    ),
    # SvelteKit / Svelte
    (
        ('sveltekit', 'svelte.dev', 'svelte '),
        ('svelte.dev', 'kit.svelte.dev', 'developer.mozilla.org'),
    ),
    # Angular
    (
        ('angular.dev', 'angular.io', 'angularjs', 'angular '),
        ('angular.dev', 'angular.io', 'developer.mozilla.org'),
    ),
    # Astro
    (
        ('astro.build', 'astrojs', 'astro framework', 'astro '),
        ('docs.astro.build', 'developer.mozilla.org'),
    ),
    # Lit / Web Components. Note: ``shadow dom`` deliberately lives in the
    # web-platform catchall below, not here — it's a platform feature, not a
    # Lit-specific construct, so "mdn shadow dom" should win MDN, not Lit.
    (
        ('lit-element', 'lit-html', 'lit.dev', 'web component', 'webcomponents',
         'custom element'),
        ('lit.dev', 'web.dev', 'developer.mozilla.org'),
    ),
    # Tailwind CSS
    (
        ('tailwindcss', 'tailwind css', 'tailwind ', 'tailwind.config'),
        ('tailwindcss.com', 'developer.mozilla.org'),
    ),
    # TypeScript
    (
        ('typescript', 'tsconfig', 'tsx ', '.tsx', '.d.ts'),
        ('typescriptlang.org', 'developer.mozilla.org', 'github.com'),
    ),
    # Vite
    (
        ('vitejs', 'vite.dev', 'vite.config', 'vite '),
        ('vite.dev', 'developer.mozilla.org', 'github.com'),
    ),
    # Deno
    (
        ('deno.land', 'docs.deno.com', 'deno deploy', 'deno '),
        ('docs.deno.com', 'developer.mozilla.org'),
    ),
    # Bun
    (
        ('bun.sh', 'bunjs', 'bun.js', 'bun runtime', 'bun '),
        ('bun.sh', 'developer.mozilla.org'),
    ),
    # Node.js (must come AFTER Next.js / Nuxt / Bun / Deno catchalls above)
    (
        ('node.js', 'nodejs', 'npm ', 'package.json', 'pnpm ', 'yarn '),
        ('nodejs.org', 'developer.mozilla.org', 'github.com'),
    ),
    # Web platform catchall — HTML / CSS / JavaScript / DOM / Web APIs / specs.
    # MUST be the last web-dev entry; matches anything that mentions a core
    # web-platform surface OR explicitly invokes MDN.
    (
        ('mdn', 'developer.mozilla', 'mozilla developer',
         'html', 'css', 'javascript', 'ecmascript', 'dom ', 'web api',
         'webapi', 'fetch api', 'web platform', 'whatwg', 'w3c',
         'service worker', 'shadow dom', 'web component', 'http header',
         'cookie ', 'cors '),
        ('developer.mozilla.org', 'web.dev', 'html.spec.whatwg.org', 'w3.org'),
    ),
)


# When the user explicitly names a docs portal, we honor that pin instead of
# falling through to the framework table. Otherwise "mdn shadow dom" would
# route to the Lit/Web Components topic just because it mentions DOM concepts.
_PORTAL_PINS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ('mdn', 'developer.mozilla', 'mozilla developer'),
        ('developer.mozilla.org', 'web.dev', 'html.spec.whatwg.org', 'w3.org'),
    ),
    (
        ('msdn', 'learn.microsoft'),
        ('learn.microsoft.com', 'devblogs.microsoft.com', 'github.com'),
    ),
    (
        ('godotengine', 'godot engine', 'docs.godotengine', 'gdscript'),
        ('docs.godotengine.org', 'github.com'),
    ),
)


def _detect_developer_topic_domains(query: str) -> list[str]:
    """Return canonical doc domains when ``query`` expresses dev-doc intent.

    Returns an empty list if no doc-intent keyword matched, or if a doc-intent
    keyword matched but no topic did. Only the first matching topic's domains
    are returned, so order topics in :data:`_DEVELOPER_TOPICS` from most
    specific to most general. Explicit portal pins (e.g. "mdn ...") win over
    the framework table so the user's stated source is respected.
    """
    q = query.lower()
    if not any(kw in q for kw in _DEV_DOCS_INTENT_KEYWORDS):
        return []
    for pin_keywords, pin_domains in _PORTAL_PINS:
        if any(kw in q for kw in pin_keywords):
            return list(pin_domains)
    for keywords, domains in _DEVELOPER_TOPICS:
        if any(kw in q for kw in keywords):
            return list(domains)
    return []


def extract_filter_from_query(query: str) -> WebSearchFilter:
    """Convenience wrapper used by the search pipeline.

    Combines two layers:

    1. The LLM-parsed :class:`WebSearchFilter` from :func:`parse_nl_filter`.
    2. Deterministic developer-doc routing via
       :func:`_detect_developer_topic_domains`, which adds canonical doc
       domains for known dev topics when the query expresses explicit
       primary-source intent.

    Honors ``ENABLE_WEB_SEARCH_NL_FILTER`` (default on) and always fails open
    so search keeps working if Layer 1 is unavailable. Layer 2 runs even when
    Layer 1 fails, since it has no external dependencies.
    """
    if not _env_flag('ENABLE_WEB_SEARCH_NL_FILTER', True):
        return WebSearchFilter()

    try:
        nl = parse_nl_filter(query)
    except Exception as e:  # defensive: never let filtering break search
        log.debug(f'nl_filter: extraction failed ({e})')
        nl = WebSearchFilter()

    topic_domains = _detect_developer_topic_domains(query)
    if topic_domains:
        merged = sorted({*nl.include_domains, *topic_domains})
        nl = nl.model_copy(update={'include_domains': merged})
        log.debug(
            'nl_filter: developer-doc routing matched query=%r -> include_domains=%s',
            query,
            merged,
        )

    return nl


# ── Pure date/time queries ───────────────────────────────────────────────────
# "What's today's date?" / "current time" style questions answer themselves
# from the server clock — there's no useful web result to retrieve, and
# spending a Kagi call (plus an NL-filter LLM call, plus a page fetch + embed)
# on each generated sub-query is pure waste. We intercept these deterministically
# upstream of the search pipeline and synthesize the answer from Python's
# datetime.
#
# Matching is intentionally strict (whole-query, anchored regex on the
# lowercased+trimmed query) so anything with additional content words —
# "Apple Q4 2024 earnings date", "release date for GTA 6" — falls through to
# real search.

# A reusable tail that matches the natural "is it (right now)?" / "(right )?now"
# tail people stick on the end of date/time questions. Encoded once so both
# the date and time patterns expand uniformly; without it, the previous
# patterns missed "What time is it now?" (the most common phrasing!) and
# fell through to a real web search where the LLM made up an answer from
# whatever search snippets came back.
_DT_TAIL = r"(?:\s+today|\s+is\s+it(?:\s+(?:right\s+)?now)?|\s+(?:right\s+)?now)?"

_DATETIME_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # date-only
    re.compile(r"^(?:what(?:'?s| is|s)?\s+)?(?:the\s+)?(?:current\s+|today'?s?\s+)?date" + _DT_TAIL + r"\??$"),
    re.compile(r"^today'?s?\s+date\??$"),
    # time-only
    re.compile(r"^(?:what(?:'?s| is|s)?\s+)?(?:the\s+)?(?:current\s+)?time" + _DT_TAIL + r"\??$"),
    re.compile(r"^what\s+time\s+is\s+it(?:\s+(?:right\s+)?now)?\??$"),
    # combined date+time
    re.compile(r"^(?:what(?:'?s| is|s)?\s+)?(?:the\s+)?(?:current\s+|today'?s?\s+)?date\s+(?:and|&|\+)\s+time" + _DT_TAIL + r"\??$"),
    re.compile(r"^(?:current\s+)?datetime\??$"),
    # weekday
    re.compile(r"^what\s+day\s+(?:is\s+(?:it|today)|of\s+the\s+week\s+is\s+it)\??$"),
    re.compile(r"^(?:what'?s|whats)\s+today\??$"),
)


_SMART_PUNCT_TRANSLATION = str.maketrans({
    # Typographic single quotes (U+2018, U+2019, U+201B) → straight apostrophe.
    # macOS / iOS autocorrect rewrites the user's ' into ’, which would
    # otherwise miss every "today's date" / "what's today" match.
    '\u2018': "'",
    '\u2019': "'",
    '\u201B': "'",
    # Typographic double quotes (U+201C..U+201F) → straight double.
    '\u201C': '"',
    '\u201D': '"',
    '\u201E': '"',
    '\u201F': '"',
})


def is_pure_datetime_query(query: Optional[str]) -> bool:
    """True iff ``query`` is *only* asking for the current date/time/weekday.

    Whole-query match. "today's date" -> True; "Apple earnings date" -> False.
    Smart quotes (e.g. macOS auto-replacing ' with ’) are normalized first so
    the same regex covers both forms.
    """
    if not query:
        return False
    q = query.strip().translate(_SMART_PUNCT_TRANSLATION).lower()
    if not q:
        return False
    return any(pattern.match(q) for pattern in _DATETIME_QUERY_PATTERNS)


def synthesize_datetime_answer(
    timezone_name: Optional[str] = None,
    *,
    current_date: Optional[str] = None,
    current_time: Optional[str] = None,
    current_weekday: Optional[str] = None,
    user_query: Optional[str] = None,
) -> str:
    """Return a human-readable current-date/time blurb suitable for injection
    as a "search result" snippet.

    When the caller can supply pre-localized strings (``current_date``,
    ``current_time``, ``current_weekday``) — typically from Open WebUI's
    ``{{CURRENT_DATE}}`` / ``{{CURRENT_TIME}}`` / ``{{CURRENT_WEEKDAY}}``
    template variables, which it substitutes from the browser's locale — we
    use those verbatim so the answer matches the user's timezone without
    requiring the IANA tz database in the container. If those are missing we
    fall back to ``ZoneInfo(timezone_name)`` when tzdata is available, then
    finally to the server's local clock.

    When ``user_query`` is provided, it is echoed verbatim into the snippet
    along with an explicit "reply in the same language" directive. This is
    the language-anchor fix for command-a-plus W4A4: on very short prompts
    ("what time is it" = 4 tokens), the heavily-quantized multilingual model
    sometimes ignored the global RAG template's "respond in the same language
    as the user's query" guideline and answered in Korean (or another
    language it has strong training mass for). Echoing the query into the
    *source* content doubles the language signal in the model's attention
    window, and the trailing directive wins on recency bias inside the long
    RAG-template wrapper.
    """
    if current_date and current_time:
        weekday = current_weekday or ''
        date_prefix = f'{weekday}, {current_date}' if weekday else current_date
        tz_label = timezone_name or 'server local time'
        snippet = (
            f"Current date: {date_prefix}.\n"
            f"Current time: {current_time} ({tz_label}).\n"
        )
        return _append_language_anchor(snippet, user_query)

    now: datetime
    tz_label = timezone_name or 'server local time'
    if timezone_name and ZoneInfo is not None:
        try:
            now = datetime.now(ZoneInfo(timezone_name))
        except Exception:
            now = datetime.now().astimezone()
            tz_label = 'server local time'
    else:
        now = datetime.now().astimezone()

    # strftime portability: %-d / %-I are GNU extensions and crash on Windows.
    # Strip leading zeros manually so this works everywhere.
    day = str(int(now.strftime('%d')))
    hour12 = str(int(now.strftime('%I')))
    snippet = (
        f"Current date: {now.strftime('%A, %B')} {day}, {now.strftime('%Y')}.\n"
        f"Current time: {hour12}:{now.strftime('%M %p')} ({tz_label}).\n"
        f"ISO 8601: {now.isoformat(timespec='seconds')}."
    )
    return _append_language_anchor(snippet, user_query)


def _append_language_anchor(snippet: str, user_query: Optional[str]) -> str:
    """Append a user-query echo + language directive to ``snippet`` if a query
    was supplied. No-op when ``user_query`` is missing or empty so callers that
    don't have the query handy (older callers, direct tests) get the bare
    date/time blurb unchanged.

    Truncates very long queries defensively — the short-circuit path is gated
    on ``is_pure_datetime_query`` which only matches short whole-query
    patterns, so this should never fire in practice, but it prevents a
    pathological caller from blowing up the snippet size.
    """
    if not user_query:
        return snippet
    q = user_query.strip()
    if not q:
        return snippet
    if len(q) > 200:
        q = q[:200] + '…'
    return (
        f'{snippet.rstrip()}\n\n'
        f'User question: "{q}"\n'
        f'Reply in the same language as the user question above.\n'
    )
