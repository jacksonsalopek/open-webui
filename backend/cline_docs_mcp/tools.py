"""MCP tool definitions wrapping Open WebUI's doc-retrieval adapters.

Each ``@mcp.tool()`` is a thin pydantic-typed shim over the corresponding
``open_webui.retrieval.web.*`` ``search_*`` function. The wrappers:

- Pass ``search_filter=None`` to every adapter. The NL filter is a chat-turn
  concept (it parses recency/region/language intent out of a conversational
  query); Cline issues explicit tool calls with explicit args, so there's no
  natural-language envelope to parse. Per-engine native knobs (``version``,
  ``product``, ``category``, ``author``/``sort``, ``repo_slug``) are exposed
  as MCP arguments instead.
- Normalize the adapter's ``list[SearchResult]`` (``link`` / ``title`` /
  ``snippet``) into a stable ``SearchResponse`` JSON shape. We rename
  ``link`` -> ``url`` because that's the field name Cline's other MCP
  servers (and the broader MCP ecosystem) use.
- Open one OpenTelemetry span per call. The OTel SDK is already configured
  by ``open_webui``'s import side effects, so these spans land in the same
  collector -> Jaeger pipeline as the rest of the stack.

Importing this module triggers ``open_webui/__init__.py`` (Alembic migration
check + env validation, ~7s cold). That cost is paid once when Cline spawns
the server, not per tool call.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from opentelemetry import trace
from pydantic import BaseModel, Field

# This server only calls Open WebUI's stateless search adapters -- it never
# issues or validates a session JWT. Open WebUI's ``env`` module hard-exits
# (``raise SystemExit``) at import time when ``WEBUI_AUTH`` is enabled and
# ``WEBUI_SECRET_KEY`` is unset. The running server's secret is auto-generated
# by ``start.sh`` into a ``.webui_secret_key`` file that a fresh ``docker
# exec`` process does NOT inherit, so the import would otherwise fail. Disable
# auth for THIS subprocess so the import succeeds with zero extra config.
# ``setdefault`` so an explicit override from the MCP client's env still wins.
# Must run before the first ``open_webui`` import below.
os.environ.setdefault("WEBUI_AUTH", "False")

from open_webui.retrieval.web.arxiv import search_arxiv as _search_arxiv
from open_webui.retrieval.web.bitbucket import search_bitbucket as _search_bitbucket
from open_webui.retrieval.web.godot import search_godot as _search_godot
from open_webui.retrieval.web.huggingface import (
    search_huggingface as _search_huggingface,
)
from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.mdn import search_mdn as _search_mdn
from open_webui.retrieval.web.mslearn import search_mslearn as _search_mslearn
from open_webui.retrieval.web.utils import get_web_loader

log = logging.getLogger("cline_docs_mcp.tools")
_tracer = trace.get_tracer("cline_docs_mcp")

mcp = FastMCP("owui-docs")

# Default result count. Smaller than the chat pipeline's WEB_SEARCH_RESULT_COUNT
# (typically 15) because an editor agent works one tool call at a time and a
# tighter list keeps its context window lean; callers can raise it per call.
_DEFAULT_COUNT = 8


class SearchResultModel(BaseModel):
    """One search hit, normalized for MCP consumers."""

    title: str
    url: str
    snippet: Optional[str] = None


class SearchResponse(BaseModel):
    """Uniform response envelope for every ``search_*`` tool."""

    engine: str = Field(description="Which adapter produced the results.")
    query: str = Field(description="The query string that was searched.")
    count: int = Field(description="Number of results returned.")
    results: list[SearchResultModel]
    # Engine-specific scope echoes; populated only by the engines that use
    # them so the caller can see what narrowing was actually applied.
    version: Optional[str] = Field(
        default=None, description="Godot docs branch, if applicable."
    )
    product: Optional[str] = Field(
        default=None, description="Microsoft Learn product scope, if applicable."
    )
    category: Optional[str] = Field(
        default=None, description="arXiv category scope, if applicable."
    )
    author: Optional[str] = Field(
        default=None, description="Hugging Face author/org scope, if applicable."
    )


class FetchedPage(BaseModel):
    """A single fetched + extracted web page."""

    url: str
    title: Optional[str] = None
    markdown: str


def _to_models(results: list[SearchResult]) -> list[SearchResultModel]:
    """Map adapter ``SearchResult`` objects to MCP response models.

    Defensive against the ``title`` being ``None`` (the adapters type it as
    ``str | None``); MCP consumers expect a string, so coalesce to the URL.
    """
    models: list[SearchResultModel] = []
    for r in results:
        if not r.link:
            continue
        models.append(
            SearchResultModel(
                title=r.title or r.link,
                url=r.link,
                snippet=r.snippet,
            )
        )
    return models


@mcp.tool()
def search_godot(
    query: str,
    count: int = _DEFAULT_COUNT,
    version: Optional[str] = None,
) -> SearchResponse:
    """Search the official Godot Engine documentation.

    Backed by the Read the Docs search index for the ``godot`` project
    (docs.godotengine.org). Use for GDScript, the class reference, and
    engine tutorials.

    Args:
        query: Free-text query (e.g. "signal connect", "CharacterBody3D").
        count: Max results to return.
        version: Docs branch -- "stable" (current 4.x, default), "latest"
            (master/dev), or "3.6" (legacy 3.x). Loose aliases like
            "master" or "4" are normalized by the adapter.
    """
    with _tracer.start_as_current_span("mcp.search_godot") as span:
        span.set_attribute("query", query)
        span.set_attribute("count", count)
        if version:
            span.set_attribute("version", version)
        hits = _search_godot(query, count, search_filter=None, version=version)
        span.set_attribute("result_count", len(hits))
        return SearchResponse(
            engine="godot",
            query=query,
            count=len(hits),
            results=_to_models(hits),
            version=version or "stable",
        )


@mcp.tool()
def search_mdn(
    query: str,
    count: int = _DEFAULT_COUNT,
    language: Optional[str] = None,
) -> SearchResponse:
    """Search MDN Web Docs (developer.mozilla.org).

    The authoritative reference for the web platform: HTML, CSS,
    JavaScript, DOM, and Web APIs.

    Args:
        query: Free-text query (e.g. "fetch api", "flexbox align-items").
        count: Max results to return.
        language: ISO 639-1 code for MDN's locale (e.g. "en", "fr", "ja").
            Defaults to en-US. Unsupported locales fall back to en-US.
    """
    with _tracer.start_as_current_span("mcp.search_mdn") as span:
        span.set_attribute("query", query)
        span.set_attribute("count", count)
        search_filter = _language_filter(language)
        if language:
            span.set_attribute("language", language)
        hits = _search_mdn(query, count, search_filter=search_filter)
        span.set_attribute("result_count", len(hits))
        return SearchResponse(
            engine="mdn",
            query=query,
            count=len(hits),
            results=_to_models(hits),
        )


@mcp.tool()
def search_mslearn(
    query: str,
    count: int = _DEFAULT_COUNT,
    product: Optional[str] = None,
    language: Optional[str] = None,
) -> SearchResponse:
    """Search Microsoft Learn (learn.microsoft.com, formerly MSDN).

    The reference for Windows, .NET, C#, Azure, PowerShell, and related
    Microsoft technologies.

    Args:
        query: Free-text query (e.g. "WinUI 3 NavigationView", "EF Core
            migrations").
        count: Max results to return.
        product: Optional product slug to narrow results. Common values:
            "dotnet", "windows", "azure", "entra", "powershell",
            "aspnet-core", "ef-core".
        language: ISO 639-1 code mapped to Learn's locale (e.g. "en", "ja").
            Defaults to en-us.
    """
    with _tracer.start_as_current_span("mcp.search_mslearn") as span:
        span.set_attribute("query", query)
        span.set_attribute("count", count)
        if product:
            span.set_attribute("product", product)
        if language:
            span.set_attribute("language", language)
        search_filter = _language_filter(language)
        hits = _search_mslearn(
            query, count, search_filter=search_filter, product=product
        )
        span.set_attribute("result_count", len(hits))
        return SearchResponse(
            engine="mslearn",
            query=query,
            count=len(hits),
            results=_to_models(hits),
            product=product,
        )


@mcp.tool()
def search_arxiv(
    query: str,
    count: int = _DEFAULT_COUNT,
    category: Optional[str] = None,
) -> SearchResponse:
    """Search arXiv preprints (arxiv.org).

    Use for academic papers in CS, math, physics, and related fields.

    Args:
        query: Free-text query (e.g. "mixture of experts routing", "diffusion
            transformer").
        count: Max results to return.
        category: Optional arXiv category to scope results (e.g. "cs.LG",
            "cs.CL", "stat.ML").
    """
    with _tracer.start_as_current_span("mcp.search_arxiv") as span:
        span.set_attribute("query", query)
        span.set_attribute("count", count)
        if category:
            span.set_attribute("category", category)
        hits = _search_arxiv(query, count, search_filter=None, category=category)
        span.set_attribute("result_count", len(hits))
        return SearchResponse(
            engine="arxiv",
            query=query,
            count=len(hits),
            results=_to_models(hits),
            category=category,
        )


@mcp.tool()
def search_huggingface(
    query: str,
    count: int = _DEFAULT_COUNT,
    author: Optional[str] = None,
    sort: Optional[str] = None,
) -> SearchResponse:
    """Search the Hugging Face Hub for models (huggingface.co).

    Use to find open-weight model repositories and their model cards.

    Args:
        query: Free-text query (e.g. "qwen3 embedding", "llama 3.3 instruct").
        count: Max results to return.
        author: Optional org/user to scope results (e.g. "google",
            "meta-llama", "Qwen").
        sort: Optional sort order accepted by the Hub API (e.g. "downloads",
            "likes", "trending", "lastModified").
    """
    with _tracer.start_as_current_span("mcp.search_huggingface") as span:
        span.set_attribute("query", query)
        span.set_attribute("count", count)
        if author:
            span.set_attribute("author", author)
        if sort:
            span.set_attribute("sort", sort)
        hits = _search_huggingface(
            query, count, search_filter=None, author=author, sort=sort
        )
        span.set_attribute("result_count", len(hits))
        return SearchResponse(
            engine="huggingface",
            query=query,
            count=len(hits),
            results=_to_models(hits),
            author=author,
        )


@mcp.tool()
def search_bitbucket(
    query: str,
    count: int = _DEFAULT_COUNT,
    repo_slug: Optional[str] = None,
) -> SearchResponse:
    """Search the configured Bitbucket Cloud workspace.

    Searches code, repositories, and (when ``repo_slug`` is given) pull
    requests across the workspace named by the ``BITBUCKET_WORKSPACE``
    environment variable, authenticating with ``BITBUCKET_ACCESS_TOKEN``.
    Both are inherited from the open-webui container environment.

    Args:
        query: Free-text query (symbol name, file path fragment, PR keyword).
        count: Max results to return.
        repo_slug: Optional single repository slug to also search that repo's
            pull requests (Bitbucket Cloud has no workspace-wide PR API, so
            PR search is skipped when this is omitted).
    """
    with _tracer.start_as_current_span("mcp.search_bitbucket") as span:
        span.set_attribute("query", query)
        span.set_attribute("count", count)
        workspace = os.environ.get("BITBUCKET_WORKSPACE", "")
        token = os.environ.get("BITBUCKET_ACCESS_TOKEN", "")
        if not workspace or not token:
            raise ValueError(
                "Bitbucket search requires BITBUCKET_WORKSPACE and "
                "BITBUCKET_ACCESS_TOKEN to be set in the container environment."
            )
        if repo_slug:
            span.set_attribute("repo_slug", repo_slug)
        hits = _search_bitbucket(
            query, workspace, token, count, repo_slug=repo_slug
        )
        span.set_attribute("result_count", len(hits))
        return SearchResponse(
            engine="bitbucket",
            query=query,
            count=len(hits),
            results=_to_models(hits),
        )


@mcp.tool()
async def fetch_page(url: str) -> FetchedPage:
    """Fetch a single web page and return its extracted text as markdown.

    Routes through Open WebUI's page-fetch pipeline (``get_web_loader``),
    so it honors the configured loader engine (Playwright / Trafilatura /
    etc.), the SSRF URL allow-listing, and the shared page cache. Use this
    to read the full contents of a result URL returned by one of the
    ``search_*`` tools when the snippet isn't enough.

    Args:
        url: The page URL to fetch (typically a ``url`` from a search result).
    """
    with _tracer.start_as_current_span("mcp.fetch_page") as span:
        span.set_attribute("url", url)
        loader = get_web_loader(url)
        docs = await loader.aload()
        if not docs:
            raise ValueError(f"No content could be loaded from URL: {url}")
        doc = docs[0]
        title = doc.metadata.get("title") if doc.metadata else None
        source = (doc.metadata.get("source") if doc.metadata else None) or url
        span.set_attribute("content_length", len(doc.page_content or ""))
        return FetchedPage(
            url=source,
            title=title,
            markdown=doc.page_content or "",
        )


def _language_filter(language: Optional[str]):
    """Build a minimal ``WebSearchFilter`` carrying only ``language``.

    The MDN / MS Learn adapters read ``search_filter.language`` natively to
    pick a locale. Everything else on the filter is left empty so the
    generic post-filter is a no-op. Returns ``None`` when no language is
    requested so the adapter takes its default-locale path.
    """
    if not language:
        return None
    # Imported lazily so the module's import cost is dominated by the
    # adapter imports above, and so a future refactor of nl_filter doesn't
    # ripple into every tool import.
    from open_webui.retrieval.web.nl_filter import WebSearchFilter

    return WebSearchFilter(language=language)
