"""``{{STACK_INVENTORY}}`` system-prompt variable.

The chat surface model has *no idea* it lives inside a stack with parallel
specialty search routers, a Kagi lens config, an internal Bitbucket leg, an
always-on small task model, a vector store, etc. Those features fire on
keyword/bang triggers in the *user's* prompt rendered by
``open_webui.routers.retrieval`` -- the chat model never sees them. Worse,
adding a bang the model doesn't know about (e.g. ``!godot``) silently does
nothing if the user doesn't type it.

This module exposes a tiny one-line-per-route inventory the user can drop
into any system prompt via ``{{STACK_INVENTORY}}``, giving the model
*agency* to route deliberately ("for this question I'll suggest the user
narrow the search with !mslearn") instead of relying on the user to know
the bang vocabulary. The cost is ~200 tokens of preamble; the benefit is
materially fewer wasted turns on generic Kagi fanout for topics that have
a portal-specific route.

Design mirrors ``utils.weather`` / ``{{CURRENT_WEATHER}}``:

- Resolved lazily inside ``prompt_template`` -- never paid if the
  placeholder isn't in the template.
- Gated by ``ENABLE_STACK_INVENTORY_PROMPT_VAR`` (default ``True``) so
  admins can disable per-deployment without editing prompts.
- Fully overridable via ``STACK_INVENTORY_TEXT`` env -- handy for tuning
  the wording without a code change / rebuild.
- Reads the SAME env flags ``docker-compose.yml`` already uses to enable
  each router, so disabled routes are auto-omitted from the inventory.
  Setting ``ENABLE_HF_SEARCH=False`` for example drops the ``!hf`` line.

The inventory is intentionally terse. The model doesn't need to know HOW
each route works internally; it needs to know that ``!godot`` exists, what
it covers, and when to suggest it. Every byte beyond that bloats the chat
KV cache for zero accuracy gain.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def _flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def _build_inventory() -> str:
    """Construct the inventory text from current env state.

    Each section is omitted entirely if its feature flag is off. The result
    is a single string ready to substitute into the template; it ends WITHOUT
    a trailing newline so callers can place it inside paragraph flow.
    """
    lines: list[str] = []

    # Initial web-search fanout is run upstream of the chat completion
    # (the search toggle on the UI fires a fanout that injects loaded docs
    # into context BEFORE the chat model sees the user turn). The model
    # then has two native tools available -- when ``function_calling=native``
    # on the model preset -- to get more web information mid-turn:
    #   * search_web(query)          -> titles/links/snippets only (triage)
    #   * request_more_search(query) -> full fanout pipeline + loaded content,
    #                                   capped at WEB_SEARCH_MAX_DEEPENS per turn
    # We surface those in the inventory only when web search is enabled.
    if _flag('ENABLE_WEB_SEARCH', True):
        # Deepen cap. Defined directly here rather than imported from
        # config.py to avoid pulling the full config module into a prompt-
        # render hot path; the env read is what the ConfigVar resolves at
        # startup anyway, and a runtime DB override of this value would
        # already require a restart for the inventory cache below.
        max_deepens_raw = os.getenv('WEB_SEARCH_MAX_DEEPENS')
        try:
            max_deepens = int(max_deepens_raw) if max_deepens_raw else 2
        except ValueError:
            max_deepens = 2

        web_lines: list[str] = []
        # Specialty portal routes. Order matches the priority logic in
        # routers/retrieval.py: bangs win outright, keyword routing runs as
        # fallback, generic queries fan out across Kagi + matched specialty.
        if _flag('ENABLE_ARXIV_SEARCH', True):
            web_lines.append('  !arxiv / !papers / !preprint      — arXiv preprints (CS, math, physics)')
        if _flag('ENABLE_DOCS_ROUTING', True):
            if _flag('ENABLE_MDN_SEARCH', True):
                web_lines.append('  !mdn                              — MDN Web Docs (HTML/CSS/JS/Web APIs)')
            if _flag('ENABLE_MSLEARN_SEARCH', True):
                web_lines.append('  !mslearn / !winui / !dotnet / !azure — Microsoft Learn (Windows, .NET, Azure)')
            if _flag('ENABLE_GODOT_SEARCH', True):
                web_lines.append('  !godot / !gdscript                — Godot Engine docs (game dev, GDScript)')
        if _flag('ENABLE_HF_SEARCH', True):
            web_lines.append('  !hf / !huggingface / !models      — Hugging Face model hub (open weights)')
        if _flag('ENABLE_BITBUCKET_SEARCH', False):
            web_lines.append('  !bb / !bitbucket / !repo          — Internal Bitbucket workspace (code, repos, PRs)')

        if web_lines:
            lines.append('Web search (parallel fanout — Kagi + specialty portals):')
            lines.extend(web_lines)
            lines.append('  (no bang)                          — Kagi default; specialty routes auto-trigger on topic keywords')
            lines.append('  Narrow Kagi with `site:<host>` operators in the query when you know the canonical source.')
            lines.append('')

            # Iterative deepen via native function-call tools. Visible only
            # when the chat model is configured for native FC; the model
            # provider will simply ignore these lines if it doesn't expose
            # the tools (and the model can still suggest bangs to the user).
            if max_deepens > 0:
                lines.append('Mid-turn search tools (native function-calling):')
                lines.append('  search_web(query)               — returns titles/links/snippets only (triage cheap)')
                lines.append(
                    f'  request_more_search(query)      — full fanout (Kagi+specialty) + loaded page content, '
                    f'capped at {max_deepens} call(s)/turn'
                )
                lines.append(
                    '  Use request_more_search when the initial fanout missed the canonical source for a '
                    'specific symbol / version / CVE / proper noun. Issue a MORE SPECIFIC follow-up than '
                    'the original (use exact terms or `site:<host>`); a synonym of the same query dedups '
                    'to the same URLs and wastes the deepen budget.'
                )
                lines.append('')

    # Web loader. Mention only the path that's actually configured so the
    # model doesn't suggest features that aren't wired up.
    loader = (os.getenv('WEB_LOADER_ENGINE') or '').strip().lower()
    if loader == 'trafilatura':
        lines.append('Web loader: trafilatura (no JavaScript). Single-page apps may return empty;')
        lines.append('the loader auto-retries those URLs via Playwright when JS-shell content is detected.')
        lines.append('')
    elif loader == 'playwright':
        lines.append('Web loader: Playwright (full Chromium, executes JavaScript).')
        lines.append('')

    # gemma-3-1b compress pass. Tell the model the snippets it sees have
    # been compressed -- otherwise it's prone to second-guessing the
    # extract ("the source might say more...") or apologizing for
    # missing context that the user could easily click through to. The
    # citation metadata still points at the original URL, so it's safe
    # for the model to recommend the user open the source for full
    # context on borderline cases.
    if _flag('WEB_SEARCH_COMPRESS_ENABLED', True):
        compress_model = os.getenv('WEB_SEARCH_COMPRESS_MODEL', 'gemma-3-1b')
        lines.append(
            f'Web-search snippet content is compressed by `{compress_model}` '
            'after extraction: code blocks, identifiers, version numbers, and CVE ids '
            'are preserved verbatim; prose chrome and off-topic sections are removed.'
        )
        lines.append(
            'Treat snippets as faithful but tighter than the original page -- when a '
            "user's question needs context the snippet doesn't cover, suggest opening "
            'the source URL rather than guessing.'
        )
        lines.append('')

    # Knowledge base / RAG embedding model. Mentioned so the model knows it
    # CAN ask the user to ingest a doc once and search it later, rather than
    # repeatedly fanning out web search for the same content.
    # cline_docs_mcp tools surfaced via the streamable-HTTP MCP service.
    # When the chat model is configured with function-calling native AND
    # the open-webui admin "Tool servers" page has the cline-docs-mcp
    # URL registered, these become callable from inside the chat. They
    # complement the builtin ``request_more_search`` (which runs the
    # full Kagi+specialty fanout) by letting the model pick ONE portal
    # deliberately -- cheaper, narrower, and far less prone to drift
    # than a generic fanout when the answer is known to live in one
    # source (Godot docs / MDN / arXiv / HF Hub / internal Bitbucket).
    if _flag('ENABLE_CLINE_DOCS_MCP_INVENTORY_LINE', True):
        mcp_lines: list[str] = []
        if _flag('ENABLE_GODOT_SEARCH', True):
            mcp_lines.append('  mcp_search_godot(query, count, version)        — Godot Engine docs (single portal)')
        if _flag('ENABLE_MDN_SEARCH', True):
            mcp_lines.append('  mcp_search_mdn(query, count, language)         — MDN Web Docs (single portal)')
        if _flag('ENABLE_MSLEARN_SEARCH', True):
            mcp_lines.append('  mcp_search_mslearn(query, count, product, language) — Microsoft Learn (single portal)')
        if _flag('ENABLE_ARXIV_SEARCH', True):
            mcp_lines.append('  mcp_search_arxiv(query, count, category)       — arXiv preprints (single portal)')
        if _flag('ENABLE_HF_SEARCH', True):
            mcp_lines.append('  mcp_search_huggingface(query, count, author, sort) — Hugging Face Hub (single portal)')
        if _flag('ENABLE_BITBUCKET_SEARCH', False):
            mcp_lines.append('  mcp_search_bitbucket(query, count, repo_slug)  — Internal Bitbucket (single portal)')
        # mcp_fetch_page is always available regardless of the per-portal
        # toggles -- it's a generic URL loader, not portal-specific.
        mcp_lines.append('  mcp_fetch_page(url)                            — Load one URL via the same loader chain as web search')
        if mcp_lines:
            lines.append('MCP tools (single-portal calls, complement request_more_search):')
            lines.extend(mcp_lines)
            lines.append(
                '  Use these when you already know WHICH portal has the answer. The '
                'builtin request_more_search fans out across Kagi+specialty; the MCP '
                'variants stay portal-only, so they cost less context and avoid noisy '
                'Kagi results when the canonical source is known.'
            )
            lines.append('')

    rag_model = os.getenv('RAG_EMBEDDING_MODEL') or ''
    if rag_model:
        lines.append(f'Knowledge base: vector store via embedding model `{rag_model}`.')
        # Mention the AST splitter so the model knows code-bearing KB
        # files are chunked at definition boundaries (not character
        # boundaries). It can then trust function-level retrieval to
        # land on a clean signature + body instead of bracing the user
        # for "the chunk might be cut in half".
        if _flag('KB_CODE_AST_SPLIT_ENABLED', True):
            lines.append(
                'Source-code files ingested into the KB are split AST-aware via '
                'tree-sitter (one chunk per function/class/method + a preamble '
                'chunk for imports). Falls back to character chunking on unknown '
                'extensions or parse failures.'
            )
        lines.append('Suggest the user ingest a doc into the KB when they will reference it across multiple turns.')
        lines.append('')

    # Always-on task model. The model itself doesn't call this -- Open WebUI
    # does internally for query gen / title gen / autocomplete -- but knowing
    # it exists keeps the chat model from second-guessing the user's
    # phrasing ("did you mean ...") since Open WebUI already rewrote the
    # query upstream.
    task_model = os.getenv('TASK_MODEL') or ''
    if task_model:
        lines.append(f'Auxiliary task model (sub-second): `{task_model}` — used by Open WebUI internally')
        lines.append('for web-search query generation, title gen, follow-up suggestions, auto-memory extraction.')
        lines.append('')

    # Coaching line. Without this the model still wastes turns dumping
    # generic search queries when a portal-specific route would have
    # returned tighter results. Empirically the single biggest accuracy
    # delta of this whole preamble.
    if any(l.startswith('  !') for l in lines):
        lines.append(
            'Prefer specialty routes over generic Kagi when the topic clearly matches one '
            '(web platform → !mdn, .NET/Windows → !mslearn, game dev → !godot, open models → !hf, '
            'papers → !arxiv). Suggest the bang to the user when they ask about a topic that fits.'
        )

    return '\n'.join(lines).strip()


# Module-level cache. The inventory only changes on env-flag changes, which
# require a container restart anyway, so a process-lifetime cache is fine
# and avoids re-walking ``os.getenv`` on every prompt render.
_CACHED: str | None = None


def get_stack_inventory() -> str:
    """Return the rendered inventory string (cached for the process lifetime).

    Returns the empty string when the feature is disabled OR when no
    routes are enabled (so a deployment that turns everything off doesn't
    inject a useless ``Web search:`` header into every system prompt).
    """
    global _CACHED
    if not _flag('ENABLE_STACK_INVENTORY_PROMPT_VAR', True):
        return ''
    if _CACHED is not None:
        return _CACHED

    # Full override path: trust the operator to write something useful
    # rather than walking the env-flag matrix above.
    override = os.getenv('STACK_INVENTORY_TEXT')
    if override is not None:
        _CACHED = override.strip()
        return _CACHED

    _CACHED = _build_inventory()
    return _CACHED
