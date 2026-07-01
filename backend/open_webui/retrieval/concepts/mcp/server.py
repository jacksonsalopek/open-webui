"""Concept-graph MCP server — FastMCP transport over core tool handlers."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

from open_webui.retrieval.concepts.retrieve.router import RouterConfig
from open_webui.retrieval.concepts.store.protocol import GraphStore

from .context import CallerContext
from .tools import (
    explain_region as core_explain_region,
    find_concept as core_find_concept,
    impact_analysis as core_impact_analysis,
    trace_neighborhood as core_trace_neighborhood,
    where_used as core_where_used,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AppState:
    store: GraphStore | None
    router_config: RouterConfig


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Load pre-built store on startup; never run a full embedder rebuild here."""
    from open_webui.retrieval.concepts.integration.store_factory import (
        create_graph_store_from_config,
    )

    log.info('concept_graph_mcp: opening GraphStore (read-only attach)')
    try:
        store = create_graph_store_from_config()
    except Exception:
        log.exception('concept_graph_mcp: failed to open GraphStore')
        store = None
    yield AppState(store=store, router_config=RouterConfig())
    log.info('concept_graph_mcp: shutdown')


mcp = FastMCP(
    'concept-graph',
    instructions=(
        'Concept-neighbor exploration tools for a codebase knowledge graph. '
        'Use find_concept to orient on a term; where_used for provenance; '
        'explain_region for local structure; trace_neighborhood for query-driven walks; '
        'impact_analysis for paths between concepts.'
    ),
    lifespan=lifespan,
)


def _app_state(ctx: Context) -> AppState:
    return ctx.request_context.lifespan_context


def resolve_caller_context(ctx: Context) -> CallerContext:
    """Resolve ACL context from MCP session metadata or stdio defaults."""
    request = ctx.request_context.request
    headers: dict[str, str] = {}
    if request is not None and hasattr(request, 'headers'):
        raw_headers = request.headers
        headers = {
            str(key).lower(): str(value)
            for key, value in raw_headers.items()
        }

    user_id = headers.get('x-openwebui-user-id', 'stdio')
    role = headers.get('x-openwebui-user-role', '')
    acl_mode = os.environ.get('CONCEPT_GRAPH_MCP_ACL_MODE', 'trust_os')
    bypass = (
        role == 'admin'
        or acl_mode == 'trust_os'
        or user_id == 'stdio'
    )

    accessible_raw = headers.get('x-openwebui-accessible-artifacts', '')
    accessible_paths = frozenset(
        path.strip()
        for path in accessible_raw.split(',')
        if path.strip()
    )

    if bypass and not accessible_paths:
        log.debug(
            'concept_graph_mcp: ACL bypass for user_id=%s role=%s acl_mode=%s',
            user_id,
            role,
            acl_mode,
        )

    return CallerContext(
        user_id=str(user_id),
        accessible_artifact_paths=accessible_paths,
        bypass_acl=bypass,
    )


def _run_tool(fn, *args, **kwargs) -> dict:
    return fn(*args, **kwargs)


@mcp.tool()
async def find_concept(name: str, kind: str | None = None, ctx: Context = None) -> dict:
    state = _app_state(ctx)
    caller = resolve_caller_context(ctx)
    return await asyncio.to_thread(
        _run_tool,
        core_find_concept,
        name,
        kind,
        store=state.store,
        caller=caller,
    )


@mcp.tool()
async def where_used(concept_name: str, ctx: Context = None) -> dict:
    state = _app_state(ctx)
    caller = resolve_caller_context(ctx)
    return await asyncio.to_thread(
        _run_tool,
        core_where_used,
        concept_name,
        store=state.store,
        caller=caller,
    )


@mcp.tool()
async def explain_region(
    concept_name: str,
    radius: int = 2,
    ctx: Context = None,
) -> dict:
    state = _app_state(ctx)
    caller = resolve_caller_context(ctx)
    return await asyncio.to_thread(
        _run_tool,
        core_explain_region,
        concept_name,
        radius,
        store=state.store,
        caller=caller,
    )


@mcp.tool()
async def trace_neighborhood(query: str, ctx: Context = None) -> dict:
    state = _app_state(ctx)
    caller = resolve_caller_context(ctx)
    return await asyncio.to_thread(
        _run_tool,
        core_trace_neighborhood,
        query,
        store=state.store,
        caller=caller,
        router_config=state.router_config,
    )


@mcp.tool()
async def impact_analysis(
    concept_a: str,
    concept_b: str,
    ctx: Context = None,
) -> dict:
    state = _app_state(ctx)
    caller = resolve_caller_context(ctx)
    return await asyncio.to_thread(
        _run_tool,
        core_impact_analysis,
        concept_a,
        concept_b,
        store=state.store,
        caller=caller,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='concept-graph-mcp')
    parser.add_argument(
        '--transport',
        choices=('stdio', 'streamable-http'),
        default='stdio',
        help='stdio for IDE agents; streamable-http for Open WebUI / remote',
    )
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8766)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.transport == 'streamable-http':
        os.environ.setdefault('FASTMCP_HOST', args.host)
        os.environ.setdefault('FASTMCP_PORT', str(args.port))
        mcp.run(transport='streamable-http')
    else:
        mcp.run(transport='stdio')


if __name__ == '__main__':
    main()
