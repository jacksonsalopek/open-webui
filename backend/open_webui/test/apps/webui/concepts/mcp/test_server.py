"""Unit tests for concept-graph MCP FastMCP server wiring."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from open_webui.retrieval.concepts.mcp.context import CallerContext
from open_webui.retrieval.concepts.mcp.server import (
    AppState,
    build_arg_parser,
    mcp,
    resolve_caller_context,
)
from open_webui.retrieval.concepts.retrieve.router import RouterConfig


def test_server_registers_5_tools():
    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}
    assert names == {
        'find_concept',
        'where_used',
        'explain_region',
        'trace_neighborhood',
        'impact_analysis',
    }


def test_server_tool_wrapper_calls_core_handler():
    fake_store = object()
    fake_caller = CallerContext(
        user_id='test',
        accessible_artifact_paths=frozenset(),
        bypass_acl=True,
    )
    fake_result = {'status': 'ok', 'tool': 'find_concept'}

    state = AppState(store=fake_store, router_config=RouterConfig())
    ctx = MagicMock()
    ctx.request_context.lifespan_context = state
    ctx.request_context.request = None

    async def _invoke():
        with patch(
            'open_webui.retrieval.concepts.mcp.server.resolve_caller_context',
            return_value=fake_caller,
        ), patch(
            'open_webui.retrieval.concepts.mcp.server.core_find_concept',
            return_value=fake_result,
        ) as mock_core:
            from open_webui.retrieval.concepts.mcp.server import find_concept as wrapped_find_concept

            result = await wrapped_find_concept(name='test', kind=None, ctx=ctx)

        assert result == fake_result
        mock_core.assert_called_once_with(
            'test',
            None,
            store=fake_store,
            caller=fake_caller,
        )

    asyncio.run(_invoke())


def test_transport_cli():
    parser = build_arg_parser()
    stdio_args = parser.parse_args(['--transport', 'stdio'])
    http_args = parser.parse_args(['--transport', 'streamable-http'])
    assert stdio_args.transport == 'stdio'
    assert http_args.transport == 'streamable-http'


def test_resolve_caller_context_stdio_bypass():
    ctx = MagicMock()
    ctx.request_context.request = None
    caller = resolve_caller_context(ctx)
    assert caller.user_id == 'stdio'
    assert caller.bypass_acl is True
