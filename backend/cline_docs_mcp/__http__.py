"""Streamable-HTTP entry point for the Open WebUI docs MCP server.

Run with::

    python -m cline_docs_mcp.__http__

Open WebUI's built-in ``MCPClient`` connects to MCP servers via
``streamablehttp_client`` (NOT stdio), so this entry point exposes the
same ``mcp`` instance as ``__main__.py`` over FastMCP's streamable-HTTP
transport. The dockerized ``cline-docs-mcp`` service in
``docker-compose.yml`` uses this entry point; Cline / Continue / other
stdio MCP clients can still use ``__main__.py`` directly.

stdout / log hygiene
--------------------
Unlike the stdio variant, the HTTP transport doesn't use stdout as a
protocol channel -- it serves over a TCP socket -- so stray prints to
stdout are merely noise rather than corruption. We still re-route the
root logger to stderr for consistency with ``__main__.py`` (and to keep
``docker logs cline-docs-mcp`` showing structured stderr-side log
records instead of having log lines interleave with FastMCP's own
startup output on stdout).

Configuration
-------------
Reads two env vars:

- ``MCP_HTTP_HOST`` -- bind address. Default ``0.0.0.0`` so the docker
  service is reachable from peer containers (open-webui) on the
  internal network.
- ``MCP_HTTP_PORT`` -- bind port. Default ``8765``. Match the value
  published in ``docker-compose.yml``'s ``cline-docs-mcp`` service and
  the URL configured in Open WebUI's "Tool servers" admin page.

FastMCP version note
--------------------
``mcp.server.fastmcp.FastMCP.run(transport='streamable-http', ...)``
landed in mcp >= 1.10. ``requirements.txt`` pins ``mcp==1.26.0`` which
is well past that, so this entry point should work out of the box.
If a future downgrade breaks it, the symptom will be a ValueError at
startup ("Unknown transport: streamable-http") -- bump the mcp pin to
the latest 1.x to fix.
"""

import logging
import os
import sys


def _route_logging_to_stderr() -> None:
    """Move any root log handler still writing to stdout over to stderr.

    open_webui's import-time ``logging.basicConfig`` binds the root
    handler to whatever ``sys.stdout`` points at when it runs. We don't
    care about stdout-corruption here (HTTP transport doesn't use it),
    but keeping log output on stderr matches the stdio variant and
    makes ``docker logs cline-docs-mcp`` show a clean stream of log
    records without FastMCP's framework lines interleaved.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, 'stream', None) is sys.stdout:
            handler.setStream(sys.stderr)


def main() -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from cline_docs_mcp.tools import mcp
    finally:
        sys.stdout = real_stdout

    _route_logging_to_stderr()

    host = os.environ.get('MCP_HTTP_HOST', '0.0.0.0')
    try:
        port = int(os.environ.get('MCP_HTTP_PORT', '8765'))
    except ValueError:
        port = 8765

    # FastMCP's streamable-HTTP transport supports per-call settings
    # via the ``settings`` attribute (host/port) when the transport
    # name is passed. We set them on the instance first so they apply
    # both to the streamable-http transport and to the legacy SSE
    # transport if a caller switches at runtime.
    mcp.settings.host = host
    mcp.settings.port = port

    logging.getLogger('cline_docs_mcp.http').info(
        'starting cline_docs_mcp streamable-http server on %s:%d', host, port,
    )
    mcp.run(transport='streamable-http')


if __name__ == '__main__':
    main()
