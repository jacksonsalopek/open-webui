"""Stdio entry point for the Open WebUI docs MCP server.

Run with::

    python -m cline_docs_mcp

Cline (or any MCP client) spawns this as a subprocess and speaks JSON-RPC
over stdio. The server instance and its registered tools live in
``cline_docs_mcp.tools``.

stdout hygiene
--------------
On an stdio MCP server, **stdout is the JSON-RPC channel** -- any stray write
to it corrupts the protocol. Open WebUI's ``env`` module reconfigures the
root logger to stream to ``sys.stdout`` (with ``force=True``) and emits a few
INFO lines at import time. So we:

1. Redirect ``sys.stdout`` to ``sys.stderr`` for the duration of the
   ``open_webui`` import, so the import-time banner/log lines never land on
   the JSON-RPC channel. (Because ``logging.basicConfig`` captures whatever
   ``sys.stdout`` points at when it runs, the root handler ends up bound to
   stderr.)
2. Defensively re-point any remaining stdout-bound log handlers at stderr
   after the import, in case ``open_webui`` was already imported and its
   handler is still attached to the real stdout.
"""

import logging
import sys


def _route_logging_to_stderr() -> None:
    """Move any root log handler still writing to stdout over to stderr."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "stream", None) is sys.stdout:
            handler.setStream(sys.stderr)


def main() -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from cline_docs_mcp.tools import mcp
    finally:
        sys.stdout = real_stdout

    _route_logging_to_stderr()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
