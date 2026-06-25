"""Cline-facing MCP server exposing Open WebUI's doc-portal adapters.

This package wraps the subject-specific search adapters in
``open_webui.retrieval.web.*`` (Godot, MDN, Microsoft Learn, arXiv, Hugging
Face, Bitbucket) plus Open WebUI's page-fetch pipeline as Model Context
Protocol tools, so editor agents like Cline can call them directly.

It is intentionally thin: every tool delegates straight to the same
``search_*`` function the Open WebUI web-search pipeline uses, so there is a
single source of truth for query construction, response shaping, and the
upstream API quirks each adapter already handles. The NL filter / Kagi
fanout / RAG-chunking layers are deliberately *not* wrapped -- Cline does
its own context management and just wants ``[{title, url, snippet}]`` plus a
way to fetch a full page.

Run as ``python -m cline_docs_mcp`` (stdio transport). See
``cline_docs_mcp.__main__``.
"""
