"""Caller ACL context threaded into MCP tool handlers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CallerContext:
    """Session-scoped caller identity and artifact visibility."""

    user_id: str
    accessible_artifact_paths: frozenset[str]
    bypass_acl: bool = False
