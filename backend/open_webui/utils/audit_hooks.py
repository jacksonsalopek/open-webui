"""PEP 578 audit-hook instrumentation for the backend.

Stage 6 of the code-safety pipeline (see
``docs/CODE_SAFETY_PIPELINE.md``). CPython has shipped
:func:`sys.addaudithook` since 3.8: it fires a callback every time the
runtime is about to do a security-relevant thing (open a file, spawn a
subprocess, connect a socket, ``compile`` / ``exec`` / ``eval``,
import a module by name, ...). Each event carries the call's arguments.

This module installs a single hook that filters down to the canonical
PEP 578 events we actually care about and emits a structured log line
for each match. The hook is the "observe what generated code touches"
layer: it sits **underneath** the pyodide sandbox, the
``CODE_INTERPRETER_BLOCKED_MODULES`` shim, and the (planned) egress
proxy, and only buys defense-in-depth -- it can never be the primary
boundary because anything that runs in our Python process is already
past the boundary that mattered.

Design notes
------------

- **Default-off** (``AUDIT_HOOKS_ENABLED`` is ``False``). The hot path
  fires on every ``open`` / ``socket.connect`` / ``import``, so until
  the request middleware is wired through (TODO below) the log lines
  would be context-free noise. Flip the flag explicitly when you're
  ready to debug a specific concern.
- **One hook, frozenset filter.** PEP 578 hooks cannot be uninstalled;
  every hook the process registers runs on every audited event for
  the lifetime of the interpreter. That makes the event-name check the
  single most important thing about this module -- the very first
  statement of the callback is a ``set membership`` test against a
  module-level :class:`frozenset`, and only matched events do any
  further work.
- **Fail-silent.** A raised exception from inside an audit hook is
  swallowed by CPython on most events but escapes on a few (``import``
  in particular); we wrap the hook body in a bare ``try / except`` so
  a bug in the hook never breaks user code. The optional blocking path
  (``AUDIT_HOOKS_BLOCK_BLOCKED_IMPORTS``) deliberately *does* raise
  ``ImportError`` -- that's the entire point of opting in to blocking.
- **Context via contextvars.** Per-request ``chat_id`` / ``user_id`` /
  ``tool_name`` are exposed as :class:`contextvars.ContextVar` slots
  with a :func:`set_audit_context` helper. They are NOT yet wired in
  from the FastAPI middleware -- that's the next step (see TODO).
- **Secret redaction is best-effort.** We strip a single very common
  ``key=value`` shape (``token=...`` / ``api_key=...`` / etc.) from
  each rendered argument and truncate every string to 200 chars. This
  is not a substitute for treating audit logs as sensitive output.
"""

from __future__ import annotations

import contextvars
import logging
import re
import sys
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)


# ── Public contextvars ──────────────────────────────────────────────────────
#
# These slots are populated per-request by the FastAPI middleware so the
# audit log can attribute an event to the chat / user / tool that caused
# it. They default to ``None`` so the hook works (just without
# attribution) when nobody has set them -- e.g. for background tasks,
# startup-time imports, or the smoke-test ``python -c`` invocation.
#
# TODO: wire these from ``open_webui/utils/middleware.py``. The natural
# integration point is the request-scoped middleware that already
# resolves the authenticated user and the chat the request belongs to;
# call ``set_audit_context(chat_id=..., user_id=..., tool_name=...)``
# at the top of the dispatch and reset the token on the way out.

_chat_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'audit_hooks.chat_id', default=None,
)
_user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'audit_hooks.user_id', default=None,
)
_tool_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'audit_hooks.tool_name', default=None,
)


def set_audit_context(
    *,
    chat_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> tuple[contextvars.Token, contextvars.Token, contextvars.Token]:
    """Populate the per-request audit-attribution contextvars.

    Returns the three :class:`~contextvars.Token` values from the
    underlying ``.set()`` calls, suitable for passing to ``.reset()``
    on the way out of the request. Keyword-only so callers don't end
    up swapping ``chat_id`` and ``user_id`` positionally.
    """
    return (
        _chat_id_var.set(chat_id),
        _user_id_var.set(user_id),
        _tool_name_var.set(tool_name),
    )


# ── Event filter ────────────────────────────────────────────────────────────
#
# The canonical PEP 578 event names we want to observe. Verified
# against CPython's ``Doc/library/audit_events.rst``; the runtime
# emits these as literal string event names from the C layer. Keep
# this as a frozenset so the membership test in the hot path is O(1)
# without any per-call allocation.

_WATCHED_EVENTS: frozenset[str] = frozenset({
    'open',
    'subprocess.Popen',
    'os.system',
    'os.exec',
    'os.posix_spawn',
    'os.fork',
    'os.spawn',
    'socket.connect',
    'socket.bind',
    'urllib.Request',
    'compile',
    'exec',
    'eval',
})

# ``import`` is special: CPython fires it on *every* import (including
# every transitive import the interpreter does during normal request
# handling), which is thousands per second on a warm process. We keep
# it out of ``_WATCHED_EVENTS`` and dispatch it through a dedicated
# branch in the hook that only logs / blocks when the module being
# imported is on the blocked list.

_IMPORT_EVENT = 'import'


# Used to scrub obvious ``token=...`` / ``key=...`` style values out of
# rendered argument lists before they hit the log. Greedy up to the
# next whitespace -- audit args are short single-line repr-style
# strings, so this is sufficient without trying to parse a full URL.
_SECRET_PATTERN = re.compile(r'(?i)\b(token|key|secret|password)=\S+')
_MAX_REPR_LEN = 200


def _sanitize_arg(value: Any) -> str:
    """Render one audit-event arg as a short, scrubbed string."""
    try:
        rendered = repr(value)
    except Exception:
        # ``repr`` raised -- this is rare but happens for objects with
        # broken ``__repr__``. Don't let it kill the hook.
        rendered = f'<unreprable {type(value).__name__}>'
    rendered = _SECRET_PATTERN.sub(lambda m: f'{m.group(1)}=<redacted>', rendered)
    if len(rendered) > _MAX_REPR_LEN:
        rendered = rendered[:_MAX_REPR_LEN] + '...<truncated>'
    return rendered


def _sanitize_args(args: tuple) -> list[str]:
    """Map :func:`_sanitize_arg` across the raw audit-event arg tuple."""
    if not args:
        return []
    return [_sanitize_arg(a) for a in args]


# ── Hook installation ───────────────────────────────────────────────────────


_INSTALLED = False


def install_audit_hook(
    blocked_modules: Optional[Iterable[str]] = None,
    *,
    block_blocked_imports: bool = False,
) -> bool:
    """Install the audit hook on the process.

    Idempotent: subsequent calls are no-ops because :func:`sys.addaudithook`
    cannot uninstall and we don't want N duplicate hooks. Returns
    ``True`` if a hook was installed by this call, ``False`` if one
    was already installed or installation failed.

    Parameters
    ----------
    blocked_modules:
        Iterable of module names (as they appear to ``import``) that
        we want to either log or actively block. Compared by exact
        match; pass the leaf name a user would actually type
        (``'os'``, ``'subprocess'``, ``'socket'`` -- not
        ``'os.path'``). ``None`` / empty iterable disables import
        logging entirely.
    block_blocked_imports:
        When ``True``, raise :class:`ImportError` for any import of a
        name in ``blocked_modules``. When ``False``, only log the
        attempted import. Default ``False`` so flipping
        ``AUDIT_HOOKS_ENABLED`` on doesn't immediately start breaking
        user code -- block mode is opt-in.
    """
    global _INSTALLED
    if _INSTALLED:
        log.debug('audit_hooks: already installed; ignoring install_audit_hook call')
        return False

    blocked_set: frozenset[str] = frozenset(blocked_modules or ())

    def _hook(event: str, args: tuple) -> None:
        try:
            if event in _WATCHED_EVENTS:
                log.info(
                    'audit_hook event=%s chat_id=%s user_id=%s tool=%s args=%s',
                    event,
                    _chat_id_var.get(),
                    _user_id_var.get(),
                    _tool_name_var.get(),
                    _sanitize_args(args),
                )
                return

            if event == _IMPORT_EVENT and blocked_set:
                # ``import`` audit args are
                # ``(module, filename, sys.path, sys.meta_path, sys.path_hooks)``
                # per PEP 578. We only need the first.
                module_name = args[0] if args else ''
                if not isinstance(module_name, str):
                    return
                # Match both ``import os`` and ``from os.path import ...``
                # by checking the leading dotted component too.
                top_level = module_name.split('.', 1)[0]
                if module_name in blocked_set or top_level in blocked_set:
                    log.warning(
                        'audit_hook event=import blocked_module=%s '
                        'chat_id=%s user_id=%s tool=%s block=%s',
                        module_name,
                        _chat_id_var.get(),
                        _user_id_var.get(),
                        _tool_name_var.get(),
                        block_blocked_imports,
                    )
                    if block_blocked_imports:
                        # Raised exceptions from audit hooks DO escape
                        # the ``import`` event (PEP 578) -- this is the
                        # one place a hook can actually enforce policy.
                        raise ImportError(
                            f'import of {module_name!r} blocked by audit hook'
                        )
        except ImportError:
            # Re-raise the deliberate block; everything else is
            # swallowed so a hook bug can never break user code.
            raise
        except Exception as exc:  # noqa: BLE001 -- defense-in-depth
            # Fall back to ``log.debug`` so a misbehaving hook doesn't
            # become its own log spam vector.
            log.debug('audit_hooks: hook raised on event=%s: %s', event, exc)

    try:
        sys.addaudithook(_hook)
    except Exception as exc:  # pragma: no cover -- only fires if CPython rejects the hook
        log.warning('audit_hooks: sys.addaudithook failed: %s', exc)
        return False

    _INSTALLED = True
    log.info(
        'audit_hooks: installed (watched=%d blocked_modules=%d block=%s)',
        len(_WATCHED_EVENTS),
        len(blocked_set),
        block_blocked_imports,
    )
    return True


__all__ = ['install_audit_hook', 'set_audit_context']
