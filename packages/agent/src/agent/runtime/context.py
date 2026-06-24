"""RuntimeContext via ContextVar — implicit propagation across LLM / tool calls.

Business code reads `get_current_span_id()` to know its parent span.
All other runtime fields (run_id, trace_id, manager) flow automatically.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.runtime.manager import RuntimeManager

_current_context: ContextVar["RuntimeContext | None"] = ContextVar("runtime_context", default=None)


class RuntimeContext:
    __slots__ = ("run_id", "trace_id", "current_span_id", "current_node_id", "manager")

    def __init__(self, run_id: str, manager: "RuntimeManager") -> None:
        self.run_id = run_id
        self.trace_id = run_id  # trace_id equals run_id for single-workflow runs
        self.current_span_id = run_id
        self.current_node_id = None
        self.manager = manager


def set_context(ctx: RuntimeContext) -> RuntimeContext:
    """Push a RuntimeContext onto the current async task."""
    _current_context.set(ctx)
    return ctx


def get_context() -> RuntimeContext | None:
    return _current_context.get(None)


def require_context() -> RuntimeContext:
    ctx = _current_context.get(None)
    if ctx is None:
        raise RuntimeError("No RuntimeContext set — ensure @traced_node or RuntimeManager is active")
    return ctx


def get_current_span_id() -> str | None:
    ctx = _current_context.get(None)
    return ctx.current_span_id if ctx else None


def set_current_span(span_id: str) -> None:
    ctx = _current_context.get(None)
    if ctx:
        ctx.current_span_id = span_id
