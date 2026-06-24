"""RuntimeManager — central orchestrator for workflow run lifecycle.

Owns EventBus, Span tree, and SSE fan-out.  One instance per app.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from agent.runtime.bus import EventBus
from agent.runtime.context import RuntimeContext, set_context
from agent.runtime.events import EventType, RuntimeEvent, Span, SpanStatus, SpanType

logger = logging.getLogger(__name__)


@dataclass
class RuntimeRun:
    run_id: str
    operator_name: str
    status: str = "running"
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    spans: dict[str, Span] = field(default_factory=dict)
    events: list[RuntimeEvent] = field(default_factory=list)
    seq: int = 0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "operator_name": self.operator_name,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "spans": {k: v.to_dict() for k, v in self.spans.items()},
        }


class RuntimeManager:
    def __init__(self, event_bus: EventBus | None = None) -> None:
        self.bus = event_bus or EventBus()
        self._runs: dict[str, RuntimeRun] = {}

    # ── Run lifecycle ────────────────────────────────────────────────────

    def create_run(self, operator_name: str) -> RuntimeRun:
        run_id = uuid.uuid4().hex[:12]
        run = RuntimeRun(run_id=run_id, operator_name=operator_name)

        # Root span for the entire workflow
        root = Span(
            span_id=run_id,
            trace_id=run_id,
            span_type=SpanType.WORKFLOW,
            name=operator_name,
        )
        run.spans[run_id] = root

        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> RuntimeRun | None:
        return self._runs.get(run_id)

    def complete_run(self, run_id: str, *, error: str | None = None) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        run.completed_at = time.monotonic()
        if error:
            run.status = "failed"
        else:
            run.status = "completed"

    # ── Span management ──────────────────────────────────────────────────

    def open_span(
        self,
        run_id: str,
        parent_span_id: str | None,
        span_type: SpanType,
        name: str,
        input: dict | None = None,
    ) -> Span:
        span = Span(
            span_id=uuid.uuid4().hex[:12],
            parent_span_id=parent_span_id,
            trace_id=run_id,
            span_type=span_type,
            name=name,
            input=input,
        )
        run = self._runs.get(run_id)
        if run:
            run.spans[span.span_id] = span
        return span

    def close_span(self, run_id: str, span: Span, status: SpanStatus, output: dict | None = None, error: str | None = None) -> None:
        span.close(status, output, error)

    # ── Event emission ───────────────────────────────────────────────────

    def emit(
        self,
        event_type: EventType,
        run_id: str,
        span: Span,
        data: dict | None = None,
    ) -> RuntimeEvent:
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        seq = run.seq
        run.seq += 1
        evt = RuntimeEvent(seq=seq, event_type=event_type, run_id=run_id, span=span, data=data or {})
        run.events.append(evt)
        # Push to SSE subscribers
        self.bus.publish(run_id, evt.to_sse())
        return evt

    # ── SSE subscriber helper ────────────────────────────────────────────

    def subscribe(self, run_id: str) -> asyncio.Queue:
        return self.bus.subscribe(run_id)

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self.bus.unsubscribe(run_id, q)

    # ── RuntimeContext ───────────────────────────────────────────────────

    def enter_context(self, run_id: str) -> RuntimeContext:
        ctx = RuntimeContext(run_id, self)
        set_context(ctx)
        return ctx
