"""Span + RuntimeEvent models for the Workflow Runtime Observability System.

All events flow through the EventBus.  Business code never touches these directly.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ── Span ────────────────────────────────────────────────────────────────────

class SpanStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class SpanType(StrEnum):
    WORKFLOW = "workflow"
    NODE = "node"
    LLM = "llm"
    TOOL = "tool"


@dataclass
class Span:
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_span_id: str | None = None
    trace_id: str = ""
    span_type: SpanType = SpanType.NODE
    name: str = ""
    status: SpanStatus = SpanStatus.RUNNING
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "span_type": self.span_type,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round((self.ended_at - self.started_at) * 1000, 1) if self.ended_at else None,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }

    def close(self, status: SpanStatus, output: dict | None = None, error: str | None = None) -> None:
        self.status = status
        self.ended_at = time.monotonic()
        if output is not None:
            self.output = output
        if error:
            self.metadata["error"] = error


# ── RuntimeEvent ─────────────────────────────────────────────────────────────

class EventType(StrEnum):
    WORKFLOW_START = "workflow.start"
    WORKFLOW_END = "workflow.end"
    WORKFLOW_ERROR = "workflow.error"
    NODE_START = "node.start"
    NODE_PROGRESS = "node.progress"
    NODE_SUCCESS = "node.success"
    NODE_ERROR = "node.error"
    NODE_SKIPPED = "node.skipped"
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    STATE_UPDATE = "state.update"
    PARAM_STEP_START = "param.step.started"
    PARAM_STEP_COMPLETE = "param.step.completed"
    PARAM_STEP_ERROR = "param.step.error"


# Backward-compatible aliases for frontend eventRouter
EVENT_ALIASES: dict[EventType, str] = {
    EventType.NODE_START: "node.started",
    EventType.NODE_PROGRESS: "node.progress",
    EventType.NODE_SUCCESS: "node.completed",
    EventType.NODE_ERROR: "node.failed",
    EventType.NODE_SKIPPED: "node.skipped",
    EventType.WORKFLOW_START: "task.running",
    EventType.WORKFLOW_END: "task.completed",
    EventType.WORKFLOW_ERROR: "task.failed",
    EventType.LLM_REQUEST: "llm.request",
    EventType.LLM_RESPONSE: "llm.response",
    EventType.LLM_ERROR: "llm.error",
    EventType.PARAM_STEP_START: "param.step.started",
    EventType.PARAM_STEP_COMPLETE: "param.step.completed",
    EventType.PARAM_STEP_ERROR: "param.step.error",
}


def event_alias(evt: EventType) -> str:
    return EVENT_ALIASES.get(evt, evt.value)


@dataclass
class RuntimeEvent:
    seq: int
    event_type: EventType
    run_id: str
    span: Span
    data: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: __import__("datetime").datetime.now().isoformat())

    def to_sse(self) -> dict[str, Any]:
        """Format for SSE delivery.  Backward compatible with frontend eventRouter."""
        return {
            "seq": self.seq,
            "event_type": event_alias(self.event_type),
            "data": {
                **self.data,
                "span_id": self.span.span_id,
                "parent_span_id": self.span.parent_span_id,
                "trace_id": self.span.trace_id,
                "span_type": self.span.span_type,
            },
        }

    def to_db(self) -> dict[str, Any]:
        """Format for pipeline_events table persistence."""
        return {
            "seq": self.seq,
            "event_type": self.event_type.value,
            "data": self.to_sse()["data"],
            "span": self.span.to_dict(),
        }
