"""LLM and MCP tracers — create child spans under the current node span.

All tracers read from RuntimeContext (ContextVar).  No business code involvement.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agent.runtime.context import get_context
from agent.runtime.events import EventType, SpanType

logger = logging.getLogger(__name__)
MAX_PREVIEW = 600


class LLMTracer(BaseCallbackHandler):
    """LangChain callback that records LLM calls as child spans of the current node."""

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}  # run_id → span_id

    def _resolve_node_id(self, ctx) -> str | None:
        node_id = ctx.current_node_id
        if node_id:
            return node_id
        run = ctx.manager.get_run(ctx.run_id)
        if not run:
            return None
        parent_span_id = ctx.current_span_id
        for _ in range(10):
            parent_span = run.spans.get(parent_span_id)
            if not parent_span:
                break
            if parent_span.span_type == SpanType.NODE:
                return parent_span.name
            parent_span_id = parent_span.parent_span_id
        return None

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        ctx = get_context()
        if ctx is None:
            return

        request_id = str(kwargs.get("run_id", ""))
        model = self._model_name(serialized)
        node_id = self._resolve_node_id(ctx) or ""

        span = ctx.manager.open_span(
            run_id=ctx.run_id,
            parent_span_id=ctx.current_span_id,
            span_type="llm",
            name=f"llm:{model}",
            input={"prompt_length": sum(len(p) for p in prompts), "prompt_count": len(prompts)},
        )
        span.metadata["model"] = model
        self._pending[request_id] = span.span_id

        for i, prompt in enumerate(prompts):
            preview = prompt[:MAX_PREVIEW]
            if len(prompt) > MAX_PREVIEW:
                preview += f"\n… ({len(prompt)} 字符)"
            ctx.manager.emit(EventType.LLM_REQUEST, ctx.run_id, span, {
                "agent_id": "doc",
                "node_id": node_id,
                "request_id": request_id,
                "model": model,
                "prompt_preview": preview,
                "prompt_full": prompt,
                "prompt_length": len(prompt),
            })

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        ctx = get_context()
        if ctx is None:
            return

        request_id = str(kwargs.get("run_id", ""))
        span_id = self._pending.pop(request_id, None)
        span = None
        if span_id:
            run = ctx.manager.get_run(ctx.run_id)
            if run:
                span = run.spans.get(span_id)

        node_id = self._resolve_node_id(ctx) or ""

        for gen_list in response.generations:
            for g in gen_list:
                text = g.text if hasattr(g, "text") else str(g.message.content if hasattr(g, "message") else str(g))
                preview = text[:MAX_PREVIEW]
                if len(text) > MAX_PREVIEW:
                    preview += f"\n… ({len(text)} 字符)"
                tok = self._token_usage(response)
                used_span = span or Span(span_id=span_id or "", trace_id=ctx.run_id, span_type="llm", name="llm")
                ctx.manager.emit(EventType.LLM_RESPONSE, ctx.run_id, used_span, {
                    "agent_id": "doc",
                    "node_id": node_id,
                    "request_id": request_id,
                    "response_preview": preview,
                    "response_full": text,
                    "response_length": len(text),
                    "token_usage": tok,
                })

        if span:
            ctx.manager.close_span(ctx.run_id, span, "success",
                output={"response_length": sum(len(g.text) if hasattr(g, "text") else 0 for gen_list in response.generations for g in gen_list)})

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        ctx = get_context()
        if ctx is None:
            return
        request_id = str(kwargs.get("run_id", ""))
        span_id = self._pending.pop(request_id, None)
        if span_id:
            run = ctx.manager.get_run(ctx.run_id)
            if run:
                span = run.spans.get(span_id)
                if span:
                    ctx.manager.close_span(ctx.run_id, span, "error", error=str(error))

    @staticmethod
    def _model_name(serialized: dict) -> str:
        name = serialized.get("name", "")
        if not name and "id" in serialized:
            ids = serialized["id"]
            name = ids[-1] if isinstance(ids, list) else str(ids)
        return name or "unknown"

    @staticmethod
    def _token_usage(response: LLMResult) -> dict | None:
        try:
            u = response.llm_output.get("token_usage", {})
            if u:
                return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
        except (AttributeError, KeyError, TypeError):
            pass
        return None


# Import Span at end to avoid circular
from agent.runtime.events import Span
