"""ShapeExtract node: extract unconditional shape values from parameter descriptions via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import SHAPE_EXTRACT_PROMPT
from agent.runtime.context import get_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_STEP_NAME = "shape_extract"
_STEP_LABEL = "Shape提取"

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def shape_extract_node(state: PipelineState) -> dict[str, Any]:
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ShapeExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ShapeExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("ShapeExtract: no parameters in state for doc_id=%s, skipping", doc_id)
            return {"error": None}

        described = [p for p in params if p.get("description")]
        undescribed = [p for p in params if not p.get("description")]

        if undescribed and ctx and ctx.manager:
            for p in undescribed:
                pn = p.get("param_name", "")
                fn = p.get("function_name", "")
                skip_span = ctx.manager.open_span(
                    run_id=ctx.run_id,
                    parent_span_id=ctx.current_span_id,
                    span_type=SpanType.NODE,
                    name=f"{_STEP_NAME}:{fn}:{pn}",
                )
                ctx.manager.close_span(ctx.run_id, skip_span, SpanStatus.SUCCESS)
                ctx.manager.emit(EventType.PARAM_STEP_ERROR, ctx.run_id, skip_span, {
                    "agent_id": "doc",
                    "node_id": _STEP_NAME,
                    "param_name": pn,
                    "function_name": fn,
                    "step_name": _STEP_NAME,
                    "message": f"参数 {pn} {_STEP_LABEL} 已跳过: 无参数描述",
                    "error": "无参数描述",
                })

        if not described:
            logger.info("ShapeExtract: no parameters with descriptions for doc_id=%s, skipping", doc_id)
            return {"error": None}

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)
        ctx = get_context()

        async def _extract_one(param: dict) -> dict | None:
            param_name = param.get("param_name", "")
            function_name = param.get("function_name", "")
            parent_span_id = ctx.current_span_id if ctx else None
            param_span = None

            if ctx and ctx.manager:
                param_span = ctx.manager.open_span(
                    run_id=ctx.run_id,
                    parent_span_id=parent_span_id,
                    span_type=SpanType.NODE,
                    name=f"{_STEP_NAME}:{function_name}:{param_name}",
                )
                ctx.manager.emit(EventType.PARAM_STEP_START, ctx.run_id, param_span, {
                    "agent_id": "doc",
                    "node_id": _STEP_NAME,
                    "param_name": param_name,
                    "function_name": function_name,
                    "step_name": _STEP_NAME,
                    "message": f"参数 {param_name} {_STEP_LABEL} 开始...",
                })

            try:
                async with sem:
                    result = await _extract_shape(llm, param)

                if ctx and ctx.manager and param_span:
                    ctx.manager.close_span(ctx.run_id, param_span, SpanStatus.SUCCESS)
                    ctx.manager.emit(EventType.PARAM_STEP_COMPLETE, ctx.run_id, param_span, {
                        "agent_id": "doc",
                        "node_id": _STEP_NAME,
                        "param_name": param_name,
                        "function_name": function_name,
                        "step_name": _STEP_NAME,
                        "message": f"参数 {param_name} {_STEP_LABEL} 完成",
                        "result_preview": (result.get("shape", "") or "")[:200],
                        "has_result": bool(result and result.get("shape")),
                    })

                return result
            except Exception as exc:
                if ctx and ctx.manager and param_span:
                    ctx.manager.close_span(ctx.run_id, param_span, SpanStatus.ERROR, error=str(exc))
                    ctx.manager.emit(EventType.PARAM_STEP_ERROR, ctx.run_id, param_span, {
                        "agent_id": "doc",
                        "node_id": _STEP_NAME,
                        "param_name": param_name,
                        "function_name": function_name,
                        "step_name": _STEP_NAME,
                        "message": f"参数 {param_name} {_STEP_LABEL} 失败: {exc}",
                        "error": str(exc),
                    })
                return None

        results = await asyncio.gather(*[_extract_one(p) for p in described])

        shape_updates = [r for r in results if r is not None and r.get("shape")]

        if shape_updates:
            result = await _mcp_client.update_param_shape(doc_id, shape_updates)
            updated = result.get("updated", 0)
            logger.info("ShapeExtract: updated shape for %d/%d parameters", updated, len(described))
        else:
            logger.info("ShapeExtract: no unconditional shapes extracted for doc_id=%s", doc_id)

        return {"error": None}

    except Exception as e:
        logger.exception("ShapeExtract failed for %s", operator_name)
        return {"error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_shape(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract shape for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("description", "")

    prompt = SHAPE_EXTRACT_PROMPT.format(param_name=param_name, params_text=description)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = _parse_shape_response(text)
    if result:
        result["function_name"] = function_name
    return result


def _parse_shape_response(text: str) -> dict | None:
    """Parse LLM JSON response into {param_name, shape} dict."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{[^{}]*\}", text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("ShapeExtract: failed to parse LLM response as JSON: %s", text[:200])
    return None
