"""AllowedRangeExtract node: extract parameter value range constraints via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import ALLOWED_RANGE_EXTRACT_PROMPT
from agent.runtime.context import get_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_STEP_NAME = "allowed_range_extract"
_STEP_LABEL = "取值范围提取"

_WS_SECTION_TYPES = [
    "params_get_workspace",
    "return_codes_get_workspace",
    "constraints",
]

_EXE_SECTION_TYPES = [
    "params_execute",
    "return_codes_execute",
    "constraints",
]

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _is_ws_function(function_name: str) -> bool:
    return "GetWorkspaceSize" in function_name


async def allowed_range_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract parameter value range constraints from document sections via LLM.

    Groups parameters by function_name:
    - GetWorkspaceSize functions → fetch params_get_workspace + return_codes_get_workspace + constraints
    - Execute functions → fetch params_execute + return_codes_execute + constraints

    Each parameter is sent to LLM with its group's section content.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("AllowedRangeExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("AllowedRangeExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("AllowedRangeExtract: no parameters in state, skipping")
            return {"error": None}

        ws_params = [p for p in params if _is_ws_function(p.get("function_name", ""))]
        exe_params = [p for p in params if not _is_ws_function(p.get("function_name", ""))]

        ws_sections_text = await _fetch_sections(doc_id, _WS_SECTION_TYPES) if ws_params else ""
        exe_sections_text = await _fetch_sections(doc_id, _EXE_SECTION_TYPES) if exe_params else ""

        ctx = get_context()

        if not ws_sections_text.strip() and not exe_sections_text.strip():
            logger.info("AllowedRangeExtract: no section content for doc_id=%s, skipping", doc_id)
            if ctx and ctx.manager:
                for p in params:
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
                        "message": f"参数 {pn} {_STEP_LABEL} 已跳过: 无段落内容",
                        "error": "无段落内容",
                    })
            return {"error": None}

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _extract_one(param: dict, sections_text: str) -> dict | None:
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
                    result = await _extract_allowed_range(llm, param, sections_text)

                range_val = result.get("allowed_range_value", "[]") if result else "[]"
                preview = range_val if len(range_val) <= 50 else range_val[:50] + "…"

                if ctx and ctx.manager and param_span:
                    ctx.manager.close_span(ctx.run_id, param_span, SpanStatus.SUCCESS)
                    ctx.manager.emit(EventType.PARAM_STEP_COMPLETE, ctx.run_id, param_span, {
                        "agent_id": "doc",
                        "node_id": _STEP_NAME,
                        "param_name": param_name,
                        "function_name": function_name,
                        "step_name": _STEP_NAME,
                        "message": f"参数 {param_name} {_STEP_LABEL} 完成",
                        "result_preview": preview,
                        "has_result": range_val != "[]",
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

        tasks: list = []
        for p in ws_params:
            tasks.append(_extract_one(p, ws_sections_text))
        for p in exe_params:
            tasks.append(_extract_one(p, exe_sections_text))

        results = await asyncio.gather(*tasks)
        updates = [r for r in results if r is not None]

        if updates:
            result = await _mcp_client.update_param_allowed_range(doc_id, updates)
            logger.info(
                "AllowedRangeExtract: updated %d parameters (doc_id=%s)",
                result.get("updated", 0),
                doc_id,
            )

        return {"error": None}

    except Exception as e:
        logger.exception("AllowedRangeExtract failed for %s", operator_name)
        return {"error": str(e)}


async def _fetch_sections(doc_id: int, section_types: list[str]) -> str:
    parts: list[str] = []
    for section_type in section_types:
        section = await _mcp_client.get_section(doc_id, section_type)
        if section and section.get("content"):
            parts.append(f"## {section_type}\n{section['content']}")
    return "\n\n".join(parts)


def _is_bool_type(param_type: str) -> bool:
    """Check if parameter type is bool."""
    return param_type.lower() == "bool"


async def _extract_allowed_range(llm: ChatOpenAI, param: dict, sections_text: str) -> dict | None:
    param_name = param.get("param_name", "")
    param_type = param.get("param_type", "")
    function_name = param.get("function_name", "")

    if not sections_text.strip():
        return {
            "function_name": function_name,
            "param_name": param_name,
            "allowed_range_value": "[]",
        }

    # Bool type: short-circuit with [true, false]
    if _is_bool_type(param_type):
        return {
            "function_name": function_name,
            "param_name": param_name,
            "allowed_range_value": json.dumps(
                [{"platform": "", "allowed_range_value": "true, false"}],
                ensure_ascii=False,
            ),
        }

    prompt = ALLOWED_RANGE_EXTRACT_PROMPT.format(
        param_name=param_name,
        param_type=param_type,
        sections_text=sections_text,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = _parse_allowed_range_response(text)
    if result is not None:
        return {
            "function_name": function_name,
            "param_name": param_name,
            "allowed_range_value": json.dumps(result, ensure_ascii=False),
        }
    return {
        "function_name": function_name,
        "param_name": param_name,
        "allowed_range_value": "[]",
    }


def _parse_allowed_range_response(text: str) -> list[dict] | None:
    """Parse LLM JSON response into list of {platform, allowed_range_value} dicts."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("AllowedRangeExtract: failed to parse LLM response: %s", text[:200])
    return None


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )
