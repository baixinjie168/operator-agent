"""DtypeExtract node: extract data types from parameter descriptions via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import DTYPE_EXTRACT_PROMPT
from agent.runtime.context import get_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_STEP_NAME = "dtype_extract"
_STEP_LABEL = "数据类型提取"

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_VALID_DTYPES = frozenset({
    "FLOAT", "FLOAT32", "FLOAT16", "INT8", "INT32", "UINT8",
    "INT16", "UINT16", "UINT32", "INT64", "UINT64", "DOUBLE",
    "FLOAT64", "BOOL", "STRING", "COMPLEX64", "COMPLEX128",
    "BF16", "BFLOAT16", "INT", "UINT1", "COMPLEX32",
})

# Matches cross-reference patterns like "与self一致", "同input相同"
_RELATIVE_REF_RE = re.compile(
    r"^(?:与|同|和|跟)"
    r".{1,20}"
    r"(?:一致|相同|一样|保持一致|保持一致|同)$",
)


def _is_dtype_valid(dtype_desc: str) -> bool:
    """Check whether an existing dtype_desc value is reasonable.

    Returns False for values that should trigger re-extraction:
    - Empty / whitespace-only strings
    - Dash variants
    - Cross-references to other params
    - Tokens not in the approved dtype whitelist
    """
    s = dtype_desc.strip()
    if not s:
        return False
    if s in ("-", "—", "–", "－"):
        return False
    cleaned = s.replace("`", "")
    if _RELATIVE_REF_RE.match(cleaned):
        return False
    tokens = [t.strip().upper() for t in re.split(r"[,、，/]", s) if t.strip()]
    if not tokens:
        return False
    return all(t in _VALID_DTYPES for t in tokens)


async def dtype_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract data type values from parameter descriptions and persist to DB.

    Reads parameters from state (populated by llm_description_extract) instead of
    making a redundant MCP query. Each parameter gets its own LLM call
    for precise extraction, with controlled concurrency.

    Flow:
    1. Read parameters from state.parameters (no MCP query needed)
    2. Filter to parameters with non-empty descriptions
    3. Concurrent LLM call per parameter (Semaphore controlled)
    4. Batch update dtype_desc field via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("DtypeExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("DtypeExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("DtypeExtract: no parameters in state for doc_id=%s, skipping", doc_id)
            return {"error": None}

        described = [
            p for p in params
            if p.get("llm_description")
            and (not p.get("dtype_desc") or not _is_dtype_valid(p.get("dtype_desc", "")))
        ]
        undescribed = [p for p in params if not p.get("llm_description")]

        ctx = get_context()
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
            logger.info("DtypeExtract: no parameters needing dtype extraction for doc_id=%s, skipping", doc_id)
            return {"error": None}

        # Clear invalid dtype values before re-extraction so downstream
        # nodes don't see stale bad data.
        invalid_clears = [
            {"function_name": p["function_name"], "param_name": p["param_name"], "dtype": ""}
            for p in described
            if p.get("dtype_desc") and not _is_dtype_valid(p.get("dtype_desc", ""))
        ]
        if invalid_clears:
            await _mcp_client.update_param_dtype(doc_id, invalid_clears)
            logger.info(
                "DtypeExtract: cleared %d invalid dtype values (doc_id=%s)",
                len(invalid_clears), doc_id,
            )

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
                    result = await _extract_dtype(llm, param)

                if ctx and ctx.manager and param_span:
                    ctx.manager.close_span(ctx.run_id, param_span, SpanStatus.SUCCESS)
                    ctx.manager.emit(EventType.PARAM_STEP_COMPLETE, ctx.run_id, param_span, {
                        "agent_id": "doc",
                        "node_id": _STEP_NAME,
                        "param_name": param_name,
                        "function_name": function_name,
                        "step_name": _STEP_NAME,
                        "message": f"参数 {param_name} {_STEP_LABEL} 完成",
                        "result_preview": result.get("dtype", "") or "" if result else "",
                        "has_result": bool(result and result.get("dtype")),
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

        dtype_updates = [r for r in results if r is not None and r.get("dtype")]
        if dtype_updates:
            result = await _mcp_client.update_param_dtype(doc_id, dtype_updates)
            logger.info(
                "DtypeExtract: updated dtype for %d/%d parameters (doc_id=%s)",
                result.get("updated", 0),
                len(described),
                doc_id,
            )
        else:
            logger.info("DtypeExtract: no dtypes extracted for doc_id=%s", doc_id)

        return {"error": None}

    except Exception as e:
        logger.exception("DtypeExtract failed for %s", operator_name)
        return {"error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_dtype(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract data type for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("llm_description", "")

    prompt = DTYPE_EXTRACT_PROMPT.format(param_name=param_name, params_text=description)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = _parse_dtype_response(text)
    if result:
        result["function_name"] = function_name
        if result.get("dtype"):
            result["dtype"] = result["dtype"].upper()
    return result


def _parse_dtype_response(text: str) -> dict | None:
    """Parse LLM JSON response into {param_name, dtype} dict."""
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

    logger.warning("DtypeExtract: failed to parse LLM response as JSON: %s", text[:200])
    return None
