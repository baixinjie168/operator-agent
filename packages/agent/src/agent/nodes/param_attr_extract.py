"""ParamAttrExtract node: extract parameter attributes from descriptions via regex."""

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.runtime.context import get_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_STEP_NAME = "param_attr_extract"
_STEP_LABEL = "非连续Tensor提取"

DISCONTINUOUS_ROW_RE = re.compile(
    r"^\|\s*非连续\s*Tensor\s*\|\s*(.+?)\s*\|",
    re.MULTILINE,
)
SUPPORTED_VALUE_RE = re.compile(r"[√✓✔]|(?<!不)支持")

PARAM_DESC_ROW_RE = re.compile(
    r"^\|\s*描述\s*\|\s*(.+?)\s*\|",
    re.MULTILINE,
)
_EMPTY_DESC_VALUES = {"（无）", "(无)", "无", ""}


async def param_attr_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract parameter attributes from descriptions using regex. No LLM calls.

    Reads parameters from state (populated by param_desc_extract). For each
    parameter with a non-empty description, extracts:
    - is_support_discontinuous: from the "非连续Tensor" row
    - param_desc: from the "描述" row

    Flow:
    1. Read parameters from state.parameters (no MCP query needed)
    2. Filter to parameters with non-empty descriptions
    3. Regex extract per parameter (no LLM call)
    4. Batch update is_support_discontinuous and param_desc via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ParamAttrExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ParamAttrExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("ParamAttrExtract: no parameters in state for doc_id=%s, skipping", doc_id)
            return {"error": None}

        described = [p for p in params if p.get("description")]
        without_desc = [p for p in params if not p.get("description")]

        ctx = get_context()
        if without_desc and ctx and ctx.manager:
            for p in without_desc:
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
                    "message": f"参数 {pn} {_STEP_LABEL} 已跳过: 无描述内容",
                    "error": "无描述内容",
                })

        if not described:
            logger.info("ParamAttrExtract: no parameters with descriptions for doc_id=%s, skipping", doc_id)
            return {"error": None}

        updates = []
        for p in described:
            param_name = p.get("param_name", "")
            function_name = p.get("function_name", "")
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
                result = _extract_attrs(p)
                disc_val = ""
                if result and result.get("is_support_discontinuous"):
                    try:
                        disc_data = json.loads(result["is_support_discontinuous"])
                        disc_val = str(disc_data.get("value", "N/A"))
                    except (json.JSONDecodeError, AttributeError):
                        disc_val = str(result["is_support_discontinuous"])

                if ctx and ctx.manager and param_span:
                    ctx.manager.close_span(ctx.run_id, param_span, SpanStatus.SUCCESS)
                    ctx.manager.emit(EventType.PARAM_STEP_COMPLETE, ctx.run_id, param_span, {
                        "agent_id": "doc",
                        "node_id": _STEP_NAME,
                        "param_name": param_name,
                        "function_name": function_name,
                        "step_name": _STEP_NAME,
                        "message": f"参数 {param_name} {_STEP_LABEL} 完成",
                        "result_preview": disc_val,
                        "has_result": bool(disc_val and disc_val != "N/A"),
                    })

                if result is not None:
                    updates.append(result)

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

        if updates:
            result = await _mcp_client.update_param_attrs(doc_id, updates)
            logger.info(
                "ParamAttrExtract: updated attrs for %d parameters (doc_id=%s)",
                result.get("updated", 0),
                doc_id,
            )

        return {"error": None}

    except Exception as e:
        logger.exception("ParamAttrExtract failed for %s", operator_name)
        return {"error": str(e)}


def _extract_attrs(param: dict) -> dict | None:
    """Extract is_support_discontinuous and param_desc for a single parameter."""
    param_type = param.get("param_type", "")
    description = param.get("description", "")
    function_name = param.get("function_name", "")
    param_name = param.get("param_name", "")

    if not description:
        return None

    result: dict[str, Any] = {
        "function_name": function_name,
        "param_name": param_name,
    }

    if "tensor" not in param_type.lower():
        result["is_support_discontinuous"] = json.dumps(
            {"value": "N/A", "src_text": ""}, ensure_ascii=False,
        )
    else:
        match = DISCONTINUOUS_ROW_RE.search(description)
        if match and SUPPORTED_VALUE_RE.search(match.group(1)):
            result["is_support_discontinuous"] = json.dumps(
                {"value": True, "src_text": "| 非连续Tensor | √ |"}, ensure_ascii=False,
            )
        else:
            result["is_support_discontinuous"] = json.dumps(
                {"value": False, "src_text": ""}, ensure_ascii=False,
            )

    desc_match = PARAM_DESC_ROW_RE.search(description)
    if desc_match:
        val = desc_match.group(1).strip()
        result["param_desc"] = "" if val in _EMPTY_DESC_VALUES else val
    else:
        result["param_desc"] = ""

    return result
