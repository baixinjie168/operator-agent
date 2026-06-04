"""ParamAttrExtract node: extract parameter attributes from descriptions via regex."""

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

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
        if not described:
            logger.info("ParamAttrExtract: no parameters with descriptions for doc_id=%s, skipping", doc_id)
            return {"error": None}

        updates = [_extract_attrs(p) for p in described]
        updates = [u for u in updates if u is not None]

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
