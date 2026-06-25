"""AllowedRangeExtract node: extract parameter value range constraints.

Refactored to use DeepAgent + knowledge base (allowed_range_agent.py)
instead of per-param LLM calls with a long prompt.

Flow:
  Phase 0:  Bool/tensor short-circuit (deterministic, zero LLM)
  Phase 1:  Batch Agent extraction (one call per function group)
  Phase 1b: Per-param fallback for missing params
  Phase 2:  DB update
"""

import json
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.allowed_range_agent import (
    _extract_batch_via_agent,
)
from agent.nodes.state import PipelineState
from agent.utils.param_validators import is_bool_type, is_tensor_type, is_ws_function

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

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


async def allowed_range_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract parameter value range constraints from document sections.

    Groups parameters by function_name:
    - GetWorkspaceSize functions -> params_get_workspace + return_codes + constraints
    - Execute functions -> params_execute + return_codes + constraints

    Phase 0: bool/tensor short-circuit (zero LLM)
    Phase 1: batch Agent extraction for remaining params (1 call per group)
    Phase 1b: per-param fallback for params missing from batch result
    Phase 2: batch DB update
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

        ws_params = [p for p in params if is_ws_function(p.get("function_name", ""))]
        exe_params = [p for p in params if not is_ws_function(p.get("function_name", ""))]

        ws_sections_text = await _fetch_sections(doc_id, _WS_SECTION_TYPES) if ws_params else ""
        exe_sections_text = await _fetch_sections(doc_id, _EXE_SECTION_TYPES) if exe_params else ""

        if not ws_sections_text.strip() and not exe_sections_text.strip():
            logger.info("AllowedRangeExtract: no section content for doc_id=%s, skipping", doc_id)
            return {"error": None}

        updates: list[dict] = []

        # Process each function group
        if ws_params and ws_sections_text.strip():
            group_updates = await _process_group(ws_params, ws_sections_text)
            updates.extend(group_updates)

        if exe_params and exe_sections_text.strip():
            group_updates = await _process_group(exe_params, exe_sections_text)
            updates.extend(group_updates)

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


async def _process_group(
    params: list[dict],
    sections_text: str,
) -> list[dict]:
    """Process a group of params: short-circuit + batch Agent.

    Returns list of DB update dicts:
    [{function_name, param_name, allowed_range_value}, ...]
    """
    updates: list[dict] = []
    agent_params: list[dict] = []

    for param in params:
        pname = param.get("param_name", "")
        ptype = param.get("param_type", "")
        fn = param.get("function_name", "")

        # Phase 0: short-circuit bool and tensor types (zero LLM)
        if is_bool_type(ptype):
            updates.append({
                "function_name": fn,
                "param_name": pname,
                "allowed_range_value": json.dumps(
                    [{"platform": "", "allowed_range_value": "true, false"}],
                    ensure_ascii=False,
                ),
            })
            continue

        if is_tensor_type(ptype):
            updates.append({
                "function_name": fn,
                "param_name": pname,
                "allowed_range_value": "[]",
            })
            continue

        agent_params.append(param)

    # Phase 1: batch Agent extraction for remaining params
    if agent_params:
        batch_result = await _extract_batch_via_agent(agent_params, sections_text)

        for param in agent_params:
            pname = param.get("param_name", "")
            fn = param.get("function_name", "")
            entries = batch_result.get(pname, [])
            updates.append({
                "function_name": fn,
                "param_name": pname,
                "allowed_range_value": json.dumps(entries, ensure_ascii=False) if entries else "[]",
            })

    short_circuit_count = len(params) - len(agent_params)
    logger.info(
        "AllowedRangeExtract: group processed %d params (%d short-circuit, %d agent)",
        len(params), short_circuit_count, len(agent_params),
    )

    return updates


async def _fetch_sections(doc_id: int, section_types: list[str]) -> str:
    parts: list[str] = []
    for section_type in section_types:
        section = await _mcp_client.get_section(doc_id, section_type)
        if section and section.get("content"):
            parts.append(f"## {section_type}\n{section['content']}")
    return "\n\n".join(parts)
