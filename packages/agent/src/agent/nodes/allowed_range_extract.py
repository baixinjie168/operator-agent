"""AllowedRangeExtract node: extract parameter value range constraints via LLM."""

import asyncio
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import ALLOWED_RANGE_EXTRACT_PROMPT
from agent.utils.llm_common import CONCURRENCY_LIMIT, create_llm, parse_json_response
from agent.utils.param_validators import is_bool_type, is_tensor_type, is_ws_function
from agent.utils.semantic_rules import build_prompt_context

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

        ws_params = [p for p in params if is_ws_function(p.get("function_name", ""))]
        exe_params = [p for p in params if not is_ws_function(p.get("function_name", ""))]

        ws_sections_text = await _fetch_sections(doc_id, _WS_SECTION_TYPES) if ws_params else ""
        exe_sections_text = await _fetch_sections(doc_id, _EXE_SECTION_TYPES) if exe_params else ""

        if not ws_sections_text.strip() and not exe_sections_text.strip():
            logger.info("AllowedRangeExtract: no section content for doc_id=%s, skipping", doc_id)
            return {"error": None}

        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _extract_one(param: dict, sections_text: str) -> dict | None:
            async with sem:
                return await _extract_allowed_range(llm, param, sections_text)

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
    if is_bool_type(param_type):
        return {
            "function_name": function_name,
            "param_name": param_name,
            "allowed_range_value": json.dumps(
                [{"platform": "", "allowed_range_value": "true, false"}],
                ensure_ascii=False,
            ),
        }

    # Tensor types have no scalar value range.  Returning early avoids
    # wasting an LLM call and prevents dimension text (e.g. "2-8") in the
    # parameter description from being mis-extracted as a value range.
    if is_tensor_type(param_type):
        return {
            "function_name": function_name,
            "param_name": param_name,
            "allowed_range_value": "[]",
        }

    prompt = ALLOWED_RANGE_EXTRACT_PROMPT.format(
        param_name=param_name,
        param_type=param_type,
        sections_text=sections_text,
        semantic_rules_context=build_prompt_context(),
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = parse_json_response(text, list)
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
