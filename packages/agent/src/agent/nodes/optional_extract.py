"""OptionalExtract node: judge whether each parameter is optional via LLM."""

import asyncio
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import OPTIONAL_EXTRACT_PROMPT
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT, parse_json_response

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def optional_extract_node(state: PipelineState) -> dict[str, Any]:
    """Judge whether each parameter is optional from its description and persist to DB.

    Reads parameters from state (populated by llm_description_extract) instead of
    making a redundant MCP query. Each parameter gets its own LLM call
    for precise判断, with controlled concurrency.

    Flow:
    1. Read parameters from state.parameters (no MCP query needed)
    2. Filter to parameters with non-empty descriptions
    3. Concurrent LLM call per parameter (Semaphore controlled)
    4. Batch update is_optional field via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("OptionalExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("OptionalExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("OptionalExtract: no parameters in state for doc_id=%s, skipping", doc_id)
            return {"error": None}

        described = [p for p in params if p.get("llm_description")]
        if not described:
            logger.info("OptionalExtract: no parameters with descriptions for doc_id=%s, skipping", doc_id)
            return {"error": None}

        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict | None:
            async with sem:
                return await _extract_optional(llm, param)

        results = await asyncio.gather(*[_extract_one(p) for p in described])

        updates = [r for r in results if r is not None]
        if updates:
            result = await _mcp_client.update_param_optional(doc_id, updates)
            logger.info(
                "OptionalExtract: updated is_optional for %d parameters (doc_id=%s)",
                result.get("updated", 0),
                doc_id,
            )

        return {"error": None}

    except Exception as e:
        logger.exception("OptionalExtract failed for %s", operator_name)
        return {"error": str(e)}


async def _extract_optional(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to judge optionality for a single parameter.

    Deterministic safety net: if the parameter name contains "Optional"
    and the LLM returns false, override to true unless the description
    explicitly says "必选"/"必须"/"不可为空". This handles documents
    where the direction column says "输入" (not "可选输入") and the
    description lacks optional keywords — the only signal is the name.
    """
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("llm_description", "")

    name_has_optional = "optional" in param_name.lower()

    prompt = OPTIONAL_EXTRACT_PROMPT.format(
        param_name=param_name,
        name_has_optional="true" if name_has_optional else "false",
        params_text=description,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = parse_json_response(text, dict)
    if result:
        result["function_name"] = function_name
        val = result.get("is_optional")
        is_optional = 1 if val is True or val == "true" else 0

        # Safety net: name has "Optional" but LLM said false
        if not is_optional and name_has_optional:
            required_markers = ("必选", "必须", "不可为空", "不可为空指针")
            if not any(m in description for m in required_markers):
                is_optional = 1
                logger.info(
                    "OptionalExtract: override %s -> is_optional=true "
                    "(name has Optional, desc has no 必选 markers)",
                    param_name,
                )

        result["is_optional"] = is_optional
    return result
