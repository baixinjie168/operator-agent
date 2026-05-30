"""OptionalExtract node: judge whether each parameter is optional via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import OPTIONAL_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def optional_extract_node(state: PipelineState) -> dict[str, Any]:
    """Judge whether each parameter is optional from its description and persist to DB.

    Reads parameters from state (populated by param_desc_extract) instead of
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

        described = [p for p in params if p.get("description")]
        if not described:
            logger.info("OptionalExtract: no parameters with descriptions for doc_id=%s, skipping", doc_id)
            return {"error": None}

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

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


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_optional(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to judge optionality for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("description", "")

    name_has_optional = "optional" in param_name.lower()

    prompt = OPTIONAL_EXTRACT_PROMPT.format(
        param_name=param_name,
        name_has_optional="true" if name_has_optional else "false",
        params_text=description,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = _parse_optional_response(text)
    if result:
        result["function_name"] = function_name
        val = result.get("is_optional")
        result["is_optional"] = 1 if val is True or val == "true" else 0
    return result


def _parse_optional_response(text: str) -> dict | None:
    """Parse LLM JSON response into {param_name, is_optional} dict."""
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

    logger.warning("OptionalExtract: failed to parse LLM response as JSON: %s", text[:200])
    return None
