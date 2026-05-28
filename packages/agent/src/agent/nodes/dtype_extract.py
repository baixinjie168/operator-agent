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

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def dtype_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract data type values from parameter descriptions and persist to DB.

    Reads parameters from state (populated by param_desc_extract) instead of
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

        described = [p for p in params if p.get("description")]
        if not described:
            logger.info("DtypeExtract: no parameters with descriptions for doc_id=%s, skipping", doc_id)
            return {"error": None}

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict | None:
            async with sem:
                return await _extract_dtype(llm, param)

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
    description = param.get("description", "")

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
