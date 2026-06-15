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

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Matches cross-reference patterns like "与self一致", "同input相同"
_RELATIVE_REF_RE = re.compile(
    r"^(?:与|同|和|跟)"
    r".{1,20}"
    r"(?:一致|相同|一样|保持一致|保持一致|同)$",
)


def _is_shape_valid(shape: str) -> bool:
    """Check whether an existing shape value is reasonable.

    Returns False for values that should trigger re-extraction from
    llm_description:
    - Empty / whitespace-only strings
    - Dash variants (-, —, –, －)
    - Cross-references to other params (e.g. "与self一致")
    """
    s = shape.strip()
    if not s:
        return False
    if s in ("-", "—", "–", "－"):
        return False
    cleaned = s.replace("`", "")
    if _RELATIVE_REF_RE.match(cleaned):
        return False
    return True


async def shape_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract unconditional shape values from parameter descriptions and persist to DB.

    Reads parameters from state (populated by llm_description_extract) instead of
    making a redundant MCP query. Each parameter gets its own LLM call
    for precise extraction, with controlled concurrency.

    Flow:
    1. Read parameters from state.parameters (no MCP query needed)
    2. Filter to parameters with non-empty descriptions
    3. Concurrent LLM call per parameter (Semaphore controlled)
    4. Batch update shape field via MCP
    """
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

        described = [
            p for p in params
            if p.get("llm_description")
            and (not p.get("shape") or not _is_shape_valid(p.get("shape", "")))
        ]
        if not described:
            logger.info("ShapeExtract: no parameters needing shape extraction for doc_id=%s, skipping", doc_id)
            return {"error": None}

        # Clear invalid shape values before re-extraction so downstream
        # nodes don't see stale bad data.
        invalid_clears = [
            {"function_name": p["function_name"], "param_name": p["param_name"], "shape": ""}
            for p in described
            if p.get("shape") and not _is_shape_valid(p.get("shape", ""))
        ]
        if invalid_clears:
            await _mcp_client.update_param_shape(doc_id, invalid_clears)
            logger.info(
                "ShapeExtract: cleared %d invalid shape values (doc_id=%s)",
                len(invalid_clears), doc_id,
            )

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict | None:
            async with sem:
                return await _extract_shape(llm, param)

        results = await asyncio.gather(*[_extract_one(p) for p in described])

        shape_updates = [r for r in results if r is not None and r.get("shape")]
        if shape_updates:
            result = await _mcp_client.update_param_shape(doc_id, shape_updates)
            logger.info(
                "ShapeExtract: updated shape for %d/%d parameters (doc_id=%s)",
                result.get("updated", 0),
                len(described),
                doc_id,
            )
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
    description = param.get("llm_description", "")

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
