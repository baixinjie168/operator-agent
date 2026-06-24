"""ShapeExtract node: extract unconditional shape values from parameter descriptions via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import SHAPE_EXTRACT_PROMPT
from agent.utils.llm_common import CONCURRENCY_LIMIT, create_llm, parse_json_response
from agent.utils.param_validators import is_cross_reference, is_dash

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


def _is_shape_valid(shape: str) -> bool:
    """Check whether an existing shape value is reasonable.

    Returns False for values that should trigger re-extraction from
    llm_description:
    - Empty / whitespace-only strings
    - Dash variants (-, —, –, －)
    - Cross-references to other params (e.g. "与self一致")

    Handles JSON format: {"*": "[M, K1]"} or {"platform": "[M, K1]"}.
    If any platform value is valid, returns True.
    """
    s = shape.strip()
    if not s:
        return False
    # Try parsing as JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and parsed:
            return any(
                _is_plain_shape_valid(v)
                for v in parsed.values()
                if isinstance(v, str)
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return _is_plain_shape_valid(s)


def _is_plain_shape_valid(s: str) -> bool:
    """Check a plain text shape value (non-JSON)."""
    s = s.strip()
    if not s:
        return False
    if is_dash(s):
        return False
    cleaned = s.replace("`", "")
    if is_cross_reference(cleaned):
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

        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict | None:
            async with sem:
                return await _extract_shape(llm, param)

        results = await asyncio.gather(*[_extract_one(p) for p in described])

        shape_updates = [r for r in results if r is not None and r.get("shape")]
        # Wrap plain text shape values as JSON: {"*": value}
        for u in shape_updates:
            val = u["shape"]
            if isinstance(val, str) and not val.startswith("{"):
                u["shape"] = json.dumps({"*": val}, ensure_ascii=False)
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


async def _extract_shape(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract shape for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("llm_description", "")

    prompt = SHAPE_EXTRACT_PROMPT.format(param_name=param_name, params_text=description)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = parse_json_response(text, dict)
    if result:
        result["function_name"] = function_name
    return result
