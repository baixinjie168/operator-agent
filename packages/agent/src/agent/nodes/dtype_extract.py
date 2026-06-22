"""DtypeExtract node: extract data types from parameter descriptions via LLM."""

import asyncio
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import DTYPE_EXTRACT_PROMPT
from agent.utils.llm_common import CONCURRENCY_LIMIT, create_llm, parse_json_response
from agent.utils.param_validators import VALID_DTYPES, is_cross_reference, is_dash

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


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
    if is_dash(s):
        return False
    cleaned = s.replace("`", "")
    if is_cross_reference(cleaned):
        return False
    tokens = [t.strip().upper() for t in re.split(r"[,、，/]", s) if t.strip()]
    if not tokens:
        return False
    return all(t in VALID_DTYPES for t in tokens)


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

        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

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


async def _extract_dtype(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract data type for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("llm_description", "")

    prompt = DTYPE_EXTRACT_PROMPT.format(param_name=param_name, params_text=description)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = parse_json_response(text, dict)
    if result:
        result["function_name"] = function_name
        if result.get("dtype"):
            result["dtype"] = result["dtype"].upper()
    return result
