"""DFormatExtract node: extract data formats from parameter descriptions via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import DFORMAT_EXTRACT_PROMPT
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT, parse_json_response
from agent.utils.param_validators import is_cross_reference, is_dash

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_VALID_DFORMATS = frozenset({
    "ND", "NCHW", "NHWC", "HWCN", "NDHWC", "NCDHW", "NC", "NCL",
    "NC1HWC0", "FRACTAL_Z", "NC1HWC0_C04", "FRACTAL_NZ",
    "NDC1HWC0", "FRACTAL_Z_3D",
})


def _is_dformat_valid(dformat_desc: str) -> bool:
    """Check whether an existing dformat_desc value is reasonable.

    Handles JSON format: {"*": "ND"} or {"platform": "NCHW"}.
    """
    s = dformat_desc.strip()
    if not s:
        return False
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and parsed:
            return any(
                _is_plain_dformat_valid(v)
                for v in parsed.values()
                if isinstance(v, str)
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return _is_plain_dformat_valid(s)


def _is_plain_dformat_valid(s: str) -> bool:
    """Check a plain text dformat value (non-JSON)."""
    s = s.strip()
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
    return all(t in _VALID_DFORMATS for t in tokens)


async def dformat_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract data format values from parameter descriptions and persist to DB.

    Reads parameters from state (populated by llm_description_extract) instead of
    making a redundant MCP query. Each parameter gets its own LLM call
    for precise extraction, with controlled concurrency.

    Flow:
    1. Read parameters from state.parameters (no MCP query needed)
    2. Filter to parameters with non-empty descriptions
    3. Concurrent LLM call per parameter (Semaphore controlled)
    4. Batch update dformat_desc field via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("DFormatExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("DFormatExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("DFormatExtract: no parameters in state for doc_id=%s, skipping", doc_id)
            return {"error": None}

        described = [
            p for p in params
            if p.get("llm_description")
            and (not p.get("dformat_desc") or not _is_dformat_valid(p.get("dformat_desc", "")))
        ]
        if not described:
            logger.info("DFormatExtract: no parameters needing dformat extraction for doc_id=%s, skipping", doc_id)
            return {"error": None}

        # Clear invalid dformat values before re-extraction so downstream
        # nodes don't see stale bad data.
        invalid_clears = [
            {"function_name": p["function_name"], "param_name": p["param_name"], "dformat": ""}
            for p in described
            if p.get("dformat_desc") and not _is_dformat_valid(p.get("dformat_desc", ""))
        ]
        if invalid_clears:
            await _mcp_client.update_param_dformat(doc_id, invalid_clears)
            logger.info(
                "DFormatExtract: cleared %d invalid dformat values (doc_id=%s)",
                len(invalid_clears), doc_id,
            )

        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict | None:
            async with sem:
                return await _extract_dformat(llm, param)

        results = await asyncio.gather(*[_extract_one(p) for p in described])

        dformat_updates = [r for r in results if r is not None and r.get("dformat")]
        # Wrap plain text dformat values as JSON: {"*": value}
        for u in dformat_updates:
            val = u["dformat"]
            if isinstance(val, str) and not val.startswith("{"):
                u["dformat"] = json.dumps({"*": val}, ensure_ascii=False)
        if dformat_updates:
            result = await _mcp_client.update_param_dformat(doc_id, dformat_updates)
            logger.info(
                "DFormatExtract: updated dformat for %d/%d parameters (doc_id=%s)",
                result.get("updated", 0),
                len(described),
                doc_id,
            )
        else:
            logger.info("DFormatExtract: no dformats extracted for doc_id=%s", doc_id)

        return {"error": None}

    except Exception as e:
        logger.exception("DFormatExtract failed for %s", operator_name)
        return {"error": str(e)}


async def _extract_dformat(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract data format for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("llm_description", "")

    prompt = DFORMAT_EXTRACT_PROMPT.format(param_name=param_name, params_text=description)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = parse_json_response(text, dict)
    if result:
        result["function_name"] = function_name
        if result.get("dformat"):
            result["dformat"] = result["dformat"].upper()
    return result
