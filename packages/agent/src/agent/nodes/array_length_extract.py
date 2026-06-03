"""ArrayLengthExtract node: extract array length constraints from parameter descriptions via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import ARRAY_LENGTH_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_ARRAY_TYPES = [
    "aclIntArray",
    "aclFloatArray",
    "aclBoolArray",
    "aclTensorList",
    "aclScalarList",
]

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def array_length_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract array length constraints from parameter descriptions and persist to DB.

    Flow:
    1. Read parameters from state.parameters
    2. Classify each param: array type → LLM extract; non-array → "N/A"
    3. Concurrent LLM calls for array-type params (Semaphore controlled)
    4. Batch update array_length field via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ArrayLengthExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ArrayLengthExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("ArrayLengthExtract: no parameters in state for doc_id=%s, skipping", doc_id)
            return {"error": None}

        described = [p for p in params if p.get("description")]
        if not described:
            logger.info("ArrayLengthExtract: no parameters with descriptions for doc_id=%s, skipping", doc_id)
            return {"error": None}

        array_params = []
        non_array_params = []
        for p in described:
            param_type = p.get("param_type", "")
            if any(t in param_type for t in _ARRAY_TYPES):
                array_params.append(p)
            else:
                non_array_params.append(p)

        updates: list[dict] = []

        # 非数组类型: 直接标记 N/A
        for p in non_array_params:
            updates.append({
                "function_name": p.get("function_name", ""),
                "param_name": p.get("param_name", ""),
                "array_length": "N/A",
            })

        # 数组类型: LLM 提取
        if array_params:
            llm = _create_llm()
            sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

            async def _extract_one(param: dict) -> dict | None:
                async with sem:
                    return await _extract_array_length(llm, param)

            results = await asyncio.gather(*[_extract_one(p) for p in array_params])
            updates.extend(r for r in results if r is not None)

        if updates:
            result = await _mcp_client.update_param_array_length(doc_id, updates)
            logger.info(
                "ArrayLengthExtract: updated array_length for %d parameters (doc_id=%s)",
                result.get("updated", 0),
                doc_id,
            )

        return {"error": None}

    except Exception as e:
        logger.exception("ArrayLengthExtract failed for %s", operator_name)
        return {"error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_array_length(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract array length for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("description", "")

    prompt = ARRAY_LENGTH_EXTRACT_PROMPT.format(
        params_text=description,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = _parse_array_length_response(text)
    if result:
        result["param_name"] = param_name
        result["function_name"] = function_name
    return result


def _parse_array_length_response(text: str) -> dict | None:
    """Parse LLM JSON response into {array_length} dict."""
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

    logger.warning("ArrayLengthExtract: failed to parse LLM response as JSON: %s", text[:200])
    return None
