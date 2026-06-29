"""ArrayLengthExtract node: extract array length constraints from parameter descriptions via LLM."""

import asyncio
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import ARRAY_LENGTH_EXTRACT_PROMPT
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT, parse_json_response

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_ARRAY_TYPES = [
    "aclIntArray",
    "aclFloatArray",
    "aclBoolArray",
    "aclTensorList",
    "aclScalarList",
]


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

        described = [p for p in params if p.get("llm_description")]
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
            llm = create_llm()
            sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

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


async def _extract_array_length(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract array length for a single parameter.

    The LLM returns ``{"value": [min, max] | null, "src_text": "..."}``;
    we serialize it into the ``array_length`` DB column as a JSON string so
    downstream ``attrs_build`` can recover both ``value`` and ``src_text``.
    """
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("llm_description", "")

    prompt = ARRAY_LENGTH_EXTRACT_PROMPT.format(
        params_text=description,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = parse_json_response(text, dict)
    if not result:
        return None

    value = result.get("value")
    src_text = result.get("src_text", "") or ""
    return {
        "function_name": function_name,
        "param_name": param_name,
        "array_length": json.dumps(
            {"value": value, "src_text": src_text}, ensure_ascii=False,
        ),
    }
