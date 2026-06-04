"""FunctionSignatureExtract node: extract function signatures via LLM."""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import FUNCTION_SIGNATURE_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def function_signature_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract function signatures from function_prototype section via LLM.

    Flow:
    1. Get parsed_data by doc_id via MCP
    2. Find function_prototype section content
    3. Call LLM to extract structured signatures
    4. Save to function_signatures table via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("FunctionSignatureExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("FunctionSignatureExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        section = await _mcp_client.get_section(doc_id, "function_prototype")
        if not section:
            logger.warning("FunctionSignatureExtract: no function_prototype section for doc_id=%s", doc_id)
            return {"error": None}

        content = section.get("content", "")
        if not content:
            logger.warning("FunctionSignatureExtract: empty function_prototype content for doc_id=%s", doc_id)
            return {"error": None}

        signatures = await _extract_signatures_via_llm(content)
        if not signatures:
            logger.info("FunctionSignatureExtract: LLM returned no results for doc_id=%s", doc_id)
            return {"function_signatures": [], "error": None}

        logger.info(
            "FunctionSignatureExtract: extracted %d signatures for %s",
            len(signatures),
            operator_name,
        )

        result = await _mcp_client.save_function_signatures(doc_id, signatures)
        logger.info(
            "FunctionSignatureExtract: saved %d signatures for doc_id=%s",
            result.get("saved", 0),
            doc_id,
        )

        return {"function_signatures": signatures, "error": None}

    except Exception as e:
        logger.exception("FunctionSignatureExtract failed for %s", operator_name)
        return {"error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_signatures_via_llm(content: str) -> list[dict]:
    """Call LLM to extract function signatures from content."""
    llm = _create_llm()
    prompt = FUNCTION_SIGNATURE_EXTRACT_PROMPT.format(content=content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_json_response(text)


def _parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)

    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            data = json.loads(array_match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []
