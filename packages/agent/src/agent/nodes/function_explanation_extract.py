"""FunctionExplanationExtract node: extracts function explanation summary via LLM."""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import FUNCTION_EXPLANATION_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def function_explanation_extract_node(
    state: PipelineState,
) -> dict[str, Any]:
    """Extract function explanation summary via LLM and save to DB.

    Flow:
    1. Query parsed_data by doc_id via MCP
    2. Find section_type == "function_description" and get its content
    3. Call LLM to summarize function explanation
    4. Save to document_versions.function_explanation_summary via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info(
        "FunctionExplanation: doc_id=%s for %s", doc_id, operator_name
    )

    if not doc_id:
        logger.warning("FunctionExplanation: no doc_id, skipping")
        return {"function_explanation_summary": {}, "error": None}

    try:
        parsed = await _mcp_client.get_parsed_by_doc_id(doc_id)
        if not parsed:
            logger.warning("FunctionExplanation: no parsed_data")
            return {"function_explanation_summary": {}, "error": None}

        content = _find_content(parsed.get("sections", []))
        if not content:
            logger.warning("FunctionExplanation: no section found")
            return {"function_explanation_summary": {}, "error": None}

        summary = await _extract_via_llm(content, operator_name)
        logger.info(
            "FunctionExplanation: extracted for %s", operator_name
        )

        await _mcp_client.save_function_explanation_summary(
            doc_id, summary
        )

        return {
            "function_explanation_summary": summary,
            "error": None,
        }

    except Exception as e:
        logger.exception("FunctionExplanation failed for %s", operator_name)
        return {"function_explanation_summary": {}, "error": str(e)}


def _find_content(sections: list[dict]) -> str | None:
    """Find the function_description section and return its content."""
    for s in sections:
        if s.get("section_type") == "function_description":
            return s.get("content")
    return None


async def _extract_via_llm(content: str, op: str) -> dict:
    """Call LLM to summarize function explanation."""
    llm = ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )
    prompt = FUNCTION_EXPLANATION_EXTRACT_PROMPT.format(
        content=content, operator_name=op
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    """Extract JSON object from LLM response."""
    m = _JSON_BLOCK_RE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        try:
            d = json.loads(obj.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return {}
