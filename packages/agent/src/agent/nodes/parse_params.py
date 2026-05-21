"""ParseParams node: extract function parameters via LLM and persist to DB."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_EXTRACT_PROMPT = """你是一个参数提取专家。从下面的函数原型内容中，找出所有函数及其所有入参和出参。

要求：
1. 每个函数对应一个对象
2. parameter 列表中包含该函数的所有参数名（包括输入和输出参数）
3. 严格按以下 JSON 格式返回，不要添加任何其他文字：

[
  {{
    "function": "函数名",
    "parameter": ["参数1", "参数2", "参数3"]
  }}
]

函数原型内容：
{content}"""

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def parse_params_node(state: PipelineState) -> dict[str, Any]:
    """Extract function parameters from parsed_data and persist to DB.

    Flow:
    1. Query parsed_data by doc_id via MCP
    2. Find section_type == "function_prototype" and extract content
    3. Call LLM to extract function names and parameters
    4. Persist parameters to DB via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    version = state.get("version", 0)

    logger.info("ParseParams: received state doc_id=%s for %s v%s", doc_id, operator_name, version)

    if not doc_id:
        logger.warning("ParseParams: no doc_id in state, skipping")
        return {"parameters": [], "error": None}

    try:
        parsed = await _mcp_client.get_parsed_by_doc_id(doc_id)
        if not parsed:
            logger.warning("ParseParams: no parsed_data for doc_id=%s", doc_id)
            return {"parameters": [], "error": None}

        content = _find_function_prototype_content(parsed.get("sections", []))
        if not content:
            logger.warning("ParseParams: no function_prototype section for doc_id=%s", doc_id)
            return {"parameters": [], "error": None}

        functions = await _extract_params_via_llm(content)
        if not functions:
            logger.warning("ParseParams: LLM returned no results for doc_id=%s", doc_id)
            return {"parameters": [], "error": None}

        parameters = _flatten_to_parameters(functions)
        logger.info(
            "ParseParams: extracted %d parameters from %d functions for %s v%s",
            len(parameters),
            len(functions),
            operator_name,
            version,
        )

        await _mcp_client.save_parameters(doc_id, parameters)
        logger.info("ParseParams: persisted %d parameters for doc_id=%s", len(parameters), doc_id)

        return {"parameters": parameters, "error": None}

    except Exception as e:
        logger.exception("ParseParams failed for %s v%s", operator_name, version)
        return {"parameters": [], "error": str(e)}


def _find_function_prototype_content(sections: list[dict]) -> str | None:
    """Find the function_prototype section and return its content."""
    for section in sections:
        if section.get("section_type") == "function_prototype":
            return section.get("content")
    return None


async def _extract_params_via_llm(content: str) -> list[dict]:
    """Call LLM to extract function parameters from content.

    Returns list of {"function": str, "parameter": list[str]}.
    """
    llm = ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )

    prompt = _EXTRACT_PROMPT.format(content=content)
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


def _flatten_to_parameters(functions: list[dict]) -> list[dict]:
    """Convert LLM output to parameter records for DB persistence.

    Each function-parameter pair becomes one record with direction inferred
    from common naming conventions (output params typically contain "output"/"out").
    """
    parameters: list[dict] = []
    for func in functions:
        func_name = func.get("function", "")
        for param_name in func.get("parameter", []):
            direction = "output" if "out" in param_name.lower() else "input"
            parameters.append({
                "function_name": func_name,
                "param_name": param_name,
                "direction": direction,
            })
    return parameters
