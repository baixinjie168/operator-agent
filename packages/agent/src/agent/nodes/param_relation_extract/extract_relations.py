"""ExtractRelations nodes: LLM-based relation extraction from section content."""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.nodes.param_relation_extract.prompts import (
    RELATION_EXTRACT_PROMPT,
    RELATION_TYPE_DEFINITIONS,
    format_implicit_params_context,
)
from agent.nodes.param_relation_extract.state import RelationExtractState

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


def _parse_relations_response(text: str) -> list[dict[str, Any]]:
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

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("ExtractRelations: failed to parse LLM response as JSON: %s", text[:200])
    return []


async def _extract_relations(
    section_content: str,
    llm: ChatOpenAI | None = None,
    implicit_params: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not section_content.strip():
        return []

    if llm is None:
        llm = _create_llm()
    prompt = RELATION_EXTRACT_PROMPT.format(
        section_content=section_content,
        relation_types=RELATION_TYPE_DEFINITIONS,
        implicit_params_context=format_implicit_params_context(implicit_params or []),
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_relations_response(text.strip())


async def extract_ws_node(state: RelationExtractState) -> dict[str, Any]:
    from agent.nodes.param_relation_extract.agent_loop import extract_relations_agent

    content = state.get("ws_section_content", "")
    param_names = state.get("param_names", [])
    implicit_params = state.get("implicit_params", [])
    logger.info(
        "ExtractWS-Agent: extracting from %d chars, %d params",
        len(content),
        len(param_names),
    )

    if not content.strip():
        return {"ws_relations": [], "coverage_report": {"ws": {}}, "error": None}

    llm = _create_llm()
    try:
        relations, report = await extract_relations_agent(
            content, param_names, llm, implicit_params=implicit_params,
        )
        logger.info(
            "ExtractWS-Agent: %d relations, coverage=%s",
            len(relations),
            report.get("coverage", ""),
        )
        return {
            "ws_relations": relations,
            "coverage_report": {"ws": report},
            "error": None,
        }
    except Exception:
        logger.exception("ExtractWS-Agent failed")
        return {"ws_relations": [], "coverage_report": {"ws": {}}, "error": "extract_ws_agent_failed"}


async def extract_exe_node(state: RelationExtractState) -> dict[str, Any]:
    from agent.nodes.param_relation_extract.agent_loop import extract_relations_agent

    content = state.get("exe_section_content", "")
    param_names = state.get("param_names", [])
    implicit_params = state.get("implicit_params", [])
    logger.info(
        "ExtractExe-Agent: extracting from %d chars, %d params",
        len(content),
        len(param_names),
    )

    if not content.strip():
        return {"exe_relations": [], "coverage_report": {"exe": {}}, "error": None}

    llm = _create_llm()
    try:
        relations, report = await extract_relations_agent(
            content, param_names, llm, implicit_params=implicit_params,
        )
        logger.info(
            "ExtractExe-Agent: %d relations, coverage=%s",
            len(relations),
            report.get("coverage", ""),
        )
        return {
            "exe_relations": relations,
            "coverage_report": {"exe": report},
            "error": None,
        }
    except Exception:
        logger.exception("ExtractExe-Agent failed")
        return {"exe_relations": [], "coverage_report": {"exe": {}}, "error": "extract_exe_agent_failed"}
