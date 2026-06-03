"""ExtractRelations nodes: LLM-based relation extraction from section content."""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.nodes.param_relation_extract.prompts import RELATION_EXTRACT_PROMPT
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


async def _extract_relations(section_content: str) -> list[dict[str, Any]]:
    if not section_content.strip():
        return []

    llm = _create_llm()
    prompt = RELATION_EXTRACT_PROMPT.format(section_content=section_content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_relations_response(text.strip())


async def extract_ws_node(state: RelationExtractState) -> dict[str, Any]:
    content = state.get("ws_section_content", "")
    logger.info("ExtractWS: extracting from %d chars", len(content))

    try:
        relations = await _extract_relations(content)
        logger.info("ExtractWS: found %d relations", len(relations))
        return {"ws_relations": relations, "error": None}
    except Exception:
        logger.exception("ExtractWS failed")
        return {"ws_relations": [], "error": "extract_ws_failed"}


async def extract_exe_node(state: RelationExtractState) -> dict[str, Any]:
    content = state.get("exe_section_content", "")
    logger.info("ExtractExe: extracting from %d chars", len(content))

    try:
        relations = await _extract_relations(content)
        logger.info("ExtractExe: found %d relations", len(relations))
        return {"exe_relations": relations, "error": None}
    except Exception:
        logger.exception("ExtractExe failed")
        return {"exe_relations": [], "error": "extract_exe_failed"}
