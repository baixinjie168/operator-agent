"""ReturnCodeExtract node: extract return value error codes from document sections via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import RETURN_CODE_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def return_code_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract return value error codes from document return_codes sections via LLM.

    Fetches return_codes_get_workspace and return_codes_execute sections,
    then calls LLM to extract structured error code data.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ReturnCodeExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ReturnCodeExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        ws_section = await _mcp_client.get_section(doc_id, "return_codes_get_workspace")
        exe_section = await _mcp_client.get_section(doc_id, "return_codes_execute")

        ws_content = ws_section.get("content", "") if ws_section else ""
        exe_content = exe_section.get("content", "") if exe_section else ""

        if not ws_content.strip() and not exe_content.strip():
            logger.info("ReturnCodeExtract: no return_codes sections for doc_id=%s, skipping", doc_id)
            return {"error": None}

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        tasks = []
        if ws_content.strip():
            ws_func_name = f"{operator_name}GetWorkspaceSize"
            tasks.append(_extract_one(sem, llm, ws_func_name, ws_content))
        if exe_content.strip():
            exe_func_name = operator_name
            tasks.append(_extract_one(sem, llm, exe_func_name, exe_content))

        results = await asyncio.gather(*tasks)
        all_codes = []
        for r in results:
            all_codes.extend(r)

        if all_codes:
            result = await _mcp_client.save_return_codes(doc_id, all_codes)
            logger.info(
                "ReturnCodeExtract: saved %d return codes (doc_id=%s)",
                result.get("saved", 0),
                doc_id,
            )

        return {"return_codes": all_codes, "error": None}

    except Exception as e:
        logger.exception("ReturnCodeExtract failed for %s", operator_name)
        return {"error": str(e)}


async def _extract_one(
    sem: asyncio.Semaphore,
    llm: ChatOpenAI,
    function_name: str,
    section_content: str,
) -> list[dict]:
    """Extract return codes from a single section."""
    async with sem:
        prompt = RETURN_CODE_EXTRACT_PROMPT.format(section_content=section_content)
        response = await llm.ainvoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_response(text)
        if parsed is None:
            return []
        for item in parsed:
            item["function_name"] = function_name
            item["source_citation"] = section_content[:200]
        return parsed


def _parse_json_response(text: str) -> list[dict] | None:
    """Parse LLM JSON response, handling ```json``` blocks and bare JSON."""
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

    logger.warning("ReturnCodeExtract: failed to parse LLM response: %s", text[:200])
    return None


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )
