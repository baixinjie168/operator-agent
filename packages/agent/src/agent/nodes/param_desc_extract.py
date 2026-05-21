"""ParamDescExtract node: extract parameter descriptions via LLM and update DB."""

import asyncio
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import PARAM_DESC_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5


async def param_desc_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract detailed descriptions for each parameter via LLM and update DB.

    Flow:
    1. Query parameters by doc_id via MCP
    2. Find the params_get_workspace section content from parsed data
    3. For each parameter, call LLM concurrently to extract description
    4. Batch update descriptions via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ParamDescExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ParamDescExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        if not params:
            logger.info("ParamDescExtract: no parameters for doc_id=%s, skipping", doc_id)
            return {"error": None}

        # Get params_get_workspace section content from parsed data
        parsed = await _mcp_client.get_parsed_by_doc_id(doc_id)
        doc_content = _find_params_section_content(parsed.get("sections", [])) if parsed else ""
        if not doc_content:
            logger.warning("ParamDescExtract: no section content for doc_id=%s", doc_id)
            return {"error": None}

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict:
            async with sem:
                desc = await _extract_desc(llm, param["param_name"], doc_content)
                return {
                    "function_name": param["function_name"],
                    "param_name": param["param_name"],
                    "description": desc,
                    "usage_notes": "",
                    "dtype_desc": "",
                    "dformat_desc": "",
                    "shape": "",
                    "memory_desc": "",
                }

        updates = await asyncio.gather(*[_extract_one(p) for p in params])
        updates = [u for u in updates if u["description"]]

        if updates:
            result = await _mcp_client.update_param_descriptions(doc_id, updates)
            logger.info(
                "ParamDescExtract: updated %d/%d parameters for doc_id=%s",
                result.get("updated", 0),
                len(params),
                doc_id,
            )
        else:
            logger.info("ParamDescExtract: no descriptions extracted for doc_id=%s", doc_id)

        return {"error": None}

    except Exception as e:
        logger.exception("ParamDescExtract failed for %s", operator_name)
        return {"error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_desc(llm: ChatOpenAI, param_name: str, content: str) -> str:
    """Call LLM to extract description for a single parameter."""
    prompt = PARAM_DESC_EXTRACT_PROMPT.format(param_name=param_name, content=content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()


def _find_params_section_content(sections: list[dict]) -> str | None:
    """Find the params_get_workspace section(s) and return concatenated content."""
    parts: list[str] = []
    for section in sections:
        if section.get("section_type") == "params_get_workspace":
            content = section.get("content", "")
            if content:
                parts.append(content)
    return "\n\n".join(parts) if parts else None
