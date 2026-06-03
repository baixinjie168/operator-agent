"""SrcContentExtract node: extract original source text for each parameter via LLM."""

import asyncio
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import SRC_CONTENT_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5


def _parse_src_content(raw: str) -> str:
    """Clean up LLM source content output.

    Returns empty string if the content is essentially empty or "(无)".
    """
    if not raw or raw.strip() in ("（无）", "(无)", "无"):
        return ""
    return raw.strip()


async def src_content_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract original source text fragments for each parameter via LLM.

    Flow:
    1. Query parameters by doc_id via MCP
    2. Fetch both params_get_workspace and params_execute section content
    3. Route GetWorkspaceSize params → ws_content, Execute params → exe_content
    4. Call LLM concurrently (limit 5) for each parameter to extract source text
    5. Batch update src_content via MCP
    6. Return parameters with src_content to state
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("SrcContentExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("SrcContentExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        if not params:
            logger.info("SrcContentExtract: no parameters for doc_id=%s, skipping", doc_id)
            return {"error": None}

        ws_section = await _mcp_client.get_section(doc_id, "params_get_workspace")
        ws_content = ws_section.get("content", "") if ws_section else ""

        exe_section = await _mcp_client.get_section(doc_id, "params_execute")
        exe_content = exe_section.get("content", "") if exe_section else ""

        ws_params = [p for p in params if p.get("function_name", "").endswith("GetWorkspaceSize")]
        exe_params = [p for p in params if not p.get("function_name", "").endswith("GetWorkspaceSize")]

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _extract_one(param: dict, content: str) -> dict | None:
            async with sem:
                raw = await _extract_src(llm, param["param_name"], content)
                src_content = _parse_src_content(raw)
                if not src_content:
                    return None
                return {
                    "function_name": param["function_name"],
                    "param_name": param["param_name"],
                    "param_type": param.get("param_type", ""),
                    "src_content": src_content,
                }

        all_updates: list[dict] = []

        if ws_params:
            if ws_content:
                results = await asyncio.gather(*[_extract_one(p, ws_content) for p in ws_params])
                all_updates.extend(r for r in results if r is not None)
            else:
                logger.warning(
                    "SrcContentExtract: params_get_workspace section not found for doc_id=%s, "
                    "skipping %d GetWorkspaceSize parameters",
                    doc_id,
                    len(ws_params),
                )

        if exe_params:
            if exe_content:
                results = await asyncio.gather(*[_extract_one(p, exe_content) for p in exe_params])
                all_updates.extend(r for r in results if r is not None)
            else:
                logger.warning(
                    "SrcContentExtract: params_execute section not found for doc_id=%s, skipping %d Execute parameters",
                    doc_id,
                    len(exe_params),
                )

        if all_updates:
            result = await _mcp_client.update_param_src_content(doc_id, all_updates)
            logger.info(
                "SrcContentExtract: updated src_content for %d/%d parameters (doc_id=%s)",
                result.get("updated", 0),
                len(params),
                doc_id,
            )
        else:
            logger.info("SrcContentExtract: no source content extracted for doc_id=%s", doc_id)

        return {"parameters": all_updates, "error": None}

    except Exception as e:
        logger.exception("SrcContentExtract failed for %s", operator_name)
        return {"error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_src(llm: ChatOpenAI, param_name: str, content: str) -> str:
    """Call LLM to extract source content for a single parameter."""
    prompt = SRC_CONTENT_EXTRACT_PROMPT.format(param_name=param_name, content=content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()
