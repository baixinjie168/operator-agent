"""ParamDescExtract node: extract parameter markdown descriptions from src_content via LLM."""

import asyncio
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import PARAM_DESC_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_DIRECTION_RE = re.compile(
    r"\|\s*输入\s*/\s*输出\s*\|\s*(输入|输出|入参|出参|计算输入|计算输出)\s*\|"
)
_DIRECTION_LOOSE_RE = re.compile(r"输入\s*/\s*输出\s*\|\s*(.+?)\s*\|")

_INPUT_KEYWORDS = ("输入", "入参", "input")
_OUTPUT_KEYWORDS = ("输出", "出参", "output")


def _parse_direction(desc: str) -> str:
    """Extract direction from the LLM-generated markdown table.

    Matches rows like ``| 输入/输出 | 输入 |`` or ``| 输入/输出 | 输出 |``
    and maps them to ``"input"`` / ``"output"``.

    Also handles variants:
    - ``| 输入/输出 | 入参 |`` / ``| 输入/输出 | 出参 |``
    - ``| 输入/输出 | 计算输入 |`` / ``| 输入/输出 | 计算输出 |``
    - ``| 输入/输出 | 输入（input） |``
    - ``| 输入/输出 | input |``
    - ``| 输入/输出 | **输入** |``
    """
    m = _DIRECTION_RE.search(desc)
    if m:
        val = m.group(1)
        if val in ("输入", "入参", "计算输入"):
            return "input"
        return "output"

    loose = _DIRECTION_LOOSE_RE.search(desc)
    if loose:
        val = loose.group(1).strip().lower()
        val = val.replace("**", "").replace("*", "")
        if any(kw in val for kw in _INPUT_KEYWORDS):
            return "input"
        if any(kw in val for kw in _OUTPUT_KEYWORDS):
            return "output"

    return ""


async def param_desc_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract structured markdown descriptions from src_content via LLM.

    Reads parameters (with src_content) from state, populated by the
    upstream src_content_extract_node. For each parameter with non-empty
    src_content, calls LLM to produce a structured markdown table.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    parameters = state.get("parameters", [])

    logger.info("ParamDescExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ParamDescExtract: no doc_id in state, skipping")
        return {"error": None}

    with_src = [p for p in parameters if p.get("src_content")]
    if not with_src:
        logger.info("ParamDescExtract: no parameters with src_content for doc_id=%s, skipping", doc_id)
        return {"error": None}

    try:
        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict | None:
            async with sem:
                desc = await _extract_desc(llm, param["param_name"], param["src_content"])
                if not desc:
                    return None
                direction = _parse_direction(desc)
                return {
                    "function_name": param["function_name"],
                    "param_name": param["param_name"],
                    "param_type": param.get("param_type", ""),
                    "direction": direction,
                    "src_content": param.get("src_content", ""),
                    "description": desc,
                    "data_type": "",
                    "data_format": "",
                    "shape": "",
                }

        results = await asyncio.gather(*[_extract_one(p) for p in with_src])
        all_updates = [r for r in results if r is not None and r.get("description")]

        if all_updates:
            result = await _mcp_client.update_param_descriptions(doc_id, all_updates)
            logger.info(
                "ParamDescExtract: updated %d/%d parameters for doc_id=%s",
                result.get("updated", 0),
                len(with_src),
                doc_id,
            )
        else:
            logger.info("ParamDescExtract: no descriptions extracted for doc_id=%s", doc_id)

        return {"parameters": all_updates, "error": None}

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


async def _extract_desc(llm: ChatOpenAI, param_name: str, src_content: str) -> str:
    """Call LLM to extract markdown table from the parameter's source content."""
    prompt = PARAM_DESC_EXTRACT_PROMPT.format(param_name=param_name, src_content=src_content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()
