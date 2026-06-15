"""ExtractWS node: LLM-based description extraction for GetWorkspaceSize parameters.

Also exposes the shared ``_extract_one`` helper used by ``extract_exe.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.nodes.context_utils import _is_ws_function, extract_param_context
from agent.nodes.llm_description_extract.state import DescriptionExtractState
from agent.prompts import LLM_DESCRIPTION_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_CONCURRENCY_LIMIT = 5

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_INPUT_KEYWORDS = ("输入", "入参", "input", "计算输入")
_OUTPUT_KEYWORDS = ("输出", "出参", "output", "计算输出")


# ---------------------------------------------------------------------------
# Helpers (shared with extract_exe)
# ---------------------------------------------------------------------------

def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


def _parse_direction(raw: str) -> str:
    """Normalize LLM direction output to ``'input'`` / ``'output'`` / ``''``."""
    if not raw:
        return ""
    val = raw.strip().lower().replace("**", "").replace("*", "")
    if any(kw in val for kw in _INPUT_KEYWORDS):
        return "input"
    if any(kw in val for kw in _OUTPUT_KEYWORDS):
        return "output"
    return ""


def _build_discontinuous_json(raw_val: Any, param_type: str) -> str:
    """Build the ``is_support_discontinuous`` JSON string from LLM output."""
    if "tensor" not in param_type.lower():
        return json.dumps({"value": "N/A", "src_text": ""}, ensure_ascii=False)
    if raw_val is None:
        return json.dumps({"value": False, "src_text": ""}, ensure_ascii=False)
    if isinstance(raw_val, bool):
        return json.dumps({"value": raw_val, "src_text": ""}, ensure_ascii=False)
    return json.dumps({"value": False, "src_text": ""}, ensure_ascii=False)


def _parse_llm_response(text: str) -> dict | None:
    """Parse an LLM JSON response, tolerating code fences and surrounding text."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Per-parameter extraction (shared with extract_exe)
# ---------------------------------------------------------------------------

async def _extract_one(
    llm: ChatOpenAI,
    param: dict,
    sections_text: str,
) -> dict | None:
    """Extract description for a single parameter with context pre-filtering.

    This is the core optimisation: instead of sending the full 3000–12 500 char
    section text to the LLM, ``extract_param_context`` slices it down to
    500–2000 chars focused on *param_name*.
    """
    param_name = param.get("param_name", "")
    param_type = param.get("param_type", "")
    function_name = param.get("function_name", "")

    if not sections_text.strip():
        return None

    # ★ Core optimisation: pre-filter context
    context = extract_param_context(sections_text, param_name)

    prompt = LLM_DESCRIPTION_EXTRACT_PROMPT.format(
        param_name=param_name,
        section_content=context,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    parsed = _parse_llm_response(text)
    if parsed is None:
        logger.warning(
            "ExtractWS: failed to parse LLM response for %s.%s",
            function_name,
            param_name,
        )
        return None

    llm_desc = parsed.get("llm_description", "").strip()
    if not llm_desc:
        return None

    direction = _parse_direction(parsed.get("direction", ""))
    src_content = parsed.get("src_content", "").strip()

    # If direction was already set by table_column_extract to a valid value,
    # preserve it.  Only "input" / "output" are authoritative — ignore any
    # other value (e.g. a DB default or placeholder).
    existing_direction = param.get("direction")
    if existing_direction in ("input", "output"):
        direction = existing_direction

    # If is_support_discontinuous was already set by table_column_extract,
    # preserve it; otherwise build from LLM output.
    existing_disc = param.get("is_support_discontinuous")
    if existing_disc:
        disc_json = existing_disc
    else:
        disc_json = _build_discontinuous_json(
            parsed.get("is_support_discontinuous"), param_type
        )

    return {
        "function_name": function_name,
        "param_name": param_name,
        "param_type": param_type,
        "llm_description": llm_desc,
        "src_content": src_content,
        "direction": direction,
        "is_support_discontinuous": disc_json,
        "_context": context,  # retained for validate_results
    }


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

async def extract_ws_node(state: DescriptionExtractState) -> dict[str, Any]:
    """Extract llm_descriptions for GetWorkspaceSize parameters."""
    sections_text = state.get("ws_sections_text", "")
    parameters = state.get("parameters", [])
    ws_params = [p for p in parameters if _is_ws_function(p.get("function_name", ""))]

    logger.info(
        "ExtractWS: %d params, %d chars context",
        len(ws_params),
        len(sections_text),
    )

    if not ws_params or not sections_text.strip():
        return {"ws_results": [], "error": None}

    try:
        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _task(p: dict) -> dict | None:
            async with sem:
                return await _extract_one(llm, p, sections_text)

        results = await asyncio.gather(*[_task(p) for p in ws_params])
        valid = [r for r in results if r is not None]

        logger.info("ExtractWS: extracted %d/%d", len(valid), len(ws_params))
        return {"ws_results": valid, "error": None}

    except Exception:
        logger.exception("ExtractWS failed")
        return {"ws_results": [], "error": "extract_ws_failed"}

