"""ExtractExe node: LLM-based description extraction for Execute parameters.

Reuses ``_extract_one`` from ``extract_ws.py`` so the two
branches share identical extraction logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.nodes.context_utils import _is_ws_function
from agent.nodes.llm_description_extract.extract_ws import _extract_one
from agent.nodes.llm_description_extract.state import DescriptionExtractState
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT

logger = logging.getLogger(__name__)


async def extract_exe_node(state: DescriptionExtractState) -> dict[str, Any]:
    """Extract llm_descriptions for Execute (non-GetWorkspaceSize) parameters."""
    sections_text = state.get("exe_sections_text", "")
    parameters = state.get("parameters", [])
    exe_params = [p for p in parameters if not _is_ws_function(p.get("function_name", ""))]

    logger.info(
        "ExtractExe: %d params, %d chars context",
        len(exe_params),
        len(sections_text),
    )

    if not exe_params or not sections_text.strip():
        return {"exe_results": [], "error": None}

    try:
        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _task(p: dict) -> dict | None:
            async with sem:
                return await _extract_one(llm, p, sections_text)

        results = await asyncio.gather(*[_task(p) for p in exe_params])
        valid = [r for r in results if r is not None]

        logger.info("ExtractExe: extracted %d/%d", len(valid), len(exe_params))
        return {"exe_results": valid, "error": None}

    except Exception:
        logger.exception("ExtractExe failed")
        return {"exe_results": [], "error": "extract_exe_failed"}

