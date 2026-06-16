"""FetchSections node: retrieve section content from MCP, split by ws/exe groups."""

from __future__ import annotations

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.context_utils import _is_ws_function
from agent.nodes.llm_description_extract.state import DescriptionExtractState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_WS_SECTION_TYPES = [
    "params_get_workspace",
    "return_codes_get_workspace",
    "constraints",
]

_EXE_SECTION_TYPES = [
    "params_execute",
    "return_codes_execute",
    "constraints",
]


async def _fetch(doc_id: int, section_types: list[str]) -> str:
    """Fetch and concatenate section content for the given section types."""
    parts: list[str] = []
    for section_type in section_types:
        section = await _mcp_client.get_section(doc_id, section_type)
        if section and section.get("content"):
            parts.append(f"## {section_type}\n{section['content']}")
    return "\n\n".join(parts)


async def fetch_sections_node(state: DescriptionExtractState) -> dict[str, Any]:
    """Fetch section content from MCP, split into ws/exe groups.

    Only fetches sections for groups that actually have parameters, avoiding
    unnecessary MCP calls.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    parameters = state.get("parameters", [])

    logger.info("FetchSections: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("FetchSections: no doc_id, skipping")
        return {"ws_sections_text": "", "exe_sections_text": "", "error": None}

    try:
        ws_params = [p for p in parameters if _is_ws_function(p.get("function_name", ""))]
        exe_params = [p for p in parameters if not _is_ws_function(p.get("function_name", ""))]

        ws_text = await _fetch(doc_id, _WS_SECTION_TYPES) if ws_params else ""
        exe_text = await _fetch(doc_id, _EXE_SECTION_TYPES) if exe_params else ""

        logger.info(
            "FetchSections: ws=%d chars, exe=%d chars (doc_id=%s)",
            len(ws_text),
            len(exe_text),
            doc_id,
        )

        return {
            "ws_sections_text": ws_text,
            "exe_sections_text": exe_text,
            "error": None,
        }

    except Exception:
        logger.exception("FetchSections failed for %s", operator_name)
        return {"ws_sections_text": "", "exe_sections_text": "", "error": "fetch_failed"}

