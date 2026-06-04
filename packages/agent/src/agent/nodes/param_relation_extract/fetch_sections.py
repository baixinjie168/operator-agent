"""FetchSections node: retrieve section content from MCP."""

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.state import RelationExtractState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def fetch_sections_node(state: RelationExtractState) -> dict[str, Any]:
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("FetchSections: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("FetchSections: no doc_id, skipping")
        return {"ws_section_content": "", "exe_section_content": "", "error": None}

    try:
        ws_section = await _mcp_client.get_section(doc_id, "params_get_workspace")
        exe_section = await _mcp_client.get_section(doc_id, "params_execute")
        constraints_section = await _mcp_client.get_section(doc_id, "constraints")

        ws_content = ws_section.get("content", "") if ws_section else ""
        exe_content = exe_section.get("content", "") if exe_section else ""

        if constraints_section and constraints_section.get("content"):
            ws_content += "\n\n---\n## 约束说明\n" + constraints_section["content"]

        logger.info(
            "FetchSections: ws=%d chars, exe=%d chars (doc_id=%s)",
            len(ws_content),
            len(exe_content),
            doc_id,
        )

        return {
            "ws_section_content": ws_content,
            "exe_section_content": exe_content,
            "error": None,
        }

    except Exception:
        logger.exception("FetchSections failed for %s", operator_name)
        return {"ws_section_content": "", "exe_section_content": "", "error": "fetch_sections_failed"}
