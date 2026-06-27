"""FetchSections node: retrieve section content from MCP."""

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.state import RelationExtractState
from agent.utils.param_validators import EXCLUDED_PARAMS as _EXCLUDED_PARAMS
from agent.utils.section_utils import resolve_ws_exe_content

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def fetch_sections_node(state: RelationExtractState) -> dict[str, Any]:
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("FetchSections: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("FetchSections: no doc_id, skipping")
        return {
            "ws_section_content": "",
            "exe_section_content": "",
            "param_names": [],
            "implicit_params": [],
            "error": None,
        }

    try:
        # resolve_ws_exe_content centralises the single-function exe->ws
        # promotion (params_get_workspace empty -> use params_execute) and
        # the constraints append, shared with constraint_extract Pass 3.
        ws_content, exe_content, _ = await resolve_ws_exe_content(
            _mcp_client, doc_id,
        )

        # Query parameter names for agent loop coverage checks
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        param_names = [
            p["param_name"]
            for p in params
            if p.get("param_name") and p["param_name"] not in _EXCLUDED_PARAMS
        ]

        logger.info(
            "FetchSections: ws=%d chars, exe=%d chars, %d params (doc_id=%s)",
            len(ws_content),
            len(exe_content),
            len(param_names),
            doc_id,
        )

        return {
            "ws_section_content": ws_content,
            "exe_section_content": exe_content,
            "param_names": param_names,
            "implicit_params": state.get("implicit_params", []),
            "error": None,
        }

    except Exception:
        logger.exception("FetchSections failed for %s", operator_name)
        return {
            "ws_section_content": "",
            "exe_section_content": "",
            "param_names": [],
            "implicit_params": [],
            "error": "fetch_sections_failed",
        }
