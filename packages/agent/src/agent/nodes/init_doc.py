"""InitDoc node: checks version, saves document, and triggers MCP parsing."""

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def init_doc_node(state: PipelineState) -> dict[str, Any]:
    """Check version, save document via MCP, and populate state for downstream nodes.

    Flow:
    1. Check if document is new/unchanged/updated
    2. If unchanged, populate from existing data and signal early exit
    3. If new/updated, save document and parse via MCP
    """
    operator_name = state.get("operator_name", "unknown")
    content = state.get("content", "")
    content_hash = state.get("content_hash", "")

    try:
        version_info = await _mcp_client.check_version(operator_name, content_hash)
        status = version_info.get("status", "new")
        existing_version = version_info.get("version")

        if status == "unchanged" and existing_version is not None:
            existing = await _mcp_client.get_parsed(operator_name, existing_version)
            if existing:
                doc_id = version_info.get("doc_id", 0)
                logger.info(
                    "Document unchanged: %s v%s, version_info=%s, doc_id=%s",
                    operator_name, existing_version, version_info, doc_id,
                )
                return {
                    "status": "unchanged",
                    "version": existing_version,
                    "doc_id": doc_id,
                    "sections": existing.get("sections", []),
                    "cann_version": existing.get("cann_version"),
                    "error": None,
                }

        # New or updated document — save it
        save_result = await _mcp_client.save_doc(operator_name, content)
        new_version = save_result["version"]
        doc_id = save_result.get("doc_id", 0)

        # Parse the document and persist parsed_data
        parsed = await _mcp_client.parse_doc(content)
        if parsed:
            await _mcp_client.save_parsed(operator_name, new_version, parsed)
            logger.info(
                "Parsed and saved data for %s v%s: %d sections",
                operator_name,
                new_version,
                len(parsed.get("sections", [])),
            )

        logger.info(
            "Document %s saved as v%s (status=%s, doc_id=%s)",
            operator_name, new_version, status, doc_id,
        )
        return {
            "status": status,
            "version": new_version,
            "operator_id": save_result.get("operator_id", 0),
            "doc_id": doc_id,
            "cann_version": parsed.get("cann_version") if parsed else None,
            "sections": [
                {"section_type": s.get("section_type", ""), "heading": s.get("heading", "")}
                for s in parsed.get("sections", [])
            ] if parsed else [],
            "error": None,
        }

    except Exception as e:
        logger.exception("InitDoc failed for %s", operator_name)
        return {"status": "error", "error": str(e)}
