"""SaveRelations node: persist merged relations to database via MCP."""

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.state import RelationExtractState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def save_relations_node(state: RelationExtractState) -> dict[str, Any]:
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    merged = state.get("merged_relations", [])

    logger.info("SaveRelations: doc_id=%s for %s, %d relations", doc_id, operator_name, len(merged))

    if not doc_id:
        logger.warning("SaveRelations: no doc_id, skipping")
        return {"error": None}

    if not merged:
        logger.info("SaveRelations: no relations to save for doc_id=%s", doc_id)
        return {"error": None}

    try:
        result = await _mcp_client.save_param_relations(doc_id, merged)
        logger.info(
            "SaveRelations: saved %d relations (doc_id=%s)",
            result.get("saved", 0),
            doc_id,
        )
        return {"error": None}

    except Exception:
        logger.exception("SaveRelations failed for %s", operator_name)
        return {"error": "save_relations_failed"}
