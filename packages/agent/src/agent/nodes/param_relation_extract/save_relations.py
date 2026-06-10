"""SaveRelations node: persist merged relations to database via MCP."""

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.state import RelationExtractState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


def _saved_count(res: object) -> int:
    """Safely extract 'saved' count from MCP response."""
    if isinstance(res, dict):
        return res.get("saved", 0)
    logger.warning("SaveRelations: unexpected MCP response type: %r", res)
    return 0


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

    # Log coverage report for monitoring
    report = state.get("coverage_report")
    if report:
        logger.info(
            "CoverageReport: doc_id=%s coverage=%s uncovered=%s rounds=%d total=%d",
            doc_id,
            report.get("coverage", ""),
            report.get("uncovered_params", []),
            report.get("total_rounds", 0),
            report.get("total", 0),
        )

    try:
        result = await _mcp_client.save_param_relations(doc_id, merged)
        logger.info(
            "SaveRelations: saved %d relations (doc_id=%s)",
            _saved_count(result),
            doc_id,
        )
        return {"error": None}

    except Exception:
        logger.exception("SaveRelations failed for %s", operator_name)
        return {"error": "save_relations_failed"}
