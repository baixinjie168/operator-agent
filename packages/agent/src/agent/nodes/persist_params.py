"""PersistParams node: saves parsed parameters to the database via MCP."""

from __future__ import annotations

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def persist_params_node(state: PipelineState) -> dict[str, Any]:
    """Save parsed parameters to the database via MCP save_params tool."""
    operator_name = state.get("operator_name", "")
    version = state.get("version", 0)
    parameters = state.get("parameters", [])

    if not parameters:
        logger.info("PersistParams: no parameters to save for %s v%s", operator_name, version)
        return {"error": None}

    try:
        result = await _mcp_client.save_parameters(operator_name, version, parameters)
        saved_count = result.get("saved", 0) if isinstance(result, dict) else 0
        logger.info(
            "PersistParams: saved %d parameters for %s v%s",
            saved_count,
            operator_name,
            version,
        )
        return {"error": None}
    except Exception as e:
        logger.exception("PersistParams failed for %s v%s", operator_name, version)
        return {"error": str(e)}
