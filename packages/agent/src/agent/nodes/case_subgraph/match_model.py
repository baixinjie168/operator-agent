"""Step 1 of GeneratorAgent: match data model for the operator.

Looks up the latest document version for ``operator_name`` via MCP and
loads the assembled ``json_constraints``.  Pure data lookup — no LLM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def case_match_model_node(state: PipelineState) -> dict[str, Any]:
    """Match the data model: fetch json_constraints for the operator from MCP."""
    operator_name = state.get("operator_name", "")
    if not operator_name:
        return {"error": "operator_name is required"}

    logger.info("case_match_model: matching model for %s", operator_name)
    await asyncio.sleep(0.15)

    try:
        constraints = await _mcp_client.get_json_constraints(operator_name)
        if not constraints:
            return {
                "error": f"json_constraints not found for {operator_name}; "
                "run the doc + constraint pipeline first",
                "constraints_raw": None,
            }

        return {"constraints_raw": constraints, "error": None}
    except Exception as e:
        logger.exception("case_match_model failed for %s", operator_name)
        return {"error": str(e), "constraints_raw": None}
