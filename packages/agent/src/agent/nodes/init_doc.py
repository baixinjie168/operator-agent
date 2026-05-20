"""InitDoc node: calls MCP to parse and save document sections.

Placeholder — logic to be filled in later.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)


async def init_doc_node(state: PipelineState) -> dict[str, Any]:
    """Call MCP parse_doc + save_parsed, populate sections and operator_id.

    TODO: implement MCP call logic.
    """
    operator_name = state.get("operator_name", "unknown")
    version = state.get("version", 0)
    logger.info("InitDoc node — placeholder for %s v%s", operator_name, version)
    return {"sections": [], "operator_id": 0}
