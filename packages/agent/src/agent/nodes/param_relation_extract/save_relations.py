"""SaveRelations node: persist merged relations to database via MCP."""

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.state import RelationExtractState
from agent.runtime.context import get_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_STEP_NAME = "param_relation_extract"
_STEP_LABEL = "参数约束关系提取"


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

    ctx = get_context()

    if not doc_id:
        logger.warning("SaveRelations: no doc_id, skipping")
        return {"error": None}

    if not merged:
        logger.info("SaveRelations: no relations to save for doc_id=%s", doc_id)
        if ctx and ctx.manager:
            span = ctx.manager.open_span(
                run_id=ctx.run_id,
                parent_span_id=ctx.current_span_id,
                span_type=SpanType.NODE,
                name=_STEP_NAME,
            )
            ctx.manager.close_span(ctx.run_id, span, SpanStatus.SUCCESS)
            ctx.manager.emit(EventType.NODE_SUCCESS, ctx.run_id, span, {
                "agent_id": "doc",
                "node_id": _STEP_NAME,
                "message": f"{_STEP_LABEL} 完成",
                "meta": "0 个关系",
                "output": {"merged_relations": []},
            })
        return {"merged_relations": [], "error": None}

    # Log coverage report for monitoring
    report = state.get("coverage_report")
    if report:
        for section, section_report in report.items():
            if not isinstance(section_report, dict):
                continue
            logger.info(
                "CoverageReport: doc_id=%s section=%s coverage=%s uncovered=%s rounds=%d total=%d",
                doc_id,
                section,
                section_report.get("coverage", ""),
                section_report.get("uncovered_params", []),
                section_report.get("total_rounds", 0),
                section_report.get("total", 0),
            )

    try:
        result = await _mcp_client.save_param_relations(doc_id, merged)
        logger.info(
            "SaveRelations: saved %d relations (doc_id=%s)",
            _saved_count(result),
            doc_id,
        )

        if ctx and ctx.manager:
            span = ctx.manager.open_span(
                run_id=ctx.run_id,
                parent_span_id=ctx.current_span_id,
                span_type=SpanType.NODE,
                name=_STEP_NAME,
            )
            ctx.manager.close_span(ctx.run_id, span, SpanStatus.SUCCESS)
            ctx.manager.emit(EventType.NODE_SUCCESS, ctx.run_id, span, {
                "agent_id": "doc",
                "node_id": _STEP_NAME,
                "message": f"{_STEP_LABEL} 完成",
                "meta": f"{len(merged)} 个关系",
                "output": {"merged_relations": merged},
            })

        return {"merged_relations": merged, "error": None}

    except Exception:
        logger.exception("SaveRelations failed for %s", operator_name)
        return {"error": "save_relations_failed"}
