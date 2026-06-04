"""MergeRelations node: merge and deduplicate relations from ws and exe sections."""

import logging
from typing import Any

from agent.nodes.param_relation_extract.state import RelationExtractState

logger = logging.getLogger(__name__)


def _dedup_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, frozenset[str], str], dict[str, Any]] = {}
    for r in relations:
        key = (
            r.get("relation_type", ""),
            frozenset(r.get("params", [])),
            r.get("precondition", "无"),
        )
        existing = seen.get(key)
        if existing is None:
            seen[key] = r
        else:
            if len(r.get("source_citation", "")) > len(existing.get("source_citation", "")):
                seen[key] = r
    return list(seen.values())


async def merge_relations_node(state: RelationExtractState) -> dict[str, Any]:
    ws_relations = state.get("ws_relations", [])
    exe_relations = state.get("exe_relations", [])

    all_relations = ws_relations + exe_relations
    merged = _dedup_relations(all_relations)

    logger.info(
        "MergeRelations: ws=%d + exe=%d = %d total, %d after dedup",
        len(ws_relations),
        len(exe_relations),
        len(all_relations),
        len(merged),
    )

    return {"merged_relations": merged, "error": None}
