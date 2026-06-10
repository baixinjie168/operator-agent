"""SaveDescriptions node: persist descriptions to DB and write enriched params back.

The node writes extraction results to the MCP-managed database (stripping
internal ``_``-prefixed fields) and merges them into ``parameters`` so the
parent graph's downstream nodes see the enriched descriptions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.llm_description_extract.state import DescriptionExtractState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def _build_enriched_params(
    original_params: list[dict],
    updates: list[dict],
) -> list[dict]:
    """Merge LLM extraction results back into the original parameter list."""
    update_map: dict[tuple[str, str], dict] = {}
    for u in updates:
        key = (u["function_name"], u["param_name"])
        update_map[key] = u

    enriched: list[dict] = []
    for p in original_params:
        key = (p.get("function_name", ""), p.get("param_name", ""))
        update = update_map.get(key)
        if update:
            merged = dict(p)
            merged["llm_description"] = update.get("llm_description", "")
            merged["src_content"] = update.get("src_content", "")
            merged["direction"] = update.get("direction", "")
            merged["is_support_discontinuous"] = update.get(
                "is_support_discontinuous",
                json.dumps({"value": "N/A", "src_text": ""}, ensure_ascii=False),
            )
            # Audit record: serialize to JSON string for DB storage
            audit = update.get("description_audit")
            if audit:
                merged["description_audit"] = json.dumps(
                    audit, ensure_ascii=False
                )
            enriched.append(merged)
        else:
            enriched.append(p)
    return enriched


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

async def save_descriptions_node(state: DescriptionExtractState) -> dict[str, Any]:
    """Persist descriptions to DB and merge enriched parameters back into state.

    Returns ``{"parameters": enriched}`` so LangGraph propagates the enriched
    list to the parent graph's ``parameters`` field.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    ws_results = state.get("ws_results", [])
    exe_results = state.get("exe_results", [])
    all_updates = ws_results + exe_results
    original_params = state.get("parameters", [])

    logger.info(
        "SaveDescriptions: doc_id=%s, %d updates for %s",
        doc_id,
        len(all_updates),
        operator_name,
    )

    if not doc_id:
        logger.warning("SaveDescriptions: no doc_id, skipping")
        return {"parameters": original_params, "error": None}

    try:
        # Strip internal fields (prefixed with _) before persisting
        db_updates = [
            {k: v for k, v in u.items() if not k.startswith("_")}
            for u in all_updates
        ]

        if db_updates:
            result = await _mcp_client.update_llm_descriptions(doc_id, db_updates)
            logger.info(
                "SaveDescriptions: updated %d params (doc_id=%s)",
                result.get("updated", 0),
                doc_id,
            )

        # Merge enriched parameters for downstream consumption
        enriched = _build_enriched_params(original_params, all_updates)

        return {"parameters": enriched, "error": None}

    except Exception:
        logger.exception("SaveDescriptions failed for %s", operator_name)
        return {"parameters": original_params, "error": "save_failed"}

