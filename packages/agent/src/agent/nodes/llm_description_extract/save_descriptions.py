"""SaveDescriptions node: persist descriptions to DB and write enriched params back.

The node writes extraction results to the MCP-managed database (stripping
internal ``_``-prefixed fields and bulky audit records) and merges them
into ``parameters`` so the parent graph's downstream nodes see the
enriched descriptions.

Reliability features:
- **Batched writes**: updates are split into chunks of ``_BATCH_SIZE`` to
  keep each MCP payload small (avoids stdio / SQLite limits).
- **Retry**: each batch is retried up to ``_MAX_RETRIES`` times on failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.llm_description_extract.state import DescriptionExtractState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_BATCH_SIZE = 5
_MAX_RETRIES = 2
_RETRY_DELAY = 1.0  # seconds


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
            enriched.append(merged)
        else:
            enriched.append(p)
    return enriched


# ---------------------------------------------------------------------------
# Batched save with retry
# ---------------------------------------------------------------------------

async def _save_batch(batch: list[dict], doc_id: int) -> int:
    """Save a single batch of updates with retry logic.

    Returns the number of rows updated.
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            result = await _mcp_client.update_llm_descriptions(doc_id, batch)
            return result.get("updated", 0)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "SaveDescriptions: batch save failed (attempt %d/%d): %s — retrying",
                    attempt + 1, _MAX_RETRIES + 1, exc,
                )
                await asyncio.sleep(_RETRY_DELAY)
            else:
                raise
    # Should not be reached, but satisfies type checker
    raise last_error  # type: ignore[misc]


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
        # Strip internal fields (prefixed with _) and bulky audit records
        # before persisting.  description_audit is a large nested dict that
        # inflates the MCP payload and can cause SQLite type errors when
        # stored as-is.
        db_updates = [
            {
                k: v
                for k, v in u.items()
                if not k.startswith("_") and k != "description_audit"
            }
            for u in all_updates
        ]

        # Save in batches to keep each MCP payload small
        if db_updates:
            total_updated = 0
            for i in range(0, len(db_updates), _BATCH_SIZE):
                batch = db_updates[i : i + _BATCH_SIZE]
                updated = await _save_batch(batch, doc_id)
                total_updated += updated
            logger.info(
                "SaveDescriptions: updated %d/%d params in %d batch(es) (doc_id=%s)",
                total_updated,
                len(db_updates),
                (len(db_updates) + _BATCH_SIZE - 1) // _BATCH_SIZE,
                doc_id,
            )

        # Merge enriched parameters for downstream consumption
        enriched = _build_enriched_params(original_params, all_updates)

        return {"parameters": enriched, "error": None}

    except Exception:
        logger.exception("SaveDescriptions failed for %s", operator_name)
        return {"parameters": original_params, "error": "save_failed"}

