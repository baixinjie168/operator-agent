"""TableColumnExtract node: direct extraction from HTML parameter tables.

For parameters documented in table form (GetWorkspaceSize-style 8-column
tables), directly extract shape, dtype_desc, dformat_desc, and
is_support_discontinuous from the corresponding columns.  Zero LLM calls.

Runs in the pipeline before llm_description_extract.  Downstream extract
nodes (shape_extract, dtype_extract, dformat_extract) skip parameters
that already have values set by this node.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.utils.table_parser import (
    detect_table_columns,
    extract_columns_as_json,
    find_param_name_column,
    is_table_form,
    parse_html_tables_with_raw,
)

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# Section types to fetch — only parameter table sections;
# "constraints" is paragraph-style natural language, not tabular.
_SECTION_TYPES = [
    "params_get_workspace",
    "params_execute",
]


async def _fetch_sections_text(doc_id: int) -> str:
    """Fetch and concatenate section content for table parsing."""
    parts: list[str] = []
    for section_type in _SECTION_TYPES:
        section = await _mcp_client.get_section(doc_id, section_type)
        if section and section.get("content"):
            parts.append(section["content"])
    return "\n\n".join(parts)


def _extract_from_sections(
    sections_text: str,
    params: list[dict],
) -> list[dict]:
    """Parse HTML tables from section text and extract values for each param.

    Returns a list of dicts, one per successfully extracted parameter.
    Value fields (shape, dtype_desc, dformat_desc, param_desc, usage_notes)
    are stored as JSON {platform: value} dicts.
    """
    tables, raw_tables = parse_html_tables_with_raw(sections_text)
    if not tables:
        return []

    # Pre-compute column mappings for each table
    table_info: list[tuple[list[list[str]], list[list[str]], dict[str, int], int]] = []
    for i, grid in enumerate(tables):
        if not grid:
            continue
        header = grid[0]
        col_map = detect_table_columns(header)
        if not is_table_form(header):
            continue
        name_idx = find_param_name_column(header)
        raw_grid = raw_tables[i] if i < len(raw_tables) else []
        table_info.append((grid, raw_grid, col_map, name_idx))

    if not table_info:
        return []

    results: list[dict] = []
    for param in params:
        param_name = param.get("param_name", "")
        param_type = param.get("param_type", "")
        function_name = param.get("function_name", "")

        if not param_name:
            continue

        # Try each table until we find a match
        extracted: dict[str, Any] = {}
        for grid, raw_grid, col_map, name_idx in table_info:
            extracted = extract_columns_as_json(
                grid, raw_grid, col_map, param_name, param_type, name_idx
            )
            if extracted:
                break

        if not extracted:
            continue

        record: dict[str, Any] = {
            "function_name": function_name,
            "param_name": param_name,
        }

        # JSON fields: shape, dtype_desc, dformat_desc, param_desc, usage_notes
        for field in ("shape", "dtype_desc", "dformat_desc", "param_desc", "usage_notes"):
            if extracted.get(field):
                record[field] = extracted[field]

        # Non-JSON fields
        if extracted.get("is_support_discontinuous"):
            record["is_support_discontinuous"] = extracted["is_support_discontinuous"]
        if extracted.get("direction"):
            record["direction"] = extracted["direction"]

        # Only add if at least one field was extracted
        if len(record) > 2:
            results.append(record)

    return results


async def _persist_to_db(doc_id: int, results: list[dict]) -> None:
    """Persist extracted table column values to DB via MCP.

    Value fields (shape, dtype_desc, dformat_desc, param_desc, usage_notes)
    are stored as JSON strings: {platform: value}.
    """
    if not results:
        return

    def _updated_count(res: object) -> int:
        """Safely extract 'updated' count from MCP response."""
        if isinstance(res, dict):
            return res.get("updated", 0)
        logger.warning("TableColumnExtract: unexpected MCP response type: %r", res)
        return 0

    # JSON fields: serialize dict → JSON string
    shape_updates = [
        {"function_name": r["function_name"], "param_name": r["param_name"],
         "shape": json.dumps(r["shape"], ensure_ascii=False)}
        for r in results if r.get("shape")
    ]
    dtype_updates = [
        {"function_name": r["function_name"], "param_name": r["param_name"],
         "dtype": json.dumps(r["dtype_desc"], ensure_ascii=False)}
        for r in results if r.get("dtype_desc")
    ]
    dformat_updates = [
        {"function_name": r["function_name"], "param_name": r["param_name"],
         "dformat": json.dumps(r["dformat_desc"], ensure_ascii=False)}
        for r in results if r.get("dformat_desc")
    ]
    desc_updates = [
        {"function_name": r["function_name"], "param_name": r["param_name"],
         "param_desc": json.dumps(r["param_desc"], ensure_ascii=False)}
        for r in results if r.get("param_desc")
    ]
    usage_updates = [
        {"function_name": r["function_name"], "param_name": r["param_name"],
         "usage_notes": json.dumps(r["usage_notes"], ensure_ascii=False)}
        for r in results if r.get("usage_notes")
    ]

    # Non-JSON fields
    disc_updates = [
        {
            "function_name": r["function_name"],
            "param_name": r["param_name"],
            "is_support_discontinuous": r["is_support_discontinuous"],
        }
        for r in results if r.get("is_support_discontinuous")
    ]
    direction_updates = [
        {"function_name": r["function_name"], "param_name": r["param_name"],
         "direction": r["direction"]}
        for r in results if r.get("direction")
    ]

    if shape_updates:
        res = await _mcp_client.update_param_shape(doc_id, shape_updates)
        logger.info("TableColumnExtract: updated shape for %d params", _updated_count(res))

    if dtype_updates:
        res = await _mcp_client.update_param_dtype(doc_id, dtype_updates)
        logger.info("TableColumnExtract: updated dtype for %d params", _updated_count(res))

    if dformat_updates:
        res = await _mcp_client.update_param_dformat(doc_id, dformat_updates)
        logger.info("TableColumnExtract: updated dformat for %d params", _updated_count(res))

    if disc_updates:
        res = await _mcp_client.update_param_attrs(doc_id, disc_updates)
        logger.info("TableColumnExtract: updated is_support_discontinuous for %d params", _updated_count(res))

    if desc_updates:
        res = await _mcp_client.update_param_desc(doc_id, desc_updates)
        logger.info("TableColumnExtract: updated param_desc for %d params", _updated_count(res))

    if direction_updates:
        res = await _mcp_client.update_param_direction(doc_id, direction_updates)
        logger.info("TableColumnExtract: updated direction for %d params", _updated_count(res))

    if usage_updates:
        res = await _mcp_client.update_param_usage_notes(doc_id, usage_updates)
        logger.info("TableColumnExtract: updated usage_notes for %d params", _updated_count(res))


def _merge_into_params(
    original_params: list[dict],
    results: list[dict],
) -> list[dict]:
    """Merge extraction results back into the parameter list.

    JSON fields (shape, dtype_desc, dformat_desc, param_desc, usage_notes)
    are serialized to JSON strings for downstream compatibility.
    """
    if not results:
        return original_params

    result_map: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (r["function_name"], r["param_name"])
        result_map[key] = r

    enriched: list[dict] = []
    for p in original_params:
        key = (p.get("function_name", ""), p.get("param_name", ""))
        update = result_map.get(key)
        if update:
            merged = dict(p)
            # JSON fields: serialize dict → JSON string
            for field in ("shape", "dtype_desc", "dformat_desc", "param_desc", "usage_notes"):
                if update.get(field):
                    merged[field] = json.dumps(update[field], ensure_ascii=False)
            # Also set data_type / data_format for MCP API compatibility
            if update.get("dtype_desc"):
                merged["data_type"] = json.dumps(update["dtype_desc"], ensure_ascii=False)
            if update.get("dformat_desc"):
                merged["data_format"] = json.dumps(update["dformat_desc"], ensure_ascii=False)
            # Non-JSON fields
            if update.get("is_support_discontinuous"):
                merged["is_support_discontinuous"] = update["is_support_discontinuous"]
            if update.get("direction"):
                merged["direction"] = update["direction"]
            enriched.append(merged)
        else:
            enriched.append(p)
    return enriched


async def table_column_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract shape/dtype/dformat/is_support_discontinuous from HTML tables.

    Reads parameters from state, fetches section content from MCP, parses
    HTML tables, and extracts values directly from table columns.

    Returns enriched parameters for downstream nodes.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("TableColumnExtract: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("TableColumnExtract: no doc_id, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("TableColumnExtract: no parameters, skipping")
            return {"error": None}

        sections_text = await _fetch_sections_text(doc_id)
        if not sections_text.strip():
            logger.info("TableColumnExtract: no section content, skipping")
            return {"error": None}

        results = _extract_from_sections(sections_text, params)

        if results:
            await _persist_to_db(doc_id, results)
            logger.info(
                "TableColumnExtract: extracted values for %d/%d params (doc_id=%s)",
                len(results), len(params), doc_id,
            )
        else:
            logger.info("TableColumnExtract: no table-form params found (doc_id=%s)", doc_id)

        # Merge into parameters for downstream consumption
        enriched = _merge_into_params(params, results)
        return {"parameters": enriched, "error": None}

    except Exception as e:
        logger.exception("TableColumnExtract failed for %s", operator_name)
        return {"error": str(e)}
