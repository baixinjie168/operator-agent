"""ConstraintAssemble node: merge parallel outputs + bool narrowing + batch write.

Merges dimensions_map, allowed_range_map, and attrs_map into the final
param_constraint JSON for each (parameter x platform), then batch-updates
the DB via MCP.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.build_param_constraint._helpers import _extract_enum_from_text
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.utils.param_validators import is_bool_type

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# Matches:  param_name.range_value == True  /  param_name.range_value == False
_BOOL_RANGE_RE = re.compile(
    r"\b(\w+)\.range_value\s*==\s*(True|False)\b"
)


def _narrow_bool_allowed_range(
    ar_map: dict[str, list],
    relations: list[dict],
    bool_params: list[dict],
) -> int:
    """Narrow bool [True, False] allowed range using existing constraints.

    Scans relations for expressions like <param>.range_value == <True|False>
    and replaces default [True, False] with the single constrained value.
    """
    if not relations or not bool_params:
        return 0

    constrained: dict[str, bool] = {}
    for rel in relations:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        expr = obj.get("expr", "") or ""
        if not expr:
            continue
        fn = rel.get("function_name", "")
        for m in _BOOL_RANGE_RE.finditer(expr):
            pname = m.group(1)
            val = m.group(2) == "True"
            constrained[f"{fn}::{pname}"] = val

    if not constrained:
        return 0

    narrowed = 0
    for p in bool_params:
        fn = p.get("function_name", "")
        pname = p.get("param_name", "")
        key = f"{fn}::{pname}"
        ar_data = ar_map.get(key)
        # Support both old format (list) and new format (dict with type+value)
        if isinstance(ar_data, dict):
            if ar_data.get("value") == [True, False] and key in constrained:
                ar_data["value"] = [constrained[key]]
                narrowed += 1
                logger.debug(
                    "ConstraintAssemble: narrowed %s.%s allowed_range to [%s]",
                    fn, pname, constrained[key],
                )
        elif ar_data == [True, False] and key in constrained:
            ar_map[key] = {"type": "range", "value": [constrained[key]]}
            narrowed += 1
            logger.debug(
                "ConstraintAssemble: narrowed %s.%s allowed_range to [%s]",
                fn, pname, constrained[key],
            )
    return narrowed


async def constraint_assemble_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Merge parallel node outputs into final constraint JSON and batch-update DB."""
    params = state.get("params", [])
    supported_platforms = state.get("supported_platforms", [])
    dimensions_map = state.get("dimensions_map", {})
    ar_map = state.get("allowed_range_map", {})
    attrs_map = state.get("attrs_map", {})
    param_relations = state.get("param_relations", [])
    doc_id = state.get("doc_id", 0)

    if not params:
        return {"error": None}

    # Step 4b: bool narrowing
    bool_params = [p for p in params if is_bool_type(p.get("param_type", ""))]
    if bool_params and param_relations:
        try:
            narrowed = _narrow_bool_allowed_range(ar_map, param_relations, bool_params)
            if narrowed:
                logger.info(
                    "ConstraintAssemble: narrowed bool allowed_range for %d params",
                    narrowed,
                )
        except Exception:
            logger.warning(
                "ConstraintAssemble: failed to narrow bool ranges",
                exc_info=True,
            )

    # Assemble constraint JSON
    updates: list[dict] = []
    for param in params:
        pname = param["param_name"]
        fn_name = param["function_name"]
        constraint: dict[str, Any] = {}

        for plat in supported_platforms:
            attr_key = f"{fn_name}::{pname}::{plat}"
            map_key = f"{fn_name}::{pname}"

            attrs = dict(attrs_map.get(attr_key, {}))
            shape_raw = attrs.pop("_shape_raw", "")
            is_tensor = attrs.pop("_is_tensor", False)

            # dimensions
            dims_key = f"{fn_name}::{pname}::{shape_raw}"
            dimensions_value = dimensions_map.get(dims_key, [])

            # allowed_range_value (with usage_notes enum fallback)
            ar_data = ar_map.get(map_key, {"type": "range", "value": []})
            ar_value = ar_data.get("value", []) if isinstance(ar_data, dict) else ar_data
            ar_type = ar_data.get("type", "range") if isinstance(ar_data, dict) else "range"
            usage_val = attrs.get("usage_notes", {}).get("value", "")
            if not ar_value and usage_val and not is_tensor:
                enum_values = _extract_enum_from_text(usage_val)
                if enum_values:
                    ar_value = enum_values

            # usage_notes is internal-only (used above for enum fallback and
            # elsewhere for null-pointer dtype=N/A detection); drop it from
            # the final output JSON.
            attrs.pop("usage_notes", None)

            constraint[plat] = {
                **attrs,
                "dimensions": {"value": dimensions_value, "src_text": shape_raw},
                "allowed_range_value": {"value": ar_value, "type": ar_type, "src_text": ""},
            }

        updates.append({
            "function_name": fn_name,
            "param_name": pname,
            "param_constraint": json.dumps(constraint, ensure_ascii=False),
        })

    # Batch update DB
    if updates:
        result = await _mcp_client.update_param_constraint(doc_id, updates)
        logger.info(
            "ConstraintAssemble: updated %d/%d params (doc_id=%s)",
            result.get("updated", 0), len(updates), doc_id,
        )

    return {"error": None}
