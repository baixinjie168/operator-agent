"""ConstraintAssemble node: merge parallel outputs into param_constraint JSON.

Simplified: bool narrowing and ar_map (allowed_range_value) handling have
been removed.  allowed_range_value is now filled by assemble_result from
param_relations.  This node only merges attrs + dims into param_constraint
JSON and batch-updates the DB via MCP.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.utils.platform_utils import expand_common_in_constraint

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def constraint_assemble_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Merge parallel node outputs into final constraint JSON and batch-update DB.

    Only merges attrs_map + dimensions_map → param_constraint JSON.
    allowed_range_value is left empty (filled by assemble_result from param_relations).
    """
    params = state.get("params", [])
    supported_platforms = state.get("supported_platforms", [])
    dimensions_map = state.get("dimensions_map", {})
    attrs_map = state.get("attrs_map", {})
    doc_id = state.get("doc_id", 0)

    if not params:
        return {"error": None}

    # Empty-platform fallback: use "common" so constraint JSON is still built.
    platforms_to_build = supported_platforms or ["common"]

    # Assemble constraint JSON
    updates: list[dict] = []
    for param in params:
        pname = param["param_name"]
        fn_name = param["function_name"]
        constraint: dict[str, Any] = {}

        for plat in platforms_to_build:
            attr_key = f"{fn_name}::{pname}::{plat}"
            map_key = f"{fn_name}::{pname}"

            attrs = dict(attrs_map.get(attr_key, {}))
            shape_raw = attrs.pop("_shape_raw", "")
            is_tensor = attrs.pop("_is_tensor", False)

            # dimensions
            dims_key = f"{fn_name}::{pname}::{shape_raw}"
            dimensions_value = dimensions_map.get(dims_key, [])

            # usage_notes is internal-only; drop from final output
            attrs.pop("usage_notes", None)

            constraint[plat] = {
                **attrs,
                "dimensions": {"value": dimensions_value, "src_text": shape_raw},
                # allowed_range_value is now filled by assemble_result
                # from param_relations, not here.
                "allowed_range_value": {"value": [], "type": "range", "src_text": ""},
            }

        # Expand "common" key to per-platform entries before DB save
        expand_common_in_constraint(constraint, platforms_to_build)

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

    # Build per-param validation_results for the frontend
    validation_results: list[dict] = []
    for param in params:
        pname = param.get("param_name", "")
        fn_name = param.get("function_name", "")

        # Find this param's assembled constraint
        assembled: dict | None = None
        for u in updates:
            if u["function_name"] == fn_name and u["param_name"] == pname:
                try:
                    assembled = json.loads(u["param_constraint"])
                except (json.JSONDecodeError, TypeError):
                    assembled = None
                break

        dim_struct_failed: list[str] = []
        dim_align_failed: list[str] = []
        range_struct_failed: list[str] = []
        parsed_dims_list: list = []
        any_passed = False
        has_validation = bool(platforms_to_build) and assembled is not None

        if assembled is not None:
            for plat in platforms_to_build:
                plat_data = assembled.get(plat, {})
                dims = plat_data.get("dimensions", {}).get("value", [])
                shape_raw = plat_data.get("dimensions", {}).get("src_text", "")

                parsed_dims_list.append({
                    "platform": plat,
                    "shape": shape_raw,
                    "dimensions": dims,
                })

                if shape_raw and not dims:
                    dim_struct_failed.append(plat)
                elif dims:
                    ok = all(
                        isinstance(d, int) and 0 <= d <= 8
                        for d in dims
                    )
                    if ok:
                        any_passed = True
                    else:
                        dim_struct_failed.append(plat)

                # 2^31 alignment check removed: old per-dim [min, max] format
                # is deprecated; new enum format dims are all int rank values
                # in [0, 8], which can never exceed 2^31.

                ar_data = plat_data.get("allowed_range_value", {})
                ar_value = ar_data.get("value", []) if isinstance(ar_data, dict) else []
                ar_type = ar_data.get("type", "range") if isinstance(ar_data, dict) else "range"
                if ar_value and not isinstance(ar_value, list):
                    range_struct_failed.append(plat)
                elif ar_type and ar_type not in ("range", "enum"):
                    range_struct_failed.append(plat)

        def _phase(status: str, error: str = "", reason: str = "") -> dict:
            return {"status": status, "error": error, "reason": reason}

        dim_struct_err = (
            "; ".join(f"{p}: 无法解析 shape" for p in dim_struct_failed)
            if dim_struct_failed else ""
        )
        dim_align_err = (
            "; ".join(f"{p}: shape 维度值超出 2^31" for p in dim_align_failed)
            if dim_align_failed else ""
        )
        range_err = (
            "; ".join(f"{p}: allowed_range 结构非法" for p in range_struct_failed)
            if range_struct_failed else ""
        )

        validation_results.append({
            "function_name": fn_name,
            "param_name": pname,
            "platforms_count": len(platforms_to_build),
            "missing_platforms": [
                plat for plat in platforms_to_build
                if assembled is None or not assembled.get(plat)
            ],
            "has_constraint": assembled is not None and bool(assembled),
            "syntax_valid": bool(assembled) and not (
                dim_struct_failed or dim_align_failed or range_struct_failed
            ),
            "validation_error": "",
            "corrected": False,
            "has_validation": has_validation and any_passed,
            "parsed_dimensions": parsed_dims_list,
            "phase_dim_structure": _phase(
                "failed" if dim_struct_failed else
                ("passed" if any_passed else "skipped"),
                error=dim_struct_err,
            ),
            "phase_dim_alignment": _phase(
                "failed" if dim_align_failed else
                ("passed" if any_passed else "skipped"),
                error=dim_align_err,
            ),
            "phase_range_structure": _phase(
                "failed" if range_struct_failed else
                ("passed" if (assembled and any_passed) else "skipped"),
                error=range_err,
            ),
        })

    return {
        "error": None,
        "params_count": len(params),
        "dimensions_count": sum(
            len((dimensions_map.get(f"{p['function_name']}::{p['param_name']}::{p.get('shape','')}", []) or []))
            for p in params
        ),
        "range_count": 0,
        "validation_results": validation_results,
    }
