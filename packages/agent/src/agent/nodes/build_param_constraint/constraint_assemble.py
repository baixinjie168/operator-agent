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
from agent.utils.platform_utils import expand_common_in_constraint

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
        if isinstance(ar_data, dict) and ar_data.get("value") == [True, False] and key in constrained:
            ar_data["value"] = [constrained[key]]
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

            # allowed_range_value: try platform-specific key first (char_enum
            # produces fn::pn::platform), then fallback to generic key (fn::pn).
            plat_key = f"{fn_name}::{pname}::{plat}"
            ar_data = ar_map.get(plat_key) or ar_map.get(map_key, {"type": "range", "value": []})
            ar_value = ar_data.get("value", [])
            ar_type = ar_data.get("type", "range")
            usage_val = attrs.get("usage_notes", {}).get("value", "")
            if not ar_value and usage_val and not is_tensor:
                enum_values = _extract_enum_from_text(usage_val)
                if enum_values:
                    ar_value = enum_values

            ar_src_text = ar_data.get("src_text", "")

            # usage_notes is internal-only (used above for enum fallback and
            # elsewhere for null-pointer dtype=N/A detection); drop it from
            # the final output JSON.
            attrs.pop("usage_notes", None)

            constraint[plat] = {
                **attrs,
                "dimensions": {"value": dimensions_value, "src_text": shape_raw},
                "allowed_range_value": {"value": ar_value, "type": ar_type, "src_text": ar_src_text},
            }

        # Expand "common" key to per-platform entries before DB save
        expand_common_in_constraint(constraint, supported_platforms)

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
    # ExtractorAgent constraint detail panel (cs_constraint / cs_check_constraint).
    #
    # Frontend expected schema (cd-cst-check, _renderValidationResults with
    # rowMeta = {titleField: "param_name", paramsField: "function_name",
    # valueField: "parsed_dimensions", primaryLabel: "个参数"} and
    # phaseConfigs = [
    #   {name:"维度结构校验", field:"phase_dim_structure", kind:"dim_struct"},
    #   {name:"维度对齐校验", field:"phase_dim_alignment", kind:"dim_align"},
    #   {name:"取值范围结构校验", field:"phase_range_structure", kind:"range"},
    # ]).
    validation_results: list[dict] = []
    for param in params:
        pname = param.get("param_name", "")
        fn_name = param.get("function_name", "")
        map_key = f"{fn_name}::{pname}"
        attr_key = f"{fn_name}::{pname}::"  # match any platform

        # Find this param's assembled constraint.
        assembled: dict | None = None
        for u in updates:
            if u["function_name"] == fn_name and u["param_name"] == pname:
                try:
                    assembled = json.loads(u["param_constraint"])
                except (json.JSONDecodeError, TypeError):
                    assembled = None
                break

        # Aggregate per-platform validation results.
        dim_struct_failed: list[str] = []   # 维度结构校验失败平台
        dim_align_failed: list[str] = []   # 维度对齐校验失败平台
        range_struct_failed: list[str] = []  # 取值范围结构校验失败平台
        parsed_dims_list: list = []        # valueField: parsed_dimensions
        any_passed = False
        has_validation = bool(supported_platforms) and assembled is not None

        if assembled is not None:
            for plat in supported_platforms:
                plat_data = assembled.get(plat, {})
                dims = plat_data.get("dimensions", {}).get("value", [])
                ar_data = plat_data.get("allowed_range_value", {})
                ar_value = ar_data.get("value", []) if isinstance(ar_data, dict) else []
                ar_type = ar_data.get("type", "range") if isinstance(ar_data, dict) else "range"
                shape_raw = plat_data.get("dimensions", {}).get("src_text", "")

                parsed_dims_list.append({
                    "platform": plat,
                    "shape": shape_raw,
                    "dimensions": dims,
                })

                # 维度结构校验: dims 是否为合法格式（list of [min,max] 或 list of int）
                if shape_raw and not dims:
                    dim_struct_failed.append(plat)
                elif dims:
                    ok = all(
                        (isinstance(d, list) and len(d) == 2
                         and all(isinstance(x, (int, float)) for x in d))
                        or isinstance(d, int)
                        for d in dims
                    )
                    if ok:
                        any_passed = True
                    else:
                        dim_struct_failed.append(plat)
                else:
                    # No shape at all — treat as skipped (not failed).
                    pass

                # 维度对齐校验: dims 元素必须为 2 的幂次或带 min/max 等价
                # （仅作轻量校验：若 dims 中存在异常大的数则视为未对齐）
                if dims:
                    for d in dims:
                        if isinstance(d, list) and len(d) == 2:
                            lo, hi = d[0], d[1]
                            if hi is not None and hi > 2**31:
                                dim_align_failed.append(plat)
                                break

                # 取值范围结构校验: ar_value 必须为 list（enum 或 range）
                if ar_value and not isinstance(ar_value, list):
                    range_struct_failed.append(plat)
                elif ar_type and ar_type not in ("range", "enum"):
                    range_struct_failed.append(plat)

        def _phase(status: str, error: str = "", reason: str = "") -> dict:
            return {"status": status, "error": error, "reason": reason}

        # Aggregate error/reason messages for each phase.
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
            "platforms_count": len(supported_platforms),
            "missing_platforms": [
                plat for plat in supported_platforms
                if assembled is None or not assembled.get(plat)
            ],
            "has_constraint": assembled is not None and bool(assembled),
            "syntax_valid": bool(assembled) and not (
                dim_struct_failed or dim_align_failed or range_struct_failed
            ),
            "validation_error": "",
            "corrected": False,
            "has_validation": has_validation and any_passed,
            # valueField: 解析后的 dimensions
            "parsed_dimensions": parsed_dims_list,
            # 三段式校验（前端 phaseConfigs 期望的字段名）
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
        "range_count": sum(
            len((ar_map.get(f"{p['function_name']}::{p['param_name']}", {}).get("value", [])
                 if isinstance(ar_map.get(f"{p['function_name']}::{p['param_name']}", {}), dict)
                 else ar_map.get(f"{p['function_name']}::{p['param_name']}", []) or []))
            for p in params
        ),
        "validation_results": validation_results,
    }
