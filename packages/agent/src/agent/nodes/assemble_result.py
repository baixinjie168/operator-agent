"""Assemble result node: aggregate all extraction results into a single structured JSON."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def assemble_result_node(state: PipelineState) -> dict[str, Any]:
    """Assemble all extraction results into constraints_result table.

    This node runs after ALL parallel nodes (5a~5k + param_relation) complete.
    It queries all tables by doc_id, groups data by function_name, and saves
    the assembled JSON to the constraints_result table.
    """
    doc_id = state.get("doc_id")
    operator_name = state.get("operator_name", "")

    if not doc_id:
        logger.warning("assemble_result: missing doc_id, skipping")
        return {"error": None}

    logger.info("assemble_result: starting assembly for doc_id=%s (%s)", doc_id, operator_name)

    try:
        # Step 1: Get parsed document (for operator_name fallback)
        parsed = await _mcp_client.get_parsed_by_doc_id(doc_id)
        if parsed and not operator_name:
            operator_name = parsed.get("operator_name", "")

        # Step 2: Query all data by doc_id
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        relations = await _mcp_client.query_param_relations(doc_id)
        signatures = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        platform_support_data = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        return_codes = await _mcp_client.query_return_codes_by_doc_id(doc_id)
        dtype_combos = await _mcp_client.query_dtype_combos_by_doc_id(doc_id)

        # Log single-param vs multi-param relation breakdown
        single_count = sum(
            1 for r in relations
            if r.get("relation_type") == "self_constraint"
        )
        logger.info(
            "assemble_result: %d total relations (%d single-param, %d multi-param)",
            len(relations), single_count, len(relations) - single_count,
        )

        # Step 2b: Fetch function_explanation_summary from document_versions
        fn_expl_summary = await _mcp_client.get_function_explanation_summary(doc_id)
        description = fn_expl_summary.get("description", "")

        # Step 3: Build function_explanation JSON
        function_explanation = _build_function_explanation(
            params, relations, signatures, return_codes, dtype_combos,
            description=description,
        )

        function_explanation_raw = json.dumps(function_explanation, ensure_ascii=False)

        # Step 3.5: Build product_support from platform_support (is_supported=1)
        product_support_list = [
            p["platform_name"]
            for p in platform_support_data
            if p.get("is_supported") == 1
        ]
        product_support_raw = json.dumps(product_support_list, ensure_ascii=False)

        # Step 2c: Extract GetWorkspaceSize signature
        workspace_sig = ""
        for sig in signatures:
            if sig.get("function_name", "").endswith("GetWorkspaceSize"):
                workspace_sig = sig.get("full_signature", "")
                break

        # Step 2d: Transform return_codes (deduplicate by (return_value, error_code))
        transformed_rc = _transform_return_codes(return_codes)
        return_codes_raw = json.dumps(transformed_rc, ensure_ascii=False)

        # Step 3e: Build new fields
        det_computing = _build_deterministic_computing(platform_support_data)
        inputs_dict, outputs_dict = _build_inputs_outputs(params)
        constraints_ip = _build_constraints_in_parameters(relations, product_support_list, params)
        dtype_support = _build_dtype_support(dtype_combos)

        # Step 4: Save to constraints_result table
        await _mcp_client.save_constraints_result(
            doc_id=doc_id,
            operator_name=operator_name,
            product_support=product_support_raw,
            function_explanation=function_explanation_raw,
            function_signature=workspace_sig,
            return_codes=return_codes_raw,
            deterministic_computing=json.dumps(det_computing, ensure_ascii=False),
            inputs=json.dumps(inputs_dict, ensure_ascii=False),
            outputs=json.dumps(outputs_dict, ensure_ascii=False),
            constraints_in_parameters=json.dumps(constraints_ip, ensure_ascii=False),
            dtype_support_description=json.dumps(dtype_support, ensure_ascii=False),
        )

        # Step 5: Build result.json structure
        result_json = {
            "operator_name": operator_name,
            "function_explanation": description,
            "product_support": product_support_list,
            "function_signature": workspace_sig,
            "deterministic_computing": det_computing,
            "inputs": inputs_dict,
            "outputs": outputs_dict,
            "constraints_in_parameters": constraints_ip,
            "return_info": transformed_rc,
            "dtype_support_description": dtype_support,
        }

        # Step 6: Save to document_versions.json_constraints
        await _mcp_client.save_json_constraints(
            doc_id=doc_id,
            json_constraints=json.dumps(result_json, ensure_ascii=False),
        )

        fn_count = len(function_explanation)
        param_count = len(params)
        logger.info(
            "assemble_result: saved %d functions, %d params for %s (doc_id=%s)",
            fn_count, param_count, operator_name, doc_id,
        )

        return {"error": None}

    except Exception as e:
        logger.exception("assemble_result failed for %s", operator_name)
        return {"error": str(e)}


def _has_meaningful_expr(obj: dict) -> bool:
    """Check if a relation_object has a non-empty expr field.

    Relations with empty expr (e.g. presence_dependency descriptions like
    "当weightOptional为空时，会以self的shape创建一个全1的Tensor") are
    implementation notes, not verifiable constraints — they should be
    excluded from the final output.
    """
    if not isinstance(obj, dict):
        return True
    expr = obj.get("expr", "")
    if isinstance(expr, str):
        return bool(expr.strip())
    # Non-string expr (e.g. list, number) is considered meaningful
    return True


def _build_function_explanation(
    params: list[dict],
    relations: list[dict],
    signatures: list[dict],
    return_codes: list[dict],
    dtype_combos: list[dict],
    description: str = "",
) -> dict:
    """Group all data by function_name and build the function_explanation structure."""
    # Collect all function names from all sources
    all_fn_names: set[str] = set()
    for source in [params, relations, signatures, return_codes, dtype_combos]:
        for item in source:
            fn = item.get("function_name", "")
            if fn:
                all_fn_names.add(fn)

    result: dict[str, Any] = {}

    # Inject top-level description from function_explanation_summary
    if description:
        result["description"] = description

    for fn in sorted(all_fn_names):
        fn_params = [p for p in params if p.get("function_name") == fn]
        fn_relations = [
            r for r in relations
            if r.get("function_name") == fn
            and _has_meaningful_expr(r.get("relation_object", {}))
        ]
        fn_sig = next(
            (s for s in signatures if s.get("function_name") == fn), None,
        )
        fn_rc = [rc for rc in return_codes if rc.get("function_name") == fn]
        fn_dc = [dc for dc in dtype_combos if dc.get("function_name") == fn]

        result[fn] = {
            "signature": fn_sig or {},
            "params": fn_params,
            "relations": fn_relations,
            "return_codes": fn_rc,
            "dtype_combinations": fn_dc,
        }

    return result


def _transform_return_codes(raw_codes: list[dict]) -> list[dict]:
    """Deduplicate return codes by (return_value, error_code) and merge descriptions.

    Pure in-memory operation — no DB queries, no LLM calls.
    """
    merged: dict[tuple[str, int], list[str]] = {}
    for rc in raw_codes:
        key = (rc.get("return_value", ""), rc.get("error_code", 0))
        descs = rc.get("descriptions", [])
        if key not in merged:
            merged[key] = list(descs)
        else:
            merged[key].extend(descs)
    return [
        {
            "return_value": rv,
            "error_code": ec,
            "description": descs,
        }
        for (rv, ec), descs in merged.items()
    ]


def _build_deterministic_computing(platforms: list[dict]) -> dict[str, Any]:
    """Build deterministic_computing: {platform_name: {value, src_text}}."""
    result: dict[str, Any] = {}
    for p in platforms:
        if p.get("is_supported") == 1:
            name = p.get("platform_name", "")
            det = p.get("deterministic_computing", {})
            if name:
                result[name] = det
    return result


def _build_inputs_outputs(
    params: list[dict],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build inputs/outputs: {param_name: param_constraint} split by direction.

    Only includes parameters that meet ALL of:
    1. function_name ends with "WorkspaceSize"
    2. param_name is not "workspaceSize" or "executor"
    """
    _EXCLUDED_PARAMS = {"workspaceSize", "executor"}
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for p in params:
        fn = p.get("function_name", "")
        if not fn.endswith("WorkspaceSize"):
            continue
        name = p.get("param_name", "")
        if name in _EXCLUDED_PARAMS:
            continue
        constraint_raw = p.get("param_constraint", "{}") or "{}"
        try:
            constraint = json.loads(constraint_raw) if isinstance(constraint_raw, str) else constraint_raw
        except (json.JSONDecodeError, TypeError):
            constraint = {}
        if p.get("direction") == "output":
            outputs[name] = constraint
        else:
            inputs[name] = constraint
    return inputs, outputs


def _build_constraints_in_parameters(
    relations: list[dict],
    supported_platforms: list[str],
    params: list[dict],
) -> dict[str, list[dict]]:
    """Build constraints_in_parameters: {platform: [relation_object]}.

    Deduplicates single-parameter value_dependency constraints when the
    parameter already has a non-empty allowed_range_value in inputs/outputs
    (the structured range is the canonical representation).

    Args:
        relations: List of param_relation dicts with 'platform' and 'relation_object' fields.
        supported_platforms: List of platform names where is_supported=1.
        params: List of parameter dicts (used to build allowed_range_value lookup).

    Returns:
        Dict mapping platform name to list of relation_object dicts.
        If a relation's platform is empty, it applies to all supported platforms.
        If a relation's platform specifies platforms, it only applies to those
        that are also in supported_platforms.
    """
    from agent.utils.platform_utils import resolve_target_platforms

    # Build {param_name: allowed_range_value} lookup from param_constraint JSON
    ar_lookup: dict[str, list] = {}
    for p in params:
        name = p.get("param_name", "")
        constraint_raw = p.get("param_constraint", "{}") or "{}"
        try:
            constraint = json.loads(constraint_raw) if isinstance(constraint_raw, str) else constraint_raw
        except (json.JSONDecodeError, TypeError):
            continue
        # Extract allowed_range_value from any platform (they are all identical)
        if isinstance(constraint, dict):
            for plat_data in constraint.values():
                if isinstance(plat_data, dict):
                    ar = plat_data.get("allowed_range_value", {})
                    if isinstance(ar, dict):
                        val = ar.get("value", [])
                        if val:
                            ar_lookup[name] = val
                    break

    grouped: dict[str, list[dict]] = {}
    skipped_count = 0
    for r in relations:
        obj = r.get("relation_object", {})
        if not obj or obj == {}:
            continue
        if not _has_meaningful_expr(obj):
            continue

        # Dedup: skip single-param value_dependency when allowed_range_value covers it
        expr_type = obj.get("expr_type", "")
        rel_params = obj.get("relation_params", [])
        if (
            expr_type == "value_dependency"
            and len(rel_params) == 1
            and rel_params[0] in ar_lookup
        ):
            skipped_count += 1
            continue

        platform_str = r.get("platform", "")
        targets = resolve_target_platforms(platform_str, supported_platforms)
        for plat in targets:
            grouped.setdefault(plat, []).append(obj)

    if skipped_count > 0:
        logger.info(
            "assemble_result: deduplicated %d single-param value_dependency "
            "constraints (covered by allowed_range_value)",
            skipped_count,
        )

    return grouped


def _build_dtype_support(dtype_combos: list[dict]) -> dict[str, list[dict]]:
    """Build dtype_support_description: {platform: [combo]}."""
    grouped: dict[str, list[dict]] = {}
    for dc in dtype_combos:
        plat = dc.get("platform", "通用")
        combo = dc.get("combo", {})
        if combo:
            grouped.setdefault(plat, []).append(combo)
    return grouped

