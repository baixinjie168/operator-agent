"""Assemble result node: aggregate all extraction results into a single structured JSON."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.utils.param_validators import EXCLUDED_PARAMS
from agent.utils.platform_utils import expand_common_in_constraint

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

        # Fetch implicit params (non-operator parameters) from DB
        implicit_params_data = await _mcp_client.query_implicit_params_by_doc_id(doc_id)
        mappings = implicit_params_data.get("mappings", []) if implicit_params_data else []

        # Fetch platform constants (external constants like rankSize) early:
        # needed both to define them in inputs (per-platform allowed_range_value)
        # and to inject into constraints_in_parameters.
        platform_consts_data = await _mcp_client.query_platform_constants_by_doc_id(doc_id)
        platform_constants = (
            platform_consts_data.get("constants", []) if platform_consts_data else []
        )

        inputs_dict, outputs_dict = _build_inputs_outputs(
            params, implicit_params=mappings, platform_constants=platform_constants,
        )
        constraints_ip = _build_constraints_in_parameters(
            relations, product_support_list, params,
        )
        dtype_support = _build_dtype_support(dtype_combos)

        # Expand "common" in inputs/outputs (from implicit params) to per-platform
        for constraint in inputs_dict.values():
            expand_common_in_constraint(constraint, product_support_list)
        for constraint in outputs_dict.values():
            expand_common_in_constraint(constraint, product_support_list)

        # Step 3g: Inject parameter_representation records into constraints_in_parameters
        param_reprs_data = await _mcp_client.query_parameter_representations_by_doc_id(doc_id)
        if param_reprs_data and (
            param_reprs_data.get("representations")
            or param_reprs_data.get("platform_representations")
        ):
            _inject_parameter_representations(constraints_ip, param_reprs_data)

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


def _extract_implicit_params(mappings: list[dict]) -> dict[str, dict]:
    """Extract non-operator parameters from implicit_params mappings.

    Returns: {var_name: {"type": ..., "shape_text": ..., ...}}
    Excludes: external constants and constant values (e.g. k0=16).
    """
    result: dict[str, dict] = {}
    for m in mappings:
        if m.get("is_external_constant") or m.get("is_constant"):
            continue
        var = m["var_name"]
        if var not in result:
            # Quantization type: char-typed enum (no tensor shape reference)
            if m.get("is_quantization_type"):
                result[var] = {
                    "type": m.get("param_type", "char"),
                    "is_quantization_type": True,
                    "allowed_range_value": m.get("allowed_range_value", []),
                    "allowed_range_type": m.get("allowed_range_type", "enum"),
                    "shape_text": "",
                    "tensor_param": None,
                    "dim_index": None,
                }
            else:
                result[var] = {
                    "type": "int64_t",
                    "shape_text": m.get("shape_text", ""),
                    "tensor_param": m.get("tensor_param", ""),
                    "dim_index": m.get("dim_index"),
                }
    return result


def _build_implicit_param_constraint(info: dict) -> dict:
    """Build a minimal constraint object for a non-operator parameter.

    Format matches operator param's param_constraint:
    {platform: {description, type, format, ...}}
    """
    # Quantization type: char-typed enum with document-derived allowed values
    if info.get("is_quantization_type"):
        constraint = {
            "description": "量化粒度隐式参数（per-channel/per-group/per-tensor/per-token 之一）",
            "type": {"value": info.get("type", "char"), "src_text": ""},
            "format": {"value": "N/A", "src_text": ""},
            "is_optional": {"value": False, "src_text": ""},
            "is_support_discontinuous": {"value": "N/A", "src_text": ""},
            "is_operator_param": {"value": False, "src_text": ""},
            "dimensions": {"value": [], "src_text": ""},
            "array_length": {"value": "N/A", "src_text": ""},
            "dtype": {"value": [], "src_text": ""},
            "allowed_range_value": {
                "value": info.get("allowed_range_value", []),
                "type": info.get("allowed_range_type", "enum"),
                "src_text": "",
            },
        }
        return {"common": constraint}

    tensor_ref = ""
    if info.get("tensor_param") and info.get("dim_index") is not None:
        tensor_ref = f"{info['tensor_param']}.shape[{info['dim_index']}]"

    constraint = {
        "description": f"隐式维度变量" + (f"，对应 {tensor_ref}" if tensor_ref else ""),
        "type": {"value": info.get("type", "int64_t"), "src_text": ""},
        "format": {"value": "N/A", "src_text": ""},
        "is_optional": {"value": False, "src_text": ""},
        "is_support_discontinuous": {"value": "N/A", "src_text": ""},
        "is_operator_param": {"value": False, "src_text": ""},
        "dimensions": {"value": [], "src_text": ""},
        "array_length": {"value": "N/A", "src_text": ""},
        "dtype": {"value": [], "src_text": ""},
        "allowed_range_value": {"value": [], "type": "range", "src_text": ""},
    }
    return {"common": constraint}


def _build_external_constant_constraints(
    platform_constants: list[dict],
) -> dict[str, dict[str, dict]]:
    """Build per-platform input constraints for external constants (e.g. rankSize).

    External constants have platform-specific value ranges (e.g. rankSize is
    [2, 4, 8] on Atlas A2 but [2, 4, 8, 16] on Atlas A3), so they cannot use
    the platform-agnostic ``"common"`` key — each platform gets its own
    constraint entry with ``allowed_range_value`` populated from the values
    extracted from the document context.

    Returns ``{const_name: {platform_name: constraint_dict}}``.
    """
    result: dict[str, dict[str, dict]] = {}
    for pc in platform_constants:
        cname = pc.get("const_name", "")
        if not cname:
            continue
        desc = pc.get("description", "") or (
            f"平台外部常量 {cname}（取值随设备型号不同）"
        )
        per_platform: dict[str, dict] = {}
        for pv in pc.get("platform_values", []):
            plat = pv.get("platform", "")
            values = pv.get("values", [])
            if not plat or not values:
                continue
            per_platform[plat] = {
                "description": desc,
                "type": {"value": "int64_t", "src_text": ""},
                "format": {"value": "N/A", "src_text": ""},
                "is_optional": {"value": False, "src_text": ""},
                "is_support_discontinuous": {"value": "N/A", "src_text": ""},
                "is_operator_param": {"value": False, "src_text": ""},
                "dimensions": {"value": [], "src_text": ""},
                "array_length": {"value": "N/A", "src_text": ""},
                "dtype": {"value": [], "src_text": ""},
                "allowed_range_value": {
                    "value": values,
                    "type": "enum",
                    "src_text": pv.get("source_citation", ""),
                },
            }
        if per_platform:
            result[cname] = per_platform
    return result


def _build_inputs_outputs(
    params: list[dict],
    implicit_params: list[dict] | None = None,
    platform_constants: list[dict] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build inputs/outputs: {param_name: param_constraint} split by direction.

    Includes:
    1. Parameters from *WorkspaceSize functions (excluding workspaceSize, executor)
    2. Non-operator (implicit) parameters extracted from shape descriptions
    3. External platform constants (e.g. rankSize) with per-platform value ranges
    """
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}

    # 1. Operator params from GetWorkspaceSize function
    for p in params:
        fn = p.get("function_name", "")
        if not fn.endswith("WorkspaceSize"):
            continue
        name = p.get("param_name", "")
        if name in EXCLUDED_PARAMS:
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

    # 2. Non-operator (implicit) parameters from shape descriptions
    if implicit_params:
        extracted = _extract_implicit_params(implicit_params)
        for name, info in extracted.items():
            if name not in inputs and name not in outputs:
                inputs[name] = _build_implicit_param_constraint(info)

    # 3. External platform constants (e.g. rankSize) — per-platform allowed ranges.
    # Surfaced as inputs so their document-derived value ranges are available
    # alongside operator params; constraints_in_parameters still carries the
    # raw platform_constants metadata block.
    if platform_constants:
        ext = _build_external_constant_constraints(platform_constants)
        for name, per_platform in ext.items():
            if name not in inputs and name not in outputs:
                inputs[name] = per_platform

    return inputs, outputs


def _inject_parameter_representations(
    constraints_ip: dict[str, list[dict]],
    param_reprs_data: dict,
) -> None:
    """Inject parameter_representation records into constraints_in_parameters.

    Modifies *constraints_ip* in place. For each platform:
    - Platform-specific representations (external constant value sets,
      e.g. ``rankSize.range_value in [2, 4, 8]``) are inserted only into
      the matching platform.
    - Platform-agnostic tensor-dim representations (e.g.
      ``BS.range_value == x1.shape[0]``) are inserted into every platform.

    Insertion point is right after any ``_type: platform_constants``
    metadata entry so the final per-platform ordering is:
    ``[platform_constants, parameter_representations..., <other constraints>]``.
    """
    tensor_reps: list[dict] = param_reprs_data.get("representations", []) or []
    platform_reps: dict[str, list[dict]] = (
        param_reprs_data.get("platform_representations", {}) or {}
    )

    if not tensor_reps and not platform_reps:
        return

    for plat, constraint_list in constraints_ip.items():
        inserts: list[dict] = []
        # Platform-specific representations first
        if plat in platform_reps:
            inserts.extend(platform_reps[plat])
        # Then platform-agnostic tensor-dim representations
        if tensor_reps:
            inserts.extend(tensor_reps)

        if not inserts:
            continue

        # Prepend parameter representations to the constraint list
        constraint_list[0:0] = inserts


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

    # Build {param_name: (allowed_range_value, type)} lookup from param_constraint JSON
    ar_lookup: dict[str, tuple[list, str]] = {}
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
                        ar_type = ar.get("type", "range")
                        if val:
                            ar_lookup[name] = (val, ar_type)
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
        # But do NOT skip when type="enum" (enum semantics differ from range)
        expr_type = obj.get("expr_type", "")
        rel_params = obj.get("relation_params", [])
        if (
            expr_type == "value_dependency"
            and len(rel_params) == 1
            and rel_params[0] in ar_lookup
        ):
            _, ar_type = ar_lookup[rel_params[0]]
            if ar_type != "enum":
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
        plat = dc.get("platform", "common")
        combo = dc.get("combo", {})
        if combo:
            grouped.setdefault(plat, []).append(combo)
    return grouped

