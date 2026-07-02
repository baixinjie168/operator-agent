"""Assemble result node: aggregate all extraction results into a single structured JSON."""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.utils.param_alias import expand_expr, load_alias_map
from agent.utils.param_validators import EXCLUDED_PARAMS, get_primary_function_names
from agent.utils.platform_utils import expand_common_in_constraint

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


# Dtype derivation from C type name — mirrors attrs_build Level-3 fallback.
# Array types map to their inherent primitive dtype; non-tensor scalar types
# (bool, int64_t, char, etc.) use the type name itself; tensor types are
# excluded because "aclTensor" is a type, not a dtype.
_ARRAY_TYPE_DTYPE_FALLBACK: dict[str, str] = {
    "aclIntArray": "int",
    "aclFloatArray": "float",
    "aclBoolArray": "bool",
}
_NO_DTYPE_FALLBACK_TYPES = frozenset({"aclTensor", "aclTensorList"})


def _derive_dtype_from_type(ptype: str) -> list[str]:
    """Derive a dtype list from a C type name.

    Used for implicit params and external constants that bypass attrs_build
    but still need a non-empty dtype when the type is a scalar (bool, int64_t,
    char, etc.).  Tensor types return [] because their dtype must come from
    dtype_desc or dtype_combinations, not the type name.
    """
    if not ptype:
        return []
    if ptype in _NO_DTYPE_FALLBACK_TYPES:
        return []
    return [_ARRAY_TYPE_DTYPE_FALLBACK.get(ptype, ptype)]


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

        # Step 2c: Extract primary function signature (GetWorkspaceSize for
        # two-stage operators; the operator's only function for single-function
        # operators like aclnnCalculateMatmulWeightSize).
        workspace_sig = ""
        for sig in signatures:
            if sig.get("function_name", "").endswith("GetWorkspaceSize"):
                workspace_sig = sig.get("full_signature", "")
                break
        # Single-function fallback: no GetWorkspaceSize signature -> use the
        # first available signature so function_signature is non-empty.
        if not workspace_sig and signatures:
            workspace_sig = signatures[0].get("full_signature", "")

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
            relations=relations, signatures=signatures,
        )
        # Build AR lookup from the *filled* inputs/outputs (final state). Unlike
        # the DB-only lookup, this also covers implicit params (BS/N/H …) whose
        # allowed_range_value was derived from param_relations, so the
        # constraints_in_parameters dedup applies to them consistently.
        io_ar_lookup = _build_ar_lookup_from_io(inputs_dict, outputs_dict)
        constraints_ip = _build_constraints_in_parameters(
            relations, product_support_list, params, ar_lookup=io_ar_lookup,
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

        # FIX-6: Inject allowed_range_value as constraints (before dedup).
        # R4: src_text from AR; R5: ensure platform keys; R15: .range_value ref.
        ar_injected = _inject_allowed_range_as_constraints(
            constraints_ip, inputs_dict, outputs_dict, product_support_list,
        )
        if ar_injected:
            logger.info(
                "assemble_result: injected %d AR constraints for %s",
                ar_injected, operator_name,
            )

        # FIX-8: Expand aliases FIRST (so shorthand names are resolved
        # before FIX-2 filtering).  Step 3h (existing).
        _expand_aliases_in_constraints(operator_name, constraints_ip)

        # FIX-2: Filter invalid param refs AFTER alias expansion (R1).
        # Build valid param set: inputs + outputs + implicit +
        # external_constants (R2) + ALL params cross-function (R12).
        valid_params: set[str] = set(inputs_dict) | set(outputs_dict)
        if mappings:
            valid_params |= {
                m.get("var_name") for m in mappings if m.get("var_name")
            }
        for pc in platform_constants:  # R2: include external constants
            cname = pc.get("const_name", "")
            if cname:
                valid_params.add(cname)
        # R12: include ALL params (cross-function) — params queried at L71
        valid_params |= {
            p.get("param_name") for p in params if p.get("param_name")
        }
        filtered = _filter_invalid_param_refs(constraints_ip, valid_params)
        if filtered:
            logger.info(
                "assemble_result: filtered %d invalid constraints for %s",
                filtered, operator_name,
            )

        # FIX-10: Source citation existence verification (after FIX-2 param
        # legality, before FIX-1 semantic dedup).  Deletes constraints whose
        # src_text cannot be found in the document — covers the "params legal
        # but src_text fabricated" gap that FIX-2 cannot catch.
        if settings.relation_verify_source:
            from agent.utils.section_utils import resolve_ws_exe_content

            ws_text, exe_text, _constraints_text = await resolve_ws_exe_content(
                _mcp_client, doc_id,
            )
            # resolve_ws_exe_content already appends constraints_text to
            # ws_text (see section_utils._CONSTRAINTS_PREFIX); do NOT append
            # _constraints_text again here to avoid duplicating doc content.
            document_text = f"{ws_text}\n\n{exe_text}"
            src_report = _filter_fabricated_src_text(
                constraints_ip, document_text,
                threshold=settings.relation_verify_source_threshold,
            )
            if src_report["deleted"]:
                logger.info(
                    "assemble_result: FIX-10 deleted %d fabricated-src "
                    "constraints for %s (kept=%d, skipped=%d)",
                    src_report["deleted"], operator_name,
                    src_report["kept"], src_report["skipped"],
                )

        # FIX-1: Semantic dedup (after filtering removes invalid refs)
        deduped = _deduplicate_constraints_semantic(constraints_ip)
        if deduped:
            logger.info(
                "assemble_result: deduplicated %d constraints for %s",
                deduped, operator_name,
            )

        # FIX-5: Contradiction detection (skips IfExp constraints, R11)
        from agent.utils.expr_validation import detect_constraint_contradictions

        for plat, clist in constraints_ip.items():
            contradictions = detect_constraint_contradictions(clist)
            for i, j, reason in contradictions:
                logger.warning(
                    "assemble_result: contradiction on %s: %s", plat, reason,
                )
                clist[i]["_contradiction_warning"] = reason

        # FIX-13: expr_type normalization
        _normalize_expr_types(constraints_ip)

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
    # Fallback: when no supported platform has determinism data, include
    # all platforms that have non-empty deterministic_computing values.
    if not result:
        logger.warning(
            "assemble_result: no supported platform has deterministic_computing, "
            "falling back to all platforms"
        )
        for p in platforms:
            name = p.get("platform_name", "")
            det = p.get("deterministic_computing", {})
            if name and det.get("value"):
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
        _ptype = info.get("type", "char")
        constraint = {
            "description": "量化粒度隐式参数（per-channel/per-group/per-tensor/per-token 之一）",
            "type": {"value": _ptype, "src_text": ""},
            "format": {"value": "N/A", "src_text": ""},
            "is_optional": {"value": False, "src_text": ""},
            "is_support_discontinuous": {"value": "N/A", "src_text": ""},
            "is_operator_param": {"value": False, "src_text": ""},
            "dimensions": {"value": [], "src_text": ""},
            "array_length": {"value": "N/A", "src_text": ""},
            "dtype": {"value": _derive_dtype_from_type(_ptype), "src_text": ""},
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

    _ptype = info.get("type", "int64_t")
    constraint = {
        "description": f"隐式维度变量" + (f"，对应 {tensor_ref}" if tensor_ref else ""),
        "type": {"value": _ptype, "src_text": ""},
        "format": {"value": "N/A", "src_text": ""},
        "is_optional": {"value": False, "src_text": ""},
        "is_support_discontinuous": {"value": "N/A", "src_text": ""},
        "is_operator_param": {"value": False, "src_text": ""},
        "dimensions": {"value": [], "src_text": ""},
        "array_length": {"value": "N/A", "src_text": ""},
        "dtype": {"value": _derive_dtype_from_type(_ptype), "src_text": ""},
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
            _ptype = pc.get("const_type", "int64_t")
            per_platform[plat] = {
                "description": desc,
                "type": {"value": _ptype, "src_text": ""},
                "format": {"value": "N/A", "src_text": ""},
                "is_optional": {"value": False, "src_text": ""},
                "is_support_discontinuous": {"value": "N/A", "src_text": ""},
                "is_operator_param": {"value": False, "src_text": ""},
                "dimensions": {"value": [], "src_text": ""},
                "array_length": {"value": "N/A", "src_text": ""},
                "dtype": {"value": _derive_dtype_from_type(_ptype), "src_text": ""},
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
    relations: list[dict] | None = None,
    signatures: list[dict] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build inputs/outputs: {param_name: param_constraint} split by direction.

    Includes:
    1. Parameters from primary function(s) (excluding workspaceSize, executor).
       For two-stage operators the primary function is *GetWorkspaceSize; for
       single-function operators it is the operator's only function. When
       *signatures* is unavailable (None/empty), falls back to the legacy
       endswith("WorkspaceSize") filter to avoid regressing worse than status quo.
    2. Non-operator (implicit) parameters extracted from shape descriptions
    3. External platform constants (e.g. rankSize) with per-platform value ranges
    4. allowed_range_value filled from param_relations (self_value_range/enum/bool)
    """
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}

    # 1. Operator params from primary function(s)
    primary_fns = get_primary_function_names(signatures or [])
    for p in params:
        fn = p.get("function_name", "")
        # primary_fns is None when signatures is empty (query/parse failure):
        # fall back to the legacy endswith filter rather than an empty set that
        # would drop every parameter (a worse-than-status-quo regression).
        if primary_fns is not None:
            if fn not in primary_fns:
                continue
        elif not fn.endswith("WorkspaceSize"):
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

    # 3. External platform constants (e.g. rankSize)
    if platform_constants:
        ext = _build_external_constant_constraints(platform_constants)
        for name, per_platform in ext.items():
            if name not in inputs and name not in outputs:
                inputs[name] = per_platform

    # 4. Fill allowed_range_value from param_relations
    if relations:
        ar_lookup = _build_allowed_range_lookup(relations)
        if ar_lookup:
            _fill_ar_for_params(inputs, ar_lookup)
            _fill_ar_for_params(outputs, ar_lookup)

    # 4b. Bool default: an unconstrained bool param allows [True, False].
    #     AR is finalised here (after the relation-derived fill), so a bool
    #     with no textual restriction still gets a non-empty enum.  Mirrors
    #     allowed_range_build_node Phase 0 bool short-circuit, whose output
    #     (allowed_range_map) is no longer consumed by constraint_assemble.
    _apply_bool_default(inputs)
    _apply_bool_default(outputs)

    return inputs, outputs


def _is_compound_expr(expr: str, pname: str) -> bool:
    """True if *expr* is a compound / multi-variable constraint.

    Such constraints (e.g. ``H.range_value * rankSize.range_value``) are
    cross-parameter and must NOT be flattened into a single param's
    allowed_range_value — they belong only in constraints_in_parameters.
    """
    # Product / sum / mod of two .range_value references → compound
    if re.search(r"\.range_value\s*[*+/%]\s*\w+\.range_value", expr):
        return True
    # Any .range_value reference to a param other than *pname* → multi-var
    for m in re.finditer(r"(\w+)\.range_value", expr):
        if m.group(1) != pname:
            return True
    return False


def _recover_bounds_from_text(
    text: str, pname: str, lo: int | None, hi: int | None,
) -> tuple[int | None, int | None]:
    """Best-effort recovery of a missing range bound from source citation text.

    Splits the text into clauses (sentence + comma boundaries) and searches
    only within clauses mentioning *pname*. This avoids stealing another
    param's bound when several params share one sentence (e.g.
    "BS...不得小于0, N...不得小于1"). ``不得小于`` (a lower bound) is
    explicitly excluded from upper-bound matching via a negative lookbehind.
    Returns the (possibly updated) (lo, hi) pair.
    """
    if not text or not pname:
        return lo, hi
    pn = re.escape(pname)
    clauses = re.split(r"[。；;\n，,]", text)
    relevant = [c for c in clauses if pname in c]
    hay = " ".join(relevant) if relevant else text

    if lo is None:
        m = (
            re.search(pn + r"[^\d\n]{0,20}?(?:不得小于|不得少于|至少|最小为)\s*(\d+)", hay)
            or re.search(pn + r"[^\d\n]{0,20}?(?:>=|大于)\s*(\d+)", hay)
            or re.search(r"(\d+)\s*<=\s*" + pn, hay)
        )
        if m:
            lo = int(m.group(1))

    if hi is None:
        m = (
            re.search(pn + r"[^\d\n]{0,20}?(?:不得超过|不超过|不得大于|最多|最大为)\s*(\d+)", hay)
            or re.search(pn + r"[^\d\n]{0,20}?(?:<=|(?<!不得)小于)\s*(\d+)", hay)
            or re.search(pn + r"\s*<=\s*(\d+)", hay)
        )
        if m:
            hi = int(m.group(1))

    return lo, hi


def _ensure_complete_range(
    value_list: list, ar_type: str, src_text: str, pname: str,
) -> list | None:
    """Enforce the invariant: allowed_range_value holds only complete
    two-sided ranges ``[[lo, hi]]`` with both bounds non-null.

    - enum / bool pass through unchanged (no null-bound issue).
    - range entries missing a side are first recovered from *src_text*; if
      recovery still fails the entry is dropped (the constraint stays in
      constraints_in_parameters). ``[[None, x]]`` / ``[[x, None]]`` are never
      emitted.
    Returns the cleaned list, or None if nothing survives.
    """
    if ar_type != "range":
        return value_list
    if not isinstance(value_list, list):
        return None

    result: list = []
    for item in value_list:
        # bool scalar ([True]/[False]) — not a [lo, hi] pair
        if isinstance(item, bool):
            result.append(item)
            continue
        if not isinstance(item, list) or len(item) != 2:
            continue
        lo, hi = item[0], item[1]
        if isinstance(lo, bool) or isinstance(hi, bool):
            result.append(item)
            continue
        if lo is None or hi is None:
            lo, hi = _recover_bounds_from_text(src_text, pname, lo, hi)
        if lo is None or hi is None:
            logger.debug(
                "assemble_result: dropped incomplete range [%s, %s] for %s "
                "(missing bound not recoverable; kept in constraints_in_parameters)",
                lo, hi, pname,
            )
            continue
        result.append([lo, hi])
    return result if result else None


def _merge_and_complete(
    entries: list[tuple[list, str, str]], pname: str,
) -> tuple[list | None, str, str]:
    """Merge multiple parsed entries for one param and enforce completeness.

    - enum: take the first non-empty enum entry.
    - bool: take the first ``[True]/[False]`` entry.
    - range: merge all ``[lo, hi]`` pairs into the tightest single range
      (lo = max of non-null lows, hi = min of non-null highs), so bounds
      split across several relations are reunited. Then enforce the
      no-null-bounds invariant (with doc-text recovery).
    """
    # enum first
    for v, t, s in entries:
        if t == "enum" and v:
            return v, t, s
    # bool (single bool scalar) — force "enum" type regardless of upstream.
    # Bool values are discrete, not a continuous range. This is a defensive
    # measure: after Fix-1 _parse_range_expr already returns "enum" for bool,
    # so the enum branch above handles the normal path. This branch catches
    # any bool entry that still carries a "range" type from other sources.
    for v, t, s in entries:
        if (
            isinstance(v, list)
            and len(v) == 1
            and isinstance(v[0], bool)
        ):
            return v, "enum", s
    # range merge
    lo: int | None = None
    hi: int | None = None
    src = ""
    for v, t, s in entries:
        if t != "range" or not isinstance(v, list):
            continue
        if s and not src:
            src = s
        for item in v:
            if not (isinstance(item, list) and len(item) == 2):
                continue
            elo, ehi = item[0], item[1]
            if isinstance(elo, bool) or isinstance(ehi, bool):
                continue
            if elo is not None:
                lo = elo if lo is None else max(lo, elo)
            if ehi is not None:
                hi = ehi if hi is None else min(hi, ehi)
    if lo is None and hi is None:
        return None, "", ""

    merged = _ensure_complete_range([[lo, hi]], "range", src, pname)
    if not merged:
        return None, "", ""
    return merged, "range", src


def _build_allowed_range_lookup(
    relations: list[dict],
) -> dict[str, dict[str, tuple[list, str, str]]]:
    """Build {param_name: {platform: (value_list, ar_type, src_text)}} from param_relations.

    The returned dict is keyed by param_name, then by platform (where "" means
    common / applies to all platforms). Per-platform entries are kept separate
    so that ``_fill_ar_for_params`` can apply different values to different
    platforms.

    Invariants enforced:
      1. Only single-parameter, non-compound self range/enum/bool constraints
         are flattened into allowed_range_value. Multi-param or compound
         constraints (e.g. ``1 <= H*rankSize <= 35000``) are skipped — they
         belong only in constraints_in_parameters.
      2. Multiple relations for the same param+platform are aggregated (bounds merged).
      3. allowed_range_value never contains one-sided ranges ``[[None, x]]`` /
         ``[[x, None]]``: a missing bound is recovered from src_text, and if
         unrecoverable the entry is dropped (constraint stays in
         constraints_in_parameters).
    """
    # Collect parsed candidates per param+platform
    # candidates[pname][platform] = [(value_list, ar_type, src_text), ...]
    candidates: dict[str, dict[str, list[tuple[list, str, str]]]] = {}
    for rel in relations:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(obj, dict):
            continue
        expr = obj.get("expr", "")
        if not expr:
            continue

        rel_params = obj.get("relation_params", [])
        # Stage 1: only single-param self constraints
        if not rel_params or len(rel_params) != 1:
            continue
        pname = rel_params[0]
        # Skip compound / multi-var exprs even when len(rel_params) == 1
        if _is_compound_expr(expr, pname):
            continue

        value_list, ar_type = _parse_range_expr(expr, obj.get("expr_type", ""))
        if value_list is None:
            continue
        platform = rel.get("platform", "")
        candidates.setdefault(pname, {}).setdefault(platform, []).append(
            (value_list, ar_type, obj.get("src_text", ""))
        )

    # Merge within each param+platform
    lookup: dict[str, dict[str, tuple[list, str, str]]] = {}
    for pname, plat_entries in candidates.items():
        for platform, entries in plat_entries.items():
            value, ar_type, src = _merge_and_complete(entries, pname)
            if value:
                lookup.setdefault(pname, {})[platform] = (value, ar_type, src)

    return lookup


def _parse_range_expr(expr: str, expr_type: str) -> tuple[list | None, str]:
    """Parse a range/enum/bool expr into (value_list, ar_type).

    Returns (None, "") if the expr is not a recognizable value range.

    Supported forms:
      - bool:   ``x.range_value == True/False``
      - enum:   ``x.range_value in [32, 64]`` / ``['ND', 'NZ']``
      - chained:``LO <= x.range_value <= HI`` (also ``<`` variants; exclusive
                bounds are adjusted by ±1 to an inclusive [lo, hi])
      - split:  ``x.range_value >= LO and x.range_value <= HI``
      - one-sided: ``x.range_value <= HI`` / ``x.range_value >= LO`` (the
                missing side is left as None; callers must enforce completeness)
    """
    # Bool: "x.range_value == False" / "x.range_value == True"
    # Bool values are discrete (True/False), not a continuous range —
    # use "enum" type so downstream dedup/injection logic treats them
    # as discrete values, not as [lo, hi] range pairs.
    if expr_type == "self_value_dependency" or ".range_value ==" in expr:
        m = re.search(r"\.range_value\s*==\s*(True|False)", expr)
        if m:
            val = m.group(1) == "True"
            return [val], "enum"
        return None, ""

    # Enum: "x.range_value in [32, 64]" / "x.range_value in ['ND', 'NZ']"
    m = re.search(r"\.range_value\s+in\s+\[([^\]]+)\]", expr)
    if m:
        raw = m.group(1).strip()
        parts = [p.strip() for p in raw.split(",")]
        if all(p.lstrip("-").isdigit() for p in parts if p):
            nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
            return nums, "enum"
        strs = [p.strip().strip("'\"") for p in parts if p.strip()]
        if strs:
            return strs, "enum"
        return None, ""

    # Chained comparison: "LO <= x.range_value <= HI" (and < variants).
    # Must run before the split ge/le logic — the latter cannot see a lower
    # bound written on the left of "<=" (it only matches "var >= LO").
    m_chain = re.search(
        r"(-?\d+)\s*(<=|<)\s*\w+\.range_value\s*(<=|<)\s*(-?\d+)", expr,
    )
    if m_chain:
        lo = int(m_chain.group(1))
        lo_op = m_chain.group(2)
        hi_op = m_chain.group(3)
        hi = int(m_chain.group(4))
        if lo_op == "<":
            lo += 1
        if hi_op == "<":
            hi -= 1
        return [[lo, hi]], "range"

    # Split / one-sided range: "x >= 0 and x <= 100" / "x <= 100" / "x >= 0"
    lo: int | None = None
    hi: int | None = None
    ge = re.search(r">=\s*(-?\d+)", expr)
    le = re.search(r"<=\s*(-?\d+)", expr)
    gt = re.search(r"(?<!<)>(-?\d+)", expr)
    lt = re.search(r"(?<!!)<(-?\d+)", expr)
    if ge:
        lo = int(ge.group(1))
    elif gt:
        lo = int(gt.group(1)) + 1
    if le:
        hi = int(le.group(1))
    elif lt:
        hi = int(lt.group(1)) - 1
    if lo is not None or hi is not None:
        return [[lo, hi]], "range"

    return None, ""


def _fill_ar_for_params(
    param_dict: dict[str, Any],
    ar_lookup: dict[str, dict[str, tuple[list, str, str]]],
) -> None:
    """Fill allowed_range_value in each param's constraint dict from ar_lookup.

    ar_lookup is per-platform: {pname: {platform: (value, type, src)}}.
    For each platform, try platform-specific entry first, then common ("") entry.
    """
    for pname, constraint_dict in param_dict.items():
        if not isinstance(constraint_dict, dict):
            continue
        if pname not in ar_lookup:
            continue
        plat_lookup = ar_lookup[pname]
        common_entry = plat_lookup.get("")  # common entry (applies to all)
        for plat, plat_data in constraint_dict.items():
            if not isinstance(plat_data, dict):
                continue
            ar = plat_data.get("allowed_range_value")
            if isinstance(ar, dict) and not ar.get("value"):
                # Platform-specific entry takes precedence over common
                entry = plat_lookup.get(plat, common_entry)
                if entry:
                    value_list, ar_type, src_text = entry
                    ar["value"] = value_list
                    ar["type"] = ar_type
                    if src_text:
                        ar["src_text"] = src_text


def _apply_bool_default(param_dict: dict[str, Any]) -> None:
    """Fill ``[True, False]`` enum for bool params whose AR is still empty.

    After the relation-derived AR fill, unconstrained bool params still carry
    the empty AR hardcoded by ``constraint_assemble_node`` (the
    ``allowed_range_map`` from ``allowed_range_build_node`` is no longer
    consumed).  A bool with no textual restriction genuinely allows both True
    and False, so defaulting here keeps ``tokensIndexFlag`` etc. from shipping
    an empty value range.

    A bool restricted to a single value is left untouched: Pass 2 emits a
    ``pname.range_value == True/False`` relation for it, so ``ar_lookup``
    already filled a non-empty AR and the ``not ar.get("value")`` guard skips.
    """
    for _pname, constraint_dict in param_dict.items():
        if not isinstance(constraint_dict, dict):
            continue
        for _plat, plat_data in constraint_dict.items():
            if not isinstance(plat_data, dict):
                continue
            ptype = plat_data.get("type", {})
            ptype_val = (
                ptype.get("value", "") if isinstance(ptype, dict) else str(ptype)
            )
            if str(ptype_val).lower() != "bool":
                continue
            ar = plat_data.get("allowed_range_value")
            if isinstance(ar, dict) and not ar.get("value"):
                ar["value"] = [True, False]
                ar["type"] = "enum"
                ar["src_text"] = "bool 默认取值 [True, False]"


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

    Records are prepended to each platform's constraint list so the final
    per-platform ordering is:
    ``[parameter_representations..., <other constraints>]``.
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


def _build_ar_lookup_from_io(
    inputs: dict[str, Any], outputs: dict[str, Any],
) -> dict[str, tuple[list, str]]:
    """Build {param_name: (allowed_range_value, type)} from the *filled*
    inputs/outputs (the final AR state, including implicit params).

    Unlike the DB-only lookup built from ``params.param_constraint``, this
    sees implicit params (BS/N/H …) whose allowed_range_value was derived
    from param_relations, so the constraints_in_parameters dedup applies to
    them consistently.
    """
    lookup: dict[str, tuple[list, str]] = {}
    for io_dict in (inputs, outputs):
        for pname, constraint_dict in io_dict.items():
            if not isinstance(constraint_dict, dict):
                continue
            if pname in lookup:
                continue
            for plat_data in constraint_dict.values():
                if not isinstance(plat_data, dict):
                    continue
                ar = plat_data.get("allowed_range_value")
                if isinstance(ar, dict):
                    val = ar.get("value", [])
                    ar_type = ar.get("type", "range")
                    if val:
                        lookup[pname] = (val, ar_type)
                break
    return lookup


def _build_constraints_in_parameters(
    relations: list[dict],
    supported_platforms: list[str],
    params: list[dict],
    ar_lookup: dict[str, tuple[list, str]] | None = None,
) -> dict[str, list[dict]]:
    """Build constraints_in_parameters: {platform: [relation_object]}.

    Deduplicates single-parameter value_dependency constraints when the
    parameter already has a non-empty allowed_range_value (the structured
    range is the canonical representation).

    Args:
        relations: List of param_relation dicts with 'platform' and 'relation_object' fields.
        supported_platforms: List of platform names where is_supported=1.
        params: List of parameter dicts (used to build allowed_range_value lookup
            when *ar_lookup* is not supplied — backward-compatible path).
        ar_lookup: Optional pre-built {param_name: (value, type)} from the filled
            inputs/outputs. When supplied it takes precedence because it also
            covers implicit params (BS/N/H …); when None the lookup is rebuilt
            from ``params.param_constraint`` (DB state).

    Returns:
        Dict mapping platform name to list of relation_object dicts.
        If a relation's platform is empty, it applies to all supported platforms.
        If a relation's platform specifies platforms, it only applies to those
        that are also in supported_platforms.
    """
    from agent.utils.platform_utils import resolve_target_platforms

    # Build {param_name: (allowed_range_value, type)} lookup.
    # Prefer the caller-supplied (filled, implicit-aware) lookup; otherwise
    # fall back to rebuilding from params.param_constraint (DB state).
    if ar_lookup is None:
        ar_lookup = {}
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
        # Exception: bool enum (single bool value like [False]) — the
        # value_dependency expr "x.range_value == False" is fully equivalent
        # to allowed_range_value [False] with type "enum", so dedup is safe
        # and prevents duplication with the self_value_enum injected by FIX-6.
        expr_type = obj.get("expr_type", "")
        rel_params = obj.get("relation_params", [])
        if (
            expr_type == "value_dependency"
            and len(rel_params) == 1
            and rel_params[0] in ar_lookup
        ):
            val, ar_type = ar_lookup[rel_params[0]]
            if ar_type != "enum":
                skipped_count += 1
                continue
            # Bool enum: single bool value — dedup to avoid duplication
            if (
                isinstance(val, list)
                and len(val) == 1
                and isinstance(val[0], bool)
            ):
                skipped_count += 1
                continue

        platform_str = r.get("platform", "")
        targets = resolve_target_platforms(platform_str, supported_platforms)
        # Empty-platform fallback: keep constraints under "common" key
        if not targets:
            targets = ["common"]

        for plat in targets:
            grouped.setdefault(plat, []).append(obj)

    if skipped_count > 0:
        logger.info(
            "assemble_result: deduplicated %d single-param value_dependency "
            "constraints (covered by allowed_range_value)",
            skipped_count,
        )

    return grouped


def _expand_aliases_in_constraints(
    operator_name: str,
    constraints_ip: dict[str, list[dict]],
) -> None:
    """Expand parameter alias shorthand names in constraints_in_parameters.

    Modifies *constraints_ip* in-place. For each constraint, calls
    ``expand_expr()`` to replace shorthand param names (e.g. ``weight``)
    with actual param names (e.g. ``weight1``, ``weight2``).

    Silently skips if the alias map is empty or the operator has no
    aliases defined — does not affect the normal pipeline.
    """
    alias_map = load_alias_map()
    if not alias_map:
        return

    # Quick check: does this operator have any aliases?
    op_map = alias_map.get(operator_name, {})
    default_map = alias_map.get("_default", {})
    if not op_map and not default_map:
        return

    expanded_count = 0
    for plat, constraints in constraints_ip.items():
        for c in constraints:
            expr = c.get("expr", "")
            rp = c.get("relation_params", [])
            if not expr or not rp:
                continue
            # Quick filter: only call expand_expr if any rp entry is an alias
            all_aliases = set(op_map.keys()) | set(default_map.keys())
            if not any(p in all_aliases for p in rp):
                continue
            new_expr, new_rp = expand_expr(operator_name, expr, rp, alias_map)
            if new_expr != expr or new_rp != rp:
                c["expr"] = new_expr
                c["relation_params"] = new_rp
                expanded_count += 1

    if expanded_count > 0:
        logger.info(
            "assemble_result: expanded aliases in %d constraints for %s",
            expanded_count,
            operator_name,
        )


def _has_range_constraint(clist: list[dict], pname: str) -> bool:
    """Check if *pname* already has a self_value_range/enum constraint."""
    for c in clist:
        if pname not in c.get("relation_params", []):
            continue
        if c.get("expr_type") in ("self_value_range", "self_value_enum"):
            return True
    return False


def _build_ar_expr(pname: str, vals: list, ar_type: str) -> str:
    """Build a constraint expr from allowed_range_value.

    R15: use ``pname.range_value`` (not bare pname) for consistency with
    the codebase convention (single_param_constraint.py, implicit_value_constraint.py).
    """
    rv = f"{pname}.range_value"
    if not vals:
        return ""
    # Range type: vals[0] is [lo, hi]
    if ar_type == "range" and isinstance(vals[0], list):
        lo, hi = vals[0][0], vals[0][1]
        parts: list[str] = []
        if lo is not None:
            parts.append(f"{rv} >= {lo}")
        if hi is not None:
            parts.append(f"{rv} <= {hi}")
        return " and ".join(parts) if parts else ""
    # Enum / bool: vals is a flat list
    return f"{rv} in {vals}"


def _inject_allowed_range_as_constraints(
    constraints_ip: dict[str, list[dict]],
    inputs_dict: dict[str, Any],
    outputs_dict: dict[str, Any],
    supported_platforms: list[str],
) -> int:
    """FIX-6: Inject allowed_range_value as self_value_range/enum constraints.

    Converts the structured ``allowed_range_value`` (already filled in
    inputs/outputs) into constraint records, so parameters whose value
    ranges were extracted by ``allowed_range_build`` but never produced as
    LLM relations still appear in ``constraints_in_parameters``.

    R4: src_text comes from the AR's own src_text.
    R5: platforms without an existing constraints_ip key are created.
    R15: expr uses ``pname.range_value`` for codebase consistency.
    """
    from agent.utils.platform_utils import resolve_target_platforms

    injected = 0
    for io_dict in (inputs_dict, outputs_dict):
        for pname, plat_constraints in io_dict.items():
            if not isinstance(plat_constraints, dict):
                continue
            for plat, data in plat_constraints.items():
                if not isinstance(data, dict):
                    continue
                ar = data.get("allowed_range_value", {})
                if not isinstance(ar, dict):
                    continue
                vals = ar.get("value", [])
                ar_type = ar.get("type", "range")
                if not vals:
                    continue
                # Skip if a self_value_range/enum constraint already exists
                if _has_range_constraint(constraints_ip.get(plat, []), pname):
                    continue
                expr = _build_ar_expr(pname, vals, ar_type)
                if not expr:
                    continue
                targets = resolve_target_platforms(plat, supported_platforms)
                # Empty-platform fallback: inject AR constraint under "common"
                if not targets:
                    targets = ["common"]
                ar_src = ar.get("src_text", "")  # R4
                constraint = {
                    "expr_type": (
                        "self_value_range" if ar_type == "range"
                        else "self_value_enum"
                    ),
                    "expr": expr,
                    "relation_params": [pname],
                    "src_text": ar_src,
                }
                for t in targets:
                    if t not in constraints_ip:
                        constraints_ip[t] = []  # R5
                    constraints_ip[t].append(dict(constraint))
                    injected += 1
    return injected


def _filter_invalid_param_refs(
    constraints_ip: dict[str, list[dict]],
    valid_params: set[str],
) -> int:
    """FIX-2: Filter constraints that reference non-existent parameters.

    MUST be called AFTER ``_expand_aliases_in_constraints`` (R1) so that
    shorthand names are already resolved.  *valid_params* MUST include
    external constants (R2) and all cross-function params (R12).
    """
    removed = 0
    for plat, clist in constraints_ip.items():
        kept: list[dict] = []
        for c in clist:
            params = c.get("relation_params", [])
            invalid = [p for p in params if p not in valid_params]
            if invalid:
                logger.warning(
                    "assemble_result: filtering constraint with invalid "
                    "params %s on %s", invalid, plat,
                )
                removed += 1
                continue
            kept.append(c)
        constraints_ip[plat] = kept
    return removed


# ---------------------------------------------------------------------------
# FIX-10: Source citation existence verification
# ---------------------------------------------------------------------------

# HTML tag regex (inline, mirrors constraint_extract._strip_html without the
# cross-module import of a private function).
_HTML_TAG_RE_FIX10 = re.compile(r"<[^>]+>")


def _strip_html_inline(text: str) -> str:
    """Remove HTML tags, preserving inner content.

    Inline implementation matching constraint_extract._strip_html logic,
    avoiding a cross-module reference to a private function.
    """
    if not text:
        return text
    return _HTML_TAG_RE_FIX10.sub("", text).strip()


def _normalize_for_match(text: str) -> str:
    """Normalize text for fuzzy matching: strip HTML -> remove whitespace ->
    remove punctuation -> lowercase.
    """
    text = _strip_html_inline(text)
    text = re.sub(r"[\s]+", "", text)
    text = re.sub(r"[　，、。；：“”‘’（）\[\]\{\},.;:!?()\-]", "", text)
    return text.lower()


def _source_exists_in_doc(
    src: str, doc_norm: str, threshold: float = 0.3,
) -> tuple[bool, float]:
    """Check whether *src* appears in the document.

    Uses the first 80 chars of the normalized src as the match window. If
    src_text exceeds 80 chars, only the first 80 chars are matched (a
    performance trade-off; most src_text values are < 80 chars).
    """
    src_norm = _normalize_for_match(src)
    if not src_norm:
        return True, 1.0
    if src_norm in doc_norm:
        return True, 1.0
    window = src_norm[:80]
    best = 0.0
    step = max(1, len(window) // 4)
    for i in range(0, max(1, len(doc_norm) - len(window) + 1), step):
        chunk = doc_norm[i:i + len(window) + 20]
        ratio = SequenceMatcher(None, window, chunk).ratio()
        best = max(best, ratio)
        if best >= 0.6:  # early-exit optimization, not a separate threshold
            break
    return best >= threshold, best


# expr_type set that skips src_text verification:
# - parameter_representation: src_text is a shape label / source_section_text
#   (deterministically generated, not LLM-fabricated).
# - self_value_range / self_value_enum: src_text may come from FIX-6 AR
#   injection or placeholders ("正则提取:..." / "YAML..."), neither of which
#   is document prose.  Note: LLM-generated self_value_range is also skipped
#   by design (see mixed-fix.html §7.4 known limitations).
_SKIP_SRC_CHECK = frozenset({
    "parameter_representation",
    "self_value_range",
    "self_value_enum",
})


def _filter_fabricated_src_text(
    constraints_ip: dict[str, list[dict]],
    document_text: str,
    *,
    threshold: float = 0.3,
) -> dict[str, int]:
    """FIX-10: Delete constraints whose src_text is absent from the document.

    Empty-document guard: if *document_text* is empty (DB error / missing
    section), verification is skipped entirely to avoid deleting every
    non-skipped constraint.
    """
    # Empty-document guard (blocking safety net)
    if not document_text or not document_text.strip():
        logger.warning(
            "FIX-10: empty document_text, skipping verification "
            "(would otherwise delete all non-skipped constraints)"
        )
        return {"deleted": 0, "kept": 0, "skipped": 0}

    doc_norm = _normalize_for_match(document_text)
    report = {"deleted": 0, "kept": 0, "skipped": 0}
    for plat, clist in constraints_ip.items():
        kept: list[dict] = []
        for c in clist:
            etype = c.get("expr_type", "")
            src = c.get("src_text", "")
            if etype in _SKIP_SRC_CHECK or not src.strip():
                kept.append(c)
                report["skipped"] += 1
                continue
            exists, ratio = _source_exists_in_doc(src, doc_norm, threshold)
            if exists:
                kept.append(c)
                report["kept"] += 1
            else:
                report["deleted"] += 1
                logger.warning(
                    "FIX-10: %s src not found (ratio=%.2f, threshold=%.2f)",
                    plat, ratio, threshold,
                )
        constraints_ip[plat] = kept
    return report


def _deduplicate_constraints_semantic(
    constraints_ip: dict[str, list[dict]],
) -> int:
    """FIX-1: Cross-source semantic dedup.

    When a ``parameter_representation`` constraint and an LLM-generated
    constraint are semantically equivalent (same AST canonical key), keep
    the ``parameter_representation`` and remove the duplicate.
    """
    from agent.utils.expr_validation import _semantic_expr_key

    removed = 0
    for plat, clist in constraints_ip.items():
        # Collect keys from parameter_representation constraints
        repr_keys: set[str] = set()
        for c in clist:
            if c.get("expr_type") == "parameter_representation":
                key = _semantic_expr_key(c.get("expr", ""))
                if key:
                    repr_keys.add(key)
        if not repr_keys:
            continue
        kept: list[dict] = []
        for c in clist:
            if c.get("expr_type") == "parameter_representation":
                kept.append(c)
                continue
            key = _semantic_expr_key(c.get("expr", ""))
            if key and key in repr_keys:
                removed += 1
                continue
            kept.append(c)
        constraints_ip[plat] = kept
    return removed


def _is_dtype_only(expr: str) -> bool:
    """FIX-13: Check if all Attribute accesses in *expr* are .dtype only."""
    import ast as _ast

    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Attribute):
            if node.attr not in ("dtype",):
                return False
    return True


def _normalize_expr_types(
    constraints_ip: dict[str, list[dict]],
) -> None:
    """FIX-13: Normalize expr_type for dtype-only constraints.

    Constraints that only compare ``.dtype`` attributes but are tagged
    ``cross_param_constraint`` are re-tagged as ``type_equality``.
    """
    for plat, clist in constraints_ip.items():
        for c in clist:
            if c.get("expr_type") != "cross_param_constraint":
                continue
            if _is_dtype_only(c.get("expr", "")):
                c["expr_type"] = "type_equality"


def _build_dtype_support(dtype_combos: list[dict]) -> dict[str, list[dict]]:
    """Build dtype_support_description: {platform: [combo]}."""
    grouped: dict[str, list[dict]] = {}
    for dc in dtype_combos:
        plat = dc.get("platform", "common")
        combo = dc.get("combo", {})
        if combo:
            grouped.setdefault(plat, []).append(combo)
    return grouped

