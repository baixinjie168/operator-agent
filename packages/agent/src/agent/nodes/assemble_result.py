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
        # Step 1: Get parsed document (contains product_support)
        parsed = await _mcp_client.get_parsed_by_doc_id(doc_id)
        product_support_raw = ""
        if parsed:
            ps = parsed.get("product_support", [])
            product_support_raw = json.dumps(ps, ensure_ascii=False) if ps else "[]"
            if not operator_name:
                operator_name = parsed.get("operator_name", "")
        else:
            product_support_raw = "[]"

        # Step 2: Query all data by doc_id
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        relations = await _mcp_client.query_param_relations(doc_id)
        signatures = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        platform_support_data = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        return_codes = await _mcp_client.query_return_codes_by_doc_id(doc_id)
        dtype_combos = await _mcp_client.query_dtype_combos_by_doc_id(doc_id)

        # Step 2b: Fetch function_explanation_summary from document_versions
        fn_expl_summary = await _mcp_client.get_function_explanation_summary(doc_id)
        description = fn_expl_summary.get("description", "")

        # Step 3: Build function_explanation JSON
        function_explanation = _build_function_explanation(
            params, relations, signatures, return_codes, dtype_combos,
            description=description,
        )

        function_explanation_raw = json.dumps(function_explanation, ensure_ascii=False)

        # Step 3.5: Build platform_support list (supported platform names only)
        platform_support_list = [
            p["platform_name"]
            for p in platform_support_data
            if p.get("is_supported") == 1
        ]
        platform_support_raw = json.dumps(platform_support_list, ensure_ascii=False)

        # Step 2c: Extract GetWorkspaceSize signature
        workspace_sig = ""
        for sig in signatures:
            if sig.get("function_name", "").endswith("GetWorkspaceSize"):
                workspace_sig = sig.get("full_signature", "")
                break

        # Step 4: Save to constraints_result table
        await _mcp_client.save_constraints_result(
            doc_id=doc_id,
            operator_name=operator_name,
            product_support=product_support_raw,
            platform_support=platform_support_raw,
            function_explanation=function_explanation_raw,
            function_signature=workspace_sig,
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
        fn_relations = [r for r in relations if r.get("function_name") == fn]
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

