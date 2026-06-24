"""FetchParamData node: one-shot DB queries + shared index building."""

from __future__ import annotations

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.build_param_constraint.state import BuildParamConstraintState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def fetch_param_data_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Query all data sources and build shared indexes.

    Performs 6 DB queries in sequence, then builds indexes used by the
    parallel downstream nodes (dimensions_build, allowed_range_build,
    attrs_build).
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("FetchParamData: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("FetchParamData: no doc_id, skipping")
        return {"params": [], "error": None}

    try:
        # 6 DB queries
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        sigs = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        dtype_combos = await _mcp_client.query_dtype_combos_by_doc_id(doc_id)
        relations = await _mcp_client.query_param_relations(doc_id)
        constraints_section = await _mcp_client.get_section(doc_id, "constraints")

        if not params:
            logger.info("FetchParamData: no parameters for doc_id=%s", doc_id)
            return {"params": [], "error": None}

        # Build indexes
        sig_type_map: dict[str, str] = {}
        all_sig_param_names: set[str] = set()
        for sig in sigs:
            for p in sig.get("parameters", []):
                key = f"{sig['function_name']}::{p['name']}"
                sig_type_map[key] = p.get("type", "")
                all_sig_param_names.add(p["name"])

        # dtype_by_platform: platform -> param_name -> list[str]
        dtype_by_platform: dict[str, dict[str, set[str]]] = {}
        for combo in dtype_combos:
            plat = combo.get("platform", "common")
            dtype_by_platform.setdefault(plat, {})
            for pname, dtype_val in combo.get("combo", {}).items():
                dtype_by_platform[plat].setdefault(pname, set())
                if isinstance(dtype_val, str) and "/" in dtype_val:
                    for d in dtype_val.split("/"):
                        dtype_by_platform[plat][pname].add(d.strip())
                else:
                    dtype_by_platform[plat][pname].add(str(dtype_val))

        # Convert sets to sorted lists for JSON serialization
        dtype_by_platform_lists: dict[str, dict[str, list[str]]] = {}
        for plat, params_map in dtype_by_platform.items():
            dtype_by_platform_lists[plat] = {
                pn: sorted(dtypes) for pn, dtypes in params_map.items()
            }

        supported_platforms = [
            p["platform_name"] for p in platforms if p.get("is_supported") == 1
        ]

        constraints_text = (constraints_section or {}).get("content", "") or ""

        logger.info(
            "FetchParamData: %d params, %d sigs, %d platforms, %d combos (doc_id=%s)",
            len(params), len(sigs), len(supported_platforms),
            len(dtype_combos), doc_id,
        )

        return {
            "params": params,
            "sig_type_map": sig_type_map,
            "all_sig_param_names": sorted(all_sig_param_names),
            "dtype_by_platform": dtype_by_platform_lists,
            "supported_platforms": supported_platforms,
            "constraints_text": constraints_text,
            "param_relations": relations,
        }

    except Exception as e:
        logger.exception("FetchParamData failed for %s", operator_name)
        return {"params": [], "error": str(e)}
