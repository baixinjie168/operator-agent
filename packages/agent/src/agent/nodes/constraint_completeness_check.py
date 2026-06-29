"""ConstraintCompletenessCheck node: post-build global completeness check.

Runs after build_param_constraint, before assemble_result.
Performs 4 global completeness checks across all params/relations:
  1. dtype completeness  - every aclTensor* param has dtype constraint
  2. shape completeness  - every aclTensor* param has shape repr/constraint
  3. cross-equality      - param-text equality keywords have matching expr
  4. product coverage    - product-specific constraints cover all platforms

Missing items are either auto-injected (cross-equality) or flagged as
_needs_review for manual inspection. Zero LLM calls - all deterministic.

Position: build_param_constraint (subgraph) -> [this] -> assemble_result
Supersedes the unwired constraint_validation.py (its logic is a subset).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.utils.expr_validation import _semantic_expr_key
from agent.utils.param_validators import is_tensor_type

logger = logging.getLogger(__name__)
_mcp_client = MCPClient()

# Reuse the equality-keyword regexes from the legacy constraint_validation.py
# (Check 3 is a superset of that node's dtype/shape/length scan).
_DTYPE_EQ_RE = re.compile(
    r"(?:数据类型|dtype)\s*(?:与|和|同)\s*([A-Za-z_]\w*)\s*(?:一致|相同|保持一致)"
    r"|(?:与|和|同)\s*([A-Za-z_]\w*)\s*(?:的)?\s*数据类型\s*(?:一致|相同|保持一致)"
)
_SHAPE_EQ_RE = re.compile(
    r"(?:shape|维度)\s*(?:与|和|同)\s*([A-Za-z_]\w*)\s*(?:一致|相同|保持一致)"
    r"|(?:与|和|同)\s*([A-Za-z_]\w*)\s*(?:的)?\s*(?:shape|维度)\s*(?:一致|相同|保持一致)"
)
_LEN_EQ_RE = re.compile(
    r"(?:长度|个数|数量)\s*(?:与|和|同)\s*([A-Za-z_]\w*)\s*(?:相同|一致|保持一致)"
)

_EXCLUDE = frozenset({
    "self", "input", "output", "tensor", "shape", "dtype", "format",
    "weight", "bias", "scale", "offset", "true", "false",
})


def _collect_text(param: dict) -> str:
    """Collect all text fields for a parameter."""
    parts: list[str] = []
    for field in ("usage_notes", "llm_description", "param_desc", "src_content"):
        raw = param.get(field, "") or ""
        if not raw:
            continue
        try:
            p = json.loads(raw) if isinstance(raw, str) and raw.startswith("{") else None
        except Exception:  # noqa: BLE001
            p = None
        if isinstance(p, dict):
            parts.extend(str(v) for v in p.values() if v)
        else:
            parts.append(str(raw))
    return " ".join(parts)


def _has_dtype_constraint(pn: str, relations: list[dict]) -> bool:
    """Check if param has any dtype-related constraint."""
    for r in relations:
        if pn not in r.get("params", []):
            continue
        obj = r.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:  # noqa: BLE001
                continue
        if isinstance(obj, dict):
            et = obj.get("expr_type", "")
            expr = obj.get("expr", "")
            if "dtype" in et or ".dtype" in expr:
                return True
    return False


def _has_shape_constraint(pn: str, relations: list[dict]) -> bool:
    """Check if param has any shape-related constraint or representation."""
    for r in relations:
        if pn not in r.get("params", []):
            continue
        obj = r.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:  # noqa: BLE001
                continue
        if isinstance(obj, dict):
            et = obj.get("expr_type", "")
            expr = obj.get("expr", "")
            if "shape" in et or ".shape" in expr:
                return True
    return False


def _make_eq(pa: str, pb: str, fn: str, expr: str, src: str) -> dict:
    """Build a cross-param equality relation dict.

    Uses platform="" (empty) to match the legacy constraint_validation.py
    and constraint_extract._make_cross_rel behaviour. The MCP server's
    save_param_relations does DELETE+INSERT as-is (no secondary expansion
    of "common"), so platform="" persists as a single generic row -
    consistent with existing cross-param constraints.
    """
    return {
        "function_name": fn,
        "relation_type": "cross_param_constraint",
        "platform": "",
        "description": f"{pa}与{pb}一致",
        "params": [pa, pb],
        "param_optional": {pa: False, pb: False},
        "source_citation": src,
        "relation_object": {
            "expr_type": "cross_param_constraint",
            "expr": expr,
            "relation_params": [pa, pb],
            "src_text": src,
        },
    }


async def constraint_completeness_check_node(
    state: PipelineState,
) -> dict[str, Any]:
    """Post-build global completeness check.

    4 checks: dtype / shape / cross-equality / product-coverage.
    Auto-injects missing cross-equality constraints; flags others
    as _needs_review. Saves supplementary relations to DB.

    Wrapped in try/except - any failure returns {"error": str(e)}
    and does NOT block assemble_result (which just sees fewer supplements).
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    if not doc_id:
        return {"error": None}
    logger.info("CompletenessCheck: doc_id=%s for %s", doc_id, operator_name)

    try:
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        if not params:
            return {"error": None}
        existing = await _mcp_client.query_param_relations(doc_id)
        platform_data = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        supported = [
            p["platform_name"] for p in platform_data
            if p.get("is_supported") == 1
        ]

        all_names = {
            p.get("param_name", "") for p in params if p.get("param_name")
        }

        # R4: inject-dedup using semantic keys (not raw strings) so that
        # x.dtype==y.dtype and y.dtype==x.dtype are recognised as duplicates.
        existing_keys: set[str] = set()
        for rel in existing:
            obj = rel.get("relation_object", {})
            if isinstance(obj, str):
                try:
                    obj = json.loads(obj)
                except Exception:  # noqa: BLE001
                    continue
            if isinstance(obj, dict):
                existing_keys.add(_semantic_expr_key(obj.get("expr", "")))

        missing: list[dict] = []
        review_flags: list[dict] = []

        # --- Check 1: dtype completeness ---
        for param in params:
            pn = param.get("param_name", "")
            ptype = param.get("param_type", "")
            if not pn or not is_tensor_type(ptype):
                continue
            if _has_dtype_constraint(pn, existing + missing):
                continue
            dtype_raw = (
                param.get("data_type", "") or param.get("dtype_desc", "") or ""
            )
            if dtype_raw and dtype_raw != "{}":
                review_flags.append({
                    "param": pn, "check": "dtype",
                    "reason": "has DB dtype but no constraint",
                })
            else:
                review_flags.append({
                    "param": pn, "check": "dtype",
                    "reason": "no dtype info",
                })

        # --- Check 2: shape completeness ---
        for param in params:
            pn = param.get("param_name", "")
            ptype = param.get("param_type", "")
            if not pn or not is_tensor_type(ptype):
                continue
            if not _has_shape_constraint(pn, existing + missing):
                review_flags.append({
                    "param": pn, "check": "shape",
                    "reason": "no shape repr/constraint",
                })

        # --- Check 3: cross-equality (auto-inject) ---
        # Reuse the legacy constraint_validation regex scan; dedup via
        # semantic keys so reversed-operand writes don't double-inject.
        for param in params:
            pn = param.get("param_name", "")
            fn = param.get("function_name", "")
            if not pn:
                continue
            text = _collect_text(param)
            if not text.strip():
                continue
            # dtype equality
            for m in _DTYPE_EQ_RE.finditer(text):
                t = m.group(1) or m.group(2)
                if (t and t in all_names and t != pn
                        and t.lower() not in _EXCLUDE):
                    expr = f"{pn}.dtype == {t}.dtype"
                    skey = _semantic_expr_key(expr)
                    if skey and skey not in existing_keys:
                        missing.append(_make_eq(pn, t, fn, expr, m.group(0)))
                        existing_keys.add(skey)
            # shape equality
            for m in _SHAPE_EQ_RE.finditer(text):
                t = m.group(1) or m.group(2)
                if (t and t in all_names and t != pn
                        and t.lower() not in _EXCLUDE):
                    expr = f"{pn}.shape == {t}.shape"
                    skey = _semantic_expr_key(expr)
                    if skey and skey not in existing_keys:
                        missing.append(_make_eq(pn, t, fn, expr, m.group(0)))
                        existing_keys.add(skey)
            # length equality
            for m in _LEN_EQ_RE.finditer(text):
                t = m.group(1)
                if (t and t in all_names and t != pn
                        and t.lower() not in _EXCLUDE):
                    expr = f"len({pn}) == len({t})"
                    skey = _semantic_expr_key(expr)
                    if skey and skey not in existing_keys:
                        missing.append(_make_eq(pn, t, fn, expr, m.group(0)))
                        existing_keys.add(skey)

        # --- Check 4: product coverage ---
        # platform="" (generic) covers all platforms; product-specific
        # constraints should cover every supported platform.
        if len(supported) > 1:
            param_platforms: dict[str, set[str]] = {}
            for rel in existing:
                plat = rel.get("platform", "")
                for pn in rel.get("params", []):
                    param_platforms.setdefault(pn, set())
                    if plat:
                        param_platforms[pn].add(plat)
                    else:
                        # generic constraint covers all platforms
                        param_platforms[pn].update(supported)
            for pn, plats in param_platforms.items():
                if 0 < len(plats) < len(supported):
                    review_flags.append({
                        "param": pn, "check": "product_coverage",
                        "reason": (
                            f"covers {plats}, missing "
                            f"{set(supported) - plats}"
                        ),
                    })

        # Save supplementary constraints
        if missing:
            merged = existing + missing
            await _mcp_client.save_param_relations(doc_id, merged)
            logger.info(
                "CompletenessCheck: injected %d missing for %s",
                len(missing), operator_name,
            )
        if review_flags:
            logger.warning(
                "CompletenessCheck: %d items need review for %s: %s",
                len(review_flags), operator_name, review_flags[:5],
            )

        return {
            "completeness_report": {
                "injected": len(missing),
                "needs_review": len(review_flags),
                "review_items": review_flags,
            },
            "error": None,
        }
    except Exception as e:
        logger.exception("CompletenessCheck failed for %s", operator_name)
        return {"error": str(e)}
