"""ConstraintValidation node: post-assembly diff check for missing constraints.

Runs after constraint_assemble, before assemble_result.
Scans parameter text for constraint keywords that have no corresponding
expr in constraints_in_parameters, and injects missing ones.

Position: build_param_constraint (subgraph) -> [this] -> assemble_result
Zero LLM calls — all checks are deterministic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)
_mcp_client = MCPClient()

# Patterns where text mentions a constraint but we check if expr exists
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
_UPPER_BOUND_RE = re.compile(
    r"(\w+)\s*(?:<|小于|不超过|不大于)\s*(\d+)"
)
_LOWER_BOUND_RE = re.compile(
    r"(\w+)\s*(?:>|大于|必须大于)\s*(\d+)"
)

_EXCLUDE = frozenset({
    "self", "input", "output", "tensor", "shape", "dtype", "format",
    "weight", "bias", "scale", "offset", "true", "false",
})


def _collect_text(param):
    parts = []
    for field in ("usage_notes", "llm_description", "param_desc"):
        raw = param.get(field, "") or ""
        if not raw: continue
        try:
            p = json.loads(raw) if isinstance(raw, str) and raw.startswith("{") else None
        except: p = None
        if isinstance(p, dict):
            parts.extend(str(v) for v in p.values() if v)
        else:
            parts.append(str(raw))
    return " ".join(parts)


def _has_expr(constraints_ip, platform, expr_substring):
    """Check if any existing constraint expr contains the substring."""
    items = constraints_ip.get(platform, [])
    for item in items:
        expr = item.get("expr", "")
        if expr_substring in expr:
            return True
    return False


async def constraint_validation_node(state):
    """Post-assembly validation: inject missing constraints found in text.

    Reads params from DB, scans text for constraint keywords, checks
    if corresponding exprs exist in constraints_in_parameters (via DB),
    and injects missing ones as new param_relations.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    logger.info("ConstraintValidation: doc_id=%s for %s", doc_id, operator_name)
    if not doc_id:
        return {"error": None}
    try:
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        if not params:
            return {"error": None}
        all_names = {p.get("param_name", "") for p in params if p.get("param_name")}
        existing = await _mcp_client.query_param_relations(doc_id)

        # Build set of existing expr strings for dedup
        existing_exprs = set()
        for rel in existing:
            obj = rel.get("relation_object", {})
            if isinstance(obj, str):
                try: obj = json.loads(obj)
                except: continue
            if isinstance(obj, dict):
                existing_exprs.add(obj.get("expr", ""))

        missing = []

        for param in params:
            pn = param.get("param_name", "")
            fn = param.get("function_name", "")
            if not pn: continue
            text = _collect_text(param)
            if not text.strip(): continue

            # Check dtype equality
            for m in _DTYPE_EQ_RE.finditer(text):
                t = m.group(1) or m.group(2)
                if t and t in all_names and t != pn and t.lower() not in _EXCLUDE:
                    expr = f"{pn}.dtype == {t}.dtype"
                    if expr not in existing_exprs:
                        missing.append(_make(pn, t, fn, expr, f"{pn}的数据类型与{t}一致", m.group(0)))
                        existing_exprs.add(expr)

            # Check shape equality
            for m in _SHAPE_EQ_RE.finditer(text):
                t = m.group(1) or m.group(2)
                if t and t in all_names and t != pn and t.lower() not in _EXCLUDE:
                    expr = f"{pn}.shape == {t}.shape"
                    if expr not in existing_exprs:
                        missing.append(_make(pn, t, fn, expr, f"{pn}的shape与{t}一致", m.group(0)))
                        existing_exprs.add(expr)

            # Check length equality
            for m in _LEN_EQ_RE.finditer(text):
                t = m.group(1)
                if t and t in all_names and t != pn and t.lower() not in _EXCLUDE:
                    expr = f"len({pn}) == len({t})"
                    if expr not in existing_exprs:
                        missing.append(_make(pn, t, fn, expr, f"{pn}的长度与{t}相同", m.group(0)))
                        existing_exprs.add(expr)

        if missing:
            merged = existing + missing
            await _mcp_client.save_param_relations(doc_id, merged)
            logger.info("ConstraintValidation: injected %d missing constraints for %s", len(missing), operator_name)
        else:
            logger.info("ConstraintValidation: no missing constraints for %s", operator_name)

        return {"validation_report": {"missing_count": len(missing)}, "error": None}
    except Exception as e:
        logger.exception("ConstraintValidation failed for %s", operator_name)
        return {"error": str(e)}


def _make(pa, pb, fn, expr, desc, src):
    return {
        "function_name": fn,
        "relation_type": "cross_param_constraint",
        "platform": "",
        "description": desc,
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
