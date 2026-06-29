"""CrossParamConstraint node: deterministically extract cross-parameter constraints.

Generates constraint expressions for patterns that the LLM relation extraction
commonly misses:
  - dtype equality: A.dtype == B.dtype
  - shape equality: A.shape == B.shape
  - TensorList length: len(A) == len(B)
  - alignment/divisibility: A % B == 0

Position: build_single_param_constraint -> [this] -> build_param_constraint
Zero LLM calls.
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
_DIV_RE = re.compile(
    r"(?:必须能被|需要能被|能被)\s*([A-Za-z_]\w*)\s*整除"
    r"|(?:必须是|需要是|是)\s*([A-Za-z_]\w*)\s*(?:的)?(?:整数倍|倍数)"
)
_EXCLUDE = frozenset({"self","input","output","tensor","shape","dtype","format"})


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


def _is_param(name, all_params):
    if not name or len(name) < 2: return False
    if name.lower() in _EXCLUDE: return False
    return name in all_params


def _exists(existing, expr):
    for rel in existing:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try: obj = json.loads(obj)
            except: continue
        if isinstance(obj, dict) and obj.get("expr", "") == expr:
            return True
    return False


def _make_rel(fn, ptype, pa, pb, expr, desc, src):
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


async def cross_param_constraint_node(state):
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    logger.info("CrossParamConstraint: doc_id=%s for %s", doc_id, operator_name)
    if not doc_id:
        return {"error": None}
    try:
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        if not params:
            return {"error": None}
        all_names = {p.get("param_name", "") for p in params if p.get("param_name")}
        existing = await _mcp_client.query_param_relations(doc_id)
        new_rels = []
        for param in params:
            pn = param.get("param_name", "")
            fn = param.get("function_name", "")
            if not pn: continue
            text = _collect_text(param)
            if not text.strip(): continue
            for m in _DTYPE_EQ_RE.finditer(text):
                t = m.group(1) or m.group(2)
                if t and _is_param(t, all_names) and t != pn:
                    expr = f"{pn}.dtype == {t}.dtype"
                    if not _exists(existing, expr):
                        new_rels.append(_make_rel(fn, "dtype", pn, t, expr, f"{pn}的数据类型与{t}一致", m.group(0)))
            for m in _SHAPE_EQ_RE.finditer(text):
                t = m.group(1) or m.group(2)
                if t and _is_param(t, all_names) and t != pn:
                    expr = f"{pn}.shape == {t}.shape"
                    if not _exists(existing, expr):
                        new_rels.append(_make_rel(fn, "shape", pn, t, expr, f"{pn}的shape与{t}一致", m.group(0)))
            for m in _LEN_EQ_RE.finditer(text):
                t = m.group(1)
                if t and _is_param(t, all_names) and t != pn:
                    expr = f"len({pn}) == len({t})"
                    if not _exists(existing, expr):
                        new_rels.append(_make_rel(fn, "shape&value", pn, t, expr, f"{pn}的长度与{t}相同", m.group(0)))
            for m in _DIV_RE.finditer(text):
                t = m.group(1) or m.group(2)
                if t and _is_param(t, all_names) and t != pn:
                    expr = f"{pn} % {t} == 0"
                    if not _exists(existing, expr):
                        new_rels.append(_make_rel(fn, "value", pn, t, expr, f"{pn}必须能被{t}整除", m.group(0)))
        if new_rels:
            merged = existing + new_rels
            await _mcp_client.save_param_relations(doc_id, merged)
            logger.info("CrossParamConstraint: added %d constraints for %s", len(new_rels), operator_name)
        else:
            logger.info("CrossParamConstraint: no new constraints for %s", operator_name)
        return {"cross_param_constraints": new_rels, "error": None}
    except Exception as e:
        logger.exception("CrossParamConstraint failed for %s", operator_name)
        return {"error": str(e)}
