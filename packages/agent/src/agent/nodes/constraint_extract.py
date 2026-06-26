"""ConstraintExtract node: unified constraint extraction (Pass 1-3).

Consolidates deterministic regex patterns from 4 existing nodes into a
single node with 3 Passes:

  Pass 1 - Cross-parameter constraints (dtype/shape equality, length, divisibility)
           Merges: cross_param_constraint + constraint_validation

  Pass 2 - Single-parameter constraints (empty tensor, tensorlist consistency,
           shape upper bound, bool restriction, string length)
           Merges: single_param_constraint Layer 1 + Layer 1b

  Pass 3 - Implicit variable value constraints (K1<65536, H=32/64, even)
           Merges: implicit_value_constraint

Each Pass runs independently with try/except - Pass failure does not affect
other Passes.  All results are deduplicated against existing param_relations
and only new constraints are saved.

Position: replaces constraint_validation in the graph (after build_param_constraint,
before assemble_result).  Coexists with cross_param_constraint and
build_single_param_constraint - their dedup logic prevents duplicates.

Zero LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.build_param_constraint._helpers import _normalize_type
from agent.nodes.state import PipelineState
from agent.utils.platform_utils import expand_common_in_relations
from agent.runtime.context import get_context
from agent.runtime.events import EventType, Span, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


# ---------------------------------------------------------------------------
# Text collection (unified - union of all fields used by source nodes)
# ---------------------------------------------------------------------------


def _collect_param_text(param: dict) -> str:
    """Collect all available text for a parameter from DB fields.

    Union of fields used by cross_param_constraint, single_param_constraint,
    and constraint_validation.
    """
    parts: list[str] = []
    for field_name in ("param_desc", "llm_description", "src_content", "usage_notes"):
        raw = (param.get(field_name) or "").strip()
        if not raw:
            continue
        # usage_notes / llm_description may be stored as JSON {platform: text}
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.startswith("{") else None
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            parts.extend(str(v) for v in parsed.values() if v)
        else:
            parts.append(raw)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EXCLUDE_NAMES = frozenset({
    "self", "input", "output", "tensor", "shape", "dtype", "format",
    "weight", "bias", "scale", "offset", "true", "false",
})


def _is_param(name: str, all_names: set[str]) -> bool:
    """Check if *name* is a real parameter (not a generic word)."""
    if not name or len(name) < 2:
        return False
    if name.lower() in _EXCLUDE_NAMES:
        return False
    return name in all_names


def _expr_exists(existing: list[dict], expr: str) -> bool:
    """Check if an expr string already exists in param_relations."""
    for rel in existing:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(obj, dict) and obj.get("expr", "") == expr:
            return True
    return False


def _make_cross_rel(
    fn: str, ptype: str, pa: str, pb: str,
    expr: str, desc: str, src: str,
) -> dict:
    """Build a cross-parameter relation dict."""
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


def _make_self_rel(
    fn: str, pname: str, expr: str, desc: str, src: str,
    expr_type: str = "self_constraint",
) -> dict:
    """Build a single-parameter relation dict."""
    return {
        "function_name": fn,
        "relation_type": "self_constraint",
        "platform": "",
        "description": desc,
        "params": [pname],
        "param_optional": {pname: False},
        "source_citation": src,
        "relation_object": {
            "expr_type": expr_type,
            "expr": expr,
            "relation_params": [pname],
            "src_text": src,
        },
    }


# ===================================================================
# Pass 1: Cross-parameter constraints (deterministic regex)
# Merges: cross_param_constraint.py + constraint_validation.py
# ===================================================================

# dtype equality: "数据类型与query一致" / "与query的数据类型一致"
_P1_DTYPE_EQ_RE = re.compile(
    r"(?:数据类型|dtype)\s*(?:与|和|同)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:一致|相同|保持一致)"
    r"|(?:与|和|同)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:的)?\s*数据类型\s*(?:一致|相同|保持一致)"
)
# shape equality: "shape与query一致" / "与query的维度一致"
_P1_SHAPE_EQ_RE = re.compile(
    r"(?:shape|维度)\s*(?:与|和|同)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:一致|相同|保持一致)"
    r"|(?:与|和|同)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:的)?\s*(?:shape|维度)\s*(?:一致|相同|保持一致)"
)
# length equality: "长度与query相同"
_P1_LEN_EQ_RE = re.compile(
    r"(?:长度|个数|数量)\s*(?:与|和|同)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:相同|一致|保持一致)"
)
# divisibility: "必须能被n整除" / "必须是n的整数倍"
_P1_DIV_RE = re.compile(
    r"(?:必须能被|需要能被|能被)\s*([A-Za-z_][A-Za-z0-9_]*)\s*整除"
    r"|(?:必须是|需要是|是)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:的)?(?:整数倍|倍数)"
)


def _pass1_cross_param(
    params: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Pass 1: Extract cross-parameter constraints via regex.

    Scans each parameter text for dtype/shape equality, length equality,
    and divisibility patterns.  Deduplicates against existing relations.
    """
    all_names = {p.get("param_name", "") for p in params if p.get("param_name")}
    new_rels: list[dict] = []

    for param in params:
        pn = param.get("param_name", "")
        fn = param.get("function_name", "")
        if not pn:
            continue
        text = _collect_param_text(param)
        if not text.strip():
            continue

        # dtype equality
        for m in _P1_DTYPE_EQ_RE.finditer(text):
            t = m.group(1) or m.group(2)
            if t and _is_param(t, all_names) and t != pn:
                expr = f"{pn}.dtype == {t}.dtype"
                if not _expr_exists(existing, expr):
                    new_rels.append(_make_cross_rel(
                        fn, "dtype", pn, t, expr,
                        f"{pn}的数据类型与{t}一致", m.group(0),
                    ))

        # shape equality
        for m in _P1_SHAPE_EQ_RE.finditer(text):
            t = m.group(1) or m.group(2)
            if t and _is_param(t, all_names) and t != pn:
                expr = f"{pn}.shape == {t}.shape"
                if not _expr_exists(existing, expr):
                    new_rels.append(_make_cross_rel(
                        fn, "shape", pn, t, expr,
                        f"{pn}的shape与{t}一致", m.group(0),
                    ))

        # length equality
        for m in _P1_LEN_EQ_RE.finditer(text):
            t = m.group(1)
            if t and _is_param(t, all_names) and t != pn:
                expr = f"len({pn}) == len({t})"
                if not _expr_exists(existing, expr):
                    new_rels.append(_make_cross_rel(
                        fn, "shape&value", pn, t, expr,
                        f"{pn}的长度与{t}相同", m.group(0),
                    ))

        # divisibility
        for m in _P1_DIV_RE.finditer(text):
            t = m.group(1) or m.group(2)
            if t and _is_param(t, all_names) and t != pn:
                expr = f"{pn} % {t} == 0"
                if not _expr_exists(existing, expr):
                    new_rels.append(_make_cross_rel(
                        fn, "value", pn, t, expr,
                        f"{pn}必须能被{t}整除", m.group(0),
                    ))

    return new_rels


# ===================================================================
# Pass 2: Single-parameter constraints (deterministic regex)
# Merges: single_param_constraint.py Layer 1 + Layer 1b
# ===================================================================

_P2_EMPTY_TENSOR_RE = re.compile(
    r"不支持\s*空\s*Tensor|不允许\s*空\s*Tensor"
)
_P2_TENSORLIST_DTYPE_RE = re.compile(
    r"该参数中所有\s*Tensor\s*的数据类型保持一致"
)
_P2_TENSORLIST_FORMAT_RE = re.compile(
    r"该参数中所有\s*Tensor\s*的数据格式保持一致"
)
_P2_TENSORLIST_SHAPE_RE = re.compile(
    r"该参数中所有\s*Tensor\s*的\s*shape\s*保持一致"
    r"|该参数中所有\s*Tensor\s*的维度保持一致"
)
_P2_SHAPE_UPPER_RE = re.compile(
    r"Tensor\s*维度超过\s*(\d+)\s*维"
    r"|维度超过\s*(\d+)\s*维"
    r"|shape\s*维度不高于\s*(\d+)\s*维"
    r"|维度不高于\s*(\d+)\s*维"
    r"|shape\s*支持\s*0[-~](\d+)\s*维"
)
_P2_BOOL_NOT_TRUE_RE = re.compile(
    r"不支持.*(?:配|设|置).*True|暂不支持.*True|不能.*True|仅支持.*False|只支持.*False",
    re.IGNORECASE,
)
_P2_BOOL_NOT_FALSE_RE = re.compile(
    r"不支持.*(?:配|设|置).*False|暂不支持.*False|不能.*False|仅支持.*True|只支持.*True",
    re.IGNORECASE,
)
_P2_STRING_LENGTH_RE = re.compile(
    r"字符串长度\s*(?:要求|限制|范围)\s*[\(\(]\s*(\d+)\s*[,，]\s*(\d+)\s*[\)\)]"
)


def _is_already_covered_self(
    pname: str,
    expr_type: str,
    existing: list[dict],
) -> bool:
    """Check if an existing relation already covers this self-constraint."""
    for r in existing:
        if pname not in r.get("params", []):
            continue
        obj = r.get("relation_object", {})
        expr = obj.get("expr", "")
        if not expr:
            continue

        if expr_type == "self_shape_nonempty":
            if "d > 0" in expr and pname in expr:
                return True
        elif "consistency" in expr_type:
            if f"{pname}[0]" in expr:
                return True
        elif expr_type == "self_shape_upper_bound":
            if f"len({pname}.shape)" in expr:
                return True
        elif expr_type == "self_value_dependency":
            if f"{pname}.range_value" in expr:
                return True
        elif expr_type == "self_string_length":
            if f"len({pname})" in expr and ("and" in expr or "or" in expr):
                return True

    return False


def _extract_numeric_group(match: re.Match) -> str:
    """Return the first non-None numeric capture group from *match*."""
    for g in match.groups():
        if g and g.isdigit():
            return g
    return ""


def _pass2_single_param(
    params: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Pass 2: Extract single-parameter constraints via regex.

    Matches empty tensor, tensorlist consistency, shape upper bound,
    bool restriction, and string length patterns.
    """
    new_rels: list[dict] = []

    for param in params:
        pn = param.get("param_name", "")
        fn = param.get("function_name", "")
        ptype = param.get("param_type", "")
        if not pn:
            continue
        text = _collect_param_text(param)
        if not text.strip():
            continue

        # --- A. Empty Tensor ---
        m_empty = _P2_EMPTY_TENSOR_RE.search(text)
        if m_empty:
            et = "self_shape_nonempty"
            if not _is_already_covered_self(pn, et, existing):
                expr = f"all(d > 0 for d in {pn}.shape)"
                new_rels.append(_make_self_rel(
                    fn, pn, expr,
                    f"{pn} 不支持空Tensor，所有维度必须大于0",
                    m_empty.group(0), et,
                ))

        # --- B. TensorList internal consistency ---
        if "aclTensorList" in ptype:
            m_dt = _P2_TENSORLIST_DTYPE_RE.search(text)
            if m_dt:
                et = "self_dtype_consistency"
                if not _is_already_covered_self(pn, et, existing):
                    expr = (
                        f"all({pn}[i].dtype == {pn}[0].dtype"
                        f" for i in range(len({pn})))"
                    )
                    new_rels.append(_make_self_rel(
                        fn, pn, expr,
                        f"{pn} 中所有Tensor的数据类型必须保持一致",
                        m_dt.group(0), et,
                    ))
            m_fm = _P2_TENSORLIST_FORMAT_RE.search(text)
            if m_fm:
                et = "self_format_consistency"
                if not _is_already_covered_self(pn, et, existing):
                    expr = (
                        f"all({pn}[i].format == {pn}[0].format"
                        f" for i in range(len({pn})))"
                    )
                    new_rels.append(_make_self_rel(
                        fn, pn, expr,
                        f"{pn} 中所有Tensor的数据格式必须保持一致",
                        m_fm.group(0), et,
                    ))
            m_sh = _P2_TENSORLIST_SHAPE_RE.search(text)
            if m_sh:
                et = "self_shape_consistency"
                if not _is_already_covered_self(pn, et, existing):
                    expr = (
                        f"all({pn}[i].shape == {pn}[0].shape"
                        f" for i in range(len({pn})))"
                    )
                    new_rels.append(_make_self_rel(
                        fn, pn, expr,
                        f"{pn} 中所有Tensor的shape必须保持一致",
                        m_sh.group(0), et,
                    ))

        # --- C. Shape upper bound ---
        m_ub = _P2_SHAPE_UPPER_RE.search(text)
        if m_ub:
            et = "self_shape_upper_bound"
            if not _is_already_covered_self(pn, et, existing):
                n_val = _extract_numeric_group(m_ub)
                expr = f"len({pn}.shape) <= {n_val}"
                new_rels.append(_make_self_rel(
                    fn, pn, expr,
                    f"{pn} 的维度数不能超过{n_val}",
                    m_ub.group(0), et,
                ))

        # --- D. Bool value restriction ---
        if "bool" in ptype.lower():
            m_bt = _P2_BOOL_NOT_TRUE_RE.search(text)
            if m_bt:
                et = "self_value_dependency"
                if not _is_already_covered_self(pn, et, existing):
                    expr = f"{pn}.range_value == False"
                    new_rels.append(_make_self_rel(
                        fn, pn, expr,
                        f"{pn} 暂不支持配为True，只能为False",
                        m_bt.group(0), et,
                    ))
            m_bf = _P2_BOOL_NOT_FALSE_RE.search(text)
            if m_bf:
                et = "self_value_dependency"
                if not _is_already_covered_self(pn, et, existing):
                    expr = f"{pn}.range_value == True"
                    new_rels.append(_make_self_rel(
                        fn, pn, expr,
                        f"{pn} 暂不支持配为False，只能为True",
                        m_bf.group(0), et,
                    ))

        # --- E. String length (char* params) ---
        normalized = _normalize_type(ptype)
        if normalized in ("char", "const char"):
            m_sl = _P2_STRING_LENGTH_RE.search(text)
            if m_sl:
                et = "self_string_length"
                if not _is_already_covered_self(pn, et, existing):
                    min_val = int(m_sl.group(1))
                    max_val = int(m_sl.group(2))
                    expr = f"len({pn}) > {min_val} and len({pn}) < {max_val}"
                    new_rels.append(_make_self_rel(
                        fn, pn, expr,
                        f"{pn} 字符串长度要求({min_val}, {max_val})",
                        m_sl.group(0), et,
                    ))

    return new_rels


# ===================================================================
# Pass 3: Implicit variable value constraints (deterministic regex)
# Merges: implicit_value_constraint.py
# ===================================================================

_P3_VALUE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # "K1 < 65536" / "K1小于65536" / "K1不超过65536"
    (
        re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<|小于|不超过|不大于)\s*(\d+)"),
        "{var} < {n}",
        "self_value_range",
    ),
    # "M <= 1024" / "M最大为1024"
    (
        re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|最大为|最多为)\s*(\d+)"),
        "{var} <= {n}",
        "self_value_range",
    ),
    # "N > 0" / "N大于0"
    (
        re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:>|大于|必须大于)\s*(\d+)"),
        "{var} > {n}",
        "self_value_range",
    ),
    # "X >= 1" / "X最小为1"
    (
        re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:>=|最小为|至少为)\s*(\d+)"),
        "{var} >= {n}",
        "self_value_range",
    ),
    # "H=32/64" / "H取值32/64" / "H只支持32、64"
    (
        re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|为|取值|只支持|仅支持)\s*(\d+(?:\s*[/、，,]\s*\d+)+)"),
        "{var}.range_value in [{values}]",
        "self_value_enum",
    ),
    # "S是偶数" / "S必须是偶数"
    (
        re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:必须)?是偶数"),
        "{var} % 2 == 0",
        "self_alignment",
    ),
]


def _extract_enum_values(raw: str) -> str:
    """Parse '32/64' or '32、64' into '32, 64'."""
    parts = re.split(r"[/、，,]", raw)
    return ", ".join(p.strip() for p in parts if p.strip().isdigit())


def _pass3_implicit_value(
    sections_text: str,
    implicit_params: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Pass 3: Extract value constraints for implicit dimension variables.

    Scans constraint section text for patterns like "K1<65536" and
    generates constraint expressions for known implicit variables.
    """
    if not sections_text.strip() or not implicit_params:
        return []

    # Collect known implicit variable names (exclude constants/external)
    var_names = {
        m["var_name"] for m in implicit_params
        if m.get("var_name")
        and not m.get("is_constant")
        and not m.get("is_external_constant")
        and not m.get("is_quantization_type")
    }
    if not var_names:
        return []

    constraints: list[dict] = []
    seen_exprs: set[str] = set()

    # Collect exprs from existing for dedup
    for rel in existing:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(obj, dict):
            seen_exprs.add(obj.get("expr", ""))

    for pattern, template, expr_type in _P3_VALUE_PATTERNS:
        for m in pattern.finditer(sections_text):
            var = m.group(1)
            if var not in var_names:
                continue

            # Build expr from template
            if "{values}" in template:
                values_str = _extract_enum_values(m.group(2))
                expr = template.format(var=var, values=values_str)
            elif "{n}" in template:
                expr = template.format(var=var, n=m.group(2))
            else:
                expr = template.format(var=var)

            # Dedup
            if expr in seen_exprs:
                continue
            seen_exprs.add(expr)

            # Get source context (+/-50 chars)
            start = max(0, m.start() - 50)
            end = min(len(sections_text), m.end() + 50)
            src_text = sections_text[start:end].strip()

            constraints.append({
                "function_name": "",  # implicit params are not function-scoped
                "relation_type": "self_constraint",
                "platform": "",
                "description": expr,
                "params": [var],
                "param_optional": {var: False},
                "source_citation": src_text,
                "relation_object": {
                    "expr_type": expr_type,
                    "expr": expr,
                    "relation_params": [var],
                    "src_text": src_text,
                },
            })

    return constraints


# ===================================================================
# Pass 4: Value range extraction (deterministic regex + YAML + char* enum)
# Merges: allowed_range_build Phase 1 + Phase 1b + single_param_constraint Layer 0
# ===================================================================

# Reuse patterns from allowed_range_build (lazy import to avoid circular deps)


def _try_deterministic_range(text: str):
    """Try deterministic extraction of range from text.

    Returns (value_list, ar_type) or None.
    """
    from agent.nodes.build_param_constraint.allowed_range_build import (
        _RANGE_PATTERNS,
    )
    for pattern, extractor in _RANGE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            try:
                result = extractor(m)
                if isinstance(result, tuple) and len(result) == 2:
                    value_list, ar_type = result
                    if value_list:
                        return value_list, ar_type
                elif isinstance(result, list) and result:
                    return result, "range"
            except (ValueError, IndexError):
                continue
    return None


def _extract_param_sentences(text: str, param_name: str) -> str:
    """Extract sentences from text that mention param_name."""
    from agent.nodes.build_param_constraint.allowed_range_build import (
        _extract_param_sentences as _eps,
    )
    return _eps(text, param_name)


def _range_to_expr(pname: str, value_list: list, ar_type: str) -> str:
    """Convert a structured range/enum value to a Python expr string."""
    if ar_type == "enum":
        # Enum: [[v,v], ...] or [v, v, ...]
        flat_vals: list[str] = []
        for item in value_list:
            if isinstance(item, list) and len(item) == 2:
                flat_vals.append(str(item[0]))
            elif isinstance(item, (int, float)):
                flat_vals.append(str(int(item)))
            elif isinstance(item, str):
                flat_vals.append(f'"{item}"')
        if flat_vals:
            return f"{pname}.range_value in [{', '.join(flat_vals)}]"
    else:
        # Range: [[min, max]]
        for item in value_list:
            if isinstance(item, list) and len(item) == 2:
                lo, hi = item[0], item[1]
                parts: list[str] = []
                if lo is not None:
                    parts.append(f"{pname} >= {lo}")
                if hi is not None:
                    parts.append(f"{pname} <= {hi}")
                if parts:
                    return " and ".join(parts)
    return ""


def _pass4_value_range(
    params: list[dict],
    existing: list[dict],
    constraints_text: str,
) -> list[dict]:
    """Pass 4: Extract value range constraints for scalar parameters.

    Merges:
    - allowed_range_build Phase 1 (deterministic regex)
    - allowed_range_build Phase 1b (YAML semantic rules for scalars)
    - single_param_constraint Layer 0 (YAML semantic rules for tensors)
    - char* enum passthrough from DB (set by allowed_range_extract node 4h)
    """
    from agent.nodes.build_param_constraint._helpers import _normalize_type
    from agent.utils.param_validators import is_bool_type, is_tensor_type
    from agent.utils.semantic_rules import (
        get_allowed_range_for_scalar,
        get_expr_for_tensor,
    )

    new_rels: list[dict] = []

    for param in params:
        pn = param.get("param_name", "")
        fn = param.get("function_name", "")
        ptype = param.get("param_type", "")
        if not pn:
            continue

        # Skip bool (handled in Pass 2) and tensor (no value range)
        if is_bool_type(ptype) or is_tensor_type(ptype):
            continue

        # Check if already has a value range constraint in existing
        already_has = False
        for r in existing:
            if pn in r.get("params", []):
                obj = r.get("relation_object", {})
                et = obj.get("expr_type", "")
                if et in ("self_value_range", "self_value_enum"):
                    already_has = True
                    break
        if already_has:
            continue

        normalized = _normalize_type(ptype)
        llm_desc = param.get("llm_description", "") or ""
        param_desc = param.get("param_desc", "") or ""
        combined_text = f"{llm_desc}\n{param_desc}"

        # --- char* enum passthrough from DB (node 4h) ---
        if normalized in ("char", "const char"):
            ar_raw = param.get("allowed_range_value", "") or ""
            if ar_raw.strip() and ar_raw.strip() != "[]":
                try:
                    ar_list = json.loads(ar_raw) if isinstance(ar_raw, str) else ar_raw
                except (json.JSONDecodeError, TypeError):
                    ar_list = []
                if isinstance(ar_list, list) and ar_list:
                    # Extract enum values from platform entries
                    enum_vals: list[str] = []
                    for item in ar_list:
                        if isinstance(item, dict):
                            raw = item.get("allowed_range_value", "")
                            if raw:
                                from agent.nodes.build_param_constraint._helpers import _split_csv
                                vals = _split_csv(raw)
                                enum_vals.extend(vals)
                    if enum_vals:
                        expr = f"{pn}.range_value in [{', '.join(repr(v) for v in enum_vals)}]"
                        src = f"char*枚举值: {', '.join(enum_vals)}"
                        new_rels.append(_make_self_rel(
                            fn, pn, expr,
                            f"{pn} 取值范围: {', '.join(enum_vals)}",
                            src, "self_value_enum",
                        ))
                        continue

        # --- Skip aclIntArray (LLM handles) ---
        if "aclIntArray" in ptype:
            continue

        # --- Phase 1: Deterministic regex ---
        det = _try_deterministic_range(combined_text)
        if det is None:
            # Search in constraints text (param-specific sentences)
            param_sentences = _extract_param_sentences(constraints_text, pn)
            if param_sentences:
                det = _try_deterministic_range(param_sentences)

        if det is not None:
            det_value, det_type = det
            expr = _range_to_expr(pn, det_value, det_type)
            if expr:
                src = f"正则提取: {det_type}"
                new_rels.append(_make_self_rel(
                    fn, pn, expr,
                    f"{pn} 取值范围({det_type}): {det_value}",
                    src,
                    "self_value_enum" if det_type == "enum" else "self_value_range",
                ))
                continue

        # --- Phase 1b: YAML semantic rules ---
        yaml_search_text = f"{combined_text}\n{constraints_text}"
        yaml_ar = get_allowed_range_for_scalar(yaml_search_text, pn)
        if yaml_ar:
            expr = _range_to_expr(pn, yaml_ar, "range")
            if expr:
                new_rels.append(_make_self_rel(
                    fn, pn, expr,
                    f"{pn} 取值范围(YAML): {yaml_ar}",
                    "YAML语义规则",
                    "self_value_range",
                ))
                continue

        # --- Layer 0: YAML for Tensor element-level (from single_param_constraint) ---
        if "aclTensor" in ptype:
            result = get_expr_for_tensor(combined_text, pn)
            if result and result.get("confidence") == "high":
                expr = result.get("expr", "")
                if expr:
                    desc = result.get("description", "")
                    new_rels.append(_make_self_rel(
                        fn, pn, expr, desc,
                        desc or "YAML Tensor规则",
                        result.get("expr_type", "self_value_range"),
                    ))

    return new_rels


# ===================================================================
# Pass 5: LLM fallback (relation discovery + expr generation)
# Merges: extract_ws/exe (agent_loop) + single_param_constraint Layer 2
# ===================================================================


async def _pass5_llm_fallback(
    state: PipelineState,
    params: list[dict],
    existing: list[dict],
    sections_text: str,
    ws_content: str,
    exe_content: str,
) -> list[dict]:
    """Pass 5: LLM-based relation discovery for uncovered parameters.

    Reuses extract_relations_agent (from agent_loop.py) to discover new
    relations from section content.  New relations get expr generated via
    the existing build_param_relations logic.

    Only runs if there are uncovered parameters (not in any existing relation).
    """
    from agent.utils.param_validators import EXCLUDED_PARAMS
    from agent.core.llm import create_llm

    # Determine uncovered params
    covered: set[str] = set()
    for r in existing:
        for p in r.get("params", []):
            covered.add(p)

    param_names = [
        p["param_name"] for p in params
        if p.get("param_name") and p["param_name"] not in EXCLUDED_PARAMS
    ]
    uncovered = [n for n in param_names if n not in covered]

    if not uncovered and not sections_text.strip():
        logger.info("ConstraintExtract Pass 5: no uncovered params, skipping")
        return []

    # Build existing expr set for dedup
    existing_exprs: set[str] = set()
    existing_descs: set[str] = set()
    for r in existing:
        obj = r.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(obj, dict):
            existing_exprs.add(obj.get("expr", ""))
        existing_descs.add(r.get("description", ""))

    # Run extract_relations_agent on ws + exe content
    new_rels: list[dict] = []
    try:
        from agent.nodes.param_relation_extract.agent_loop import extract_relations_agent

        llm = create_llm()
        implicit_params = state.get("implicit_params", [])

        all_discovered: list[dict] = []

        # Process ws and exe sections separately (same as extract_ws/exe)
        for content, label in [(ws_content, "ws"), (exe_content, "exe")]:
            if not content.strip():
                continue
            try:
                relations, report = await extract_relations_agent(
                    content, param_names, llm,
                    implicit_params=implicit_params,
                )
                logger.info(
                    "ConstraintExtract Pass 5 (%s): %d relations, coverage=%s",
                    label, len(relations), report.get("coverage", ""),
                )
                all_discovered.extend(relations)
            except Exception:
                logger.warning("ConstraintExtract Pass 5 (%s) failed", label)

        # Dedup against existing by description
        for rel in all_discovered:
            desc = rel.get("description", "")
            if desc and desc in existing_descs:
                continue
            # Convert to param_relations format
            fn = rel.get("function_name", "")
            new_rels.append({
                "function_name": fn,
                "relation_type": rel.get("relation_type", ""),
                "platform": rel.get("platform", ""),
                "description": desc,
                "params": rel.get("params", []),
                "param_optional": rel.get("param_optional", {}),
                "source_citation": rel.get("source_citation", ""),
                "relation_object": {
                    "expr_type": rel.get("expr_type", ""),
                    "expr": "",  # expr will be generated below
                    "relation_params": rel.get("params", []),
                    "src_text": rel.get("source_citation", ""),
                },
            })
            existing_descs.add(desc)

    except Exception:
        logger.exception("ConstraintExtract Pass 5: agent loop failed")
        return []

    if not new_rels:
        logger.info("ConstraintExtract Pass 5: no new relations discovered")
        return []

    # Generate expr for new relations using existing build_param_relations logic
    try:
        from agent.nodes.build_param_relations import (
            _batch_extract_relation_objects,
            _format_signatures,
            _build_param_shapes_text,
        )
        from agent.nodes.param_relation_extract.prompts import (
            format_implicit_params_context,
        )

        sigs = await _mcp_client.query_function_signatures_by_doc_id(
            state.get("doc_id", 0)
        )
        signatures_text = _format_signatures(sigs, params)
        param_shapes_text = _build_param_shapes_text(params)
        implicit_params = state.get("implicit_params", [])
        implicit_text = (
            format_implicit_params_context(implicit_params) if implicit_params else ""
        )

        # Platform constants
        platform_constants = state.get("platform_constants", [])
        external_const_names = {
            pc["const_name"] for pc in platform_constants if pc.get("const_name")
        }
        for m in implicit_params:
            if m.get("is_external_constant"):
                external_const_names.add(m["var_name"])
        implicit_param_names = {
            m["var_name"] for m in implicit_params
            if not m.get("is_external_constant") and not m.get("is_constant")
        }

        llm_results = await _batch_extract_relation_objects(
            new_rels, signatures_text, param_shapes_text,
            implicit_params_text=implicit_text,
            external_constants=external_const_names,
            implicit_param_names=implicit_param_names,
        )

        # Update new_rels with generated expr
        for rel, llm_out in zip(new_rels, llm_results):
            obj = rel["relation_object"]
            obj["expr"] = llm_out.get("expr", "")
            obj["expr_type"] = llm_out.get("expr_type", obj.get("expr_type", ""))

    except Exception:
        logger.exception("ConstraintExtract Pass 5: expr generation failed")
        # Relations are still saved with empty expr

    logger.info(
        "ConstraintExtract Pass 5: %d new relations (with expr=%d)",
        len(new_rels),
        sum(1 for r in new_rels if r["relation_object"].get("expr")),
    )

    return new_rels


# ===================================================================
# Node entry point
# ===================================================================


async def constraint_extract_node(state: PipelineState) -> dict[str, Any]:
    """Unified constraint extraction node (Pass 1-5).

    Runs 5 independent Passes:
      Pass 1: cross-parameter (dtype/shape equality, length, divisibility)
      Pass 2: single-parameter (empty tensor, tensorlist, shape bound, bool, string length)
      Pass 3: implicit variable value (K1<65536, H=32/64)
      Pass 4: value range (deterministic regex + YAML + char* enum)
      Pass 5: LLM fallback (agent-based relation discovery + expr generation)

    Each Pass is wrapped in try/except - failure in one Pass does not affect
    others.  Results are deduplicated against existing param_relations and
    only new constraints are saved.

    Emits NODE_PROGRESS events for the frontend constraint detail panel.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ConstraintExtract: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        return {"error": None}

    # Setup NODE_PROGRESS emission
    ctx = get_context()
    _progress_span = Span(
        span_id="progress",
        parent_span_id=ctx.current_span_id if ctx else None,
        span_type=SpanType.NODE,
        name="constraint_extract",
    )

    def _emit(evt, data):
        if ctx and ctx.manager:
            ctx.manager.emit(evt, ctx.run_id, _progress_span, {
                "agent_id": "constraint",
                "node_id": "constraint_extract",
                **data,
            })

    try:
        # Step 1: Load data
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        existing = await _mcp_client.query_param_relations(doc_id)

        if not params:
            logger.info("ConstraintExtract: no parameters, skipping")
            return {"error": None}

        _emit(EventType.NODE_PROGRESS, {
            "message": f"已加载 {len(params)} 个参数, {len(existing)} 条已有约束",
            "phase": "data_ready",
            "params_count": len(params),
            "existing_count": len(existing),
        })

        all_new: list[dict] = []
        pass1_count = 0
        pass2_count = 0
        pass3_count = 0
        pass4_count = 0
        pass5_count = 0

        # --- Pass 1: Cross-parameter constraints ---
        try:
            pass1_results = _pass1_cross_param(params, existing)
            all_new.extend(pass1_results)
            pass1_count = len(pass1_results)
            logger.info(
                "ConstraintExtract Pass 1 (cross-param): %d new constraints",
                pass1_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 1 failed")

        # --- Pass 2: Single-parameter constraints ---
        # Dedup against existing + Pass 1 results
        try:
            pass2_results = _pass2_single_param(params, existing + all_new)
            all_new.extend(pass2_results)
            pass2_count = len(pass2_results)
            logger.info(
                "ConstraintExtract Pass 2 (single-param): %d new constraints",
                pass2_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 2 failed")

        # --- Pass 3: Implicit variable value constraints ---
        # Also fetches section content reused by Pass 4 and Pass 5
        ws_content = ""
        exe_content = ""
        constraints_text = ""
        try:
            ws_section = await _mcp_client.get_section(doc_id, "params_get_workspace")
            exe_section = await _mcp_client.get_section(doc_id, "params_execute")
            constraints_section = await _mcp_client.get_section(doc_id, "constraints")

            ws_content = ws_section.get("content", "") if ws_section else ""
            exe_content = exe_section.get("content", "") if exe_section else ""
            constraints_text = constraints_section.get("content", "") if constraints_section else ""
            if constraints_text:
                ws_content += "\n\n---\n## 约束说明\n" + constraints_text
            sections_text = f"{ws_content}\n\n{exe_content}"

            implicit_params = state.get("implicit_params", [])
            pass3_results = _pass3_implicit_value(
                sections_text, implicit_params, existing + all_new,
            )
            all_new.extend(pass3_results)
            pass3_count = len(pass3_results)
            logger.info(
                "ConstraintExtract Pass 3 (implicit-value): %d new constraints",
                pass3_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 3 failed")
            sections_text = ""

        # --- Pass 4: Value range (deterministic regex + YAML + char* enum) ---
        try:
            pass4_results = _pass4_value_range(
                params, existing + all_new, constraints_text,
            )
            all_new.extend(pass4_results)
            pass4_count = len(pass4_results)
            logger.info(
                "ConstraintExtract Pass 4 (value-range): %d new constraints",
                pass4_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 4 failed")

        # --- Pass 5: LLM fallback (agent-based relation discovery + expr) ---
        try:
            pass5_results = await _pass5_llm_fallback(
                state, params, existing + all_new,
                sections_text, ws_content, exe_content,
            )
            all_new.extend(pass5_results)
            pass5_count = len(pass5_results)
            logger.info(
                "ConstraintExtract Pass 5 (LLM fallback): %d new constraints",
                pass5_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 5 failed")

        _emit(EventType.NODE_PROGRESS, {
            "message": (
                f"P1:{pass1_count} P2:{pass2_count} P3:{pass3_count} "
                f"P4:{pass4_count} P5:{pass5_count} "
                f"合计新增 {len(all_new)} 条"
            ),
            "phase": "extract_done",
            "pass1_count": pass1_count,
            "pass2_count": pass2_count,
            "pass3_count": pass3_count,
            "pass4_count": pass4_count,
            "pass5_count": pass5_count,
            "new_count": len(all_new),
        })

        # Step 2: Merge and save
        if all_new:
            merged = existing + all_new
            # Expand platform="common" to per-platform rows before DB save
            platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)
            supported_platforms = [
                p["platform_name"] for p in platforms if p.get("is_supported") == 1
            ]
            merged = expand_common_in_relations(merged, supported_platforms)
            result = await _mcp_client.save_param_relations(doc_id, merged)
            logger.info(
                "ConstraintExtract: saved %d total relations "
                "(%d existing + %d new [P1=%d, P2=%d, P3=%d, P4=%d, P5=%d])",
                result.get("saved", 0),
                len(existing),
                len(all_new),
                pass1_count, pass2_count, pass3_count, pass4_count, pass5_count,
            )
        else:
            logger.info(
                "ConstraintExtract: no new constraints found for %s",
                operator_name,
            )

        return {
            "constraint_extract_results": all_new,
            "error": None,
        }

    except Exception:
        logger.exception("ConstraintExtract failed for %s", operator_name)
        return {"error": "constraint_extract_failed"}