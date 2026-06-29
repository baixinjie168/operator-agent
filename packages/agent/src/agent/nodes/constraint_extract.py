"""ConstraintExtract node: unified constraint extraction (Pass 1-3b).

Consolidates deterministic regex patterns from 4 existing nodes into a
single node with multiple Passes:

  Pass 1  - Cross-parameter constraints (dtype/shape equality, length, divisibility)
            Merges: cross_param_constraint + constraint_validation

  Pass 2  - Single-parameter constraints (empty tensor, tensorlist consistency,
            shape upper bound, bool restriction, string length)
            Merges: single_param_constraint Layer 1 + Layer 1b

  Pass 3  - Implicit variable value constraints (K1<65536, H=32/64, even)
            Merges: implicit_value_constraint

  Pass 3b - Implicit variable dimension equalities (K1==N2, N1==2*K2)
            Captures inter-variable equalities that Pass 1 (param-vs-param)
            and Pass 3 (var-vs-numeric) both miss.

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
from agent.core.config import settings
from agent.utils.expr_validation import _semantic_expr_key
from agent.utils.platform_utils import expand_common_in_relations
from agent.runtime.context import get_context
from agent.runtime.events import EventType, Span, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


# ---------------------------------------------------------------------------
# HTML cleaning — section content from HTML docs may contain residual tags
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags from text, preserving inner content.

    Section content fetched from the DB originates from HTML documents and
    may contain residual tags like </td>, </table>, <br> etc.  These pollute
    src_text / source_citation fields and break traceability.
    """
    if not text:
        return text
    return _HTML_TAG_RE.sub("", text).strip()


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
    # Infrastructure params — never real operator parameters
    "workspace", "workspaceSize", "executor", "stream",
})


def _is_param(name: str, all_names: set[str]) -> bool:
    """Check if *name* is a real parameter (not a generic word)."""
    if not name or len(name) < 2:
        return False
    if name.lower() in _EXCLUDE_NAMES:
        return False
    return name in all_names


def _normalize_expr(expr: str) -> str:
    """Normalize a constraint expression for dedup comparison.

    Handles commutative equivalence so that A==B and B==A are recognized
    as the same constraint.  Covers five common patterns:
      1. A.attr == B.attr   (e.g. x.dtype == y.dtype)
      2. len(A) == len(B)
      3. A.shape == B.shape
      4. A % B == 0         (divisibility, e.g. oriHeight % 16 == 0)
      5. A == N * B         (multiple, e.g. N1 == 2 * K2)

    For each equality pattern the two operands are sorted alphabetically so
    that the normalized form is deterministic regardless of source order.
    This is the string-based fallback; richer AST-level canonicalisation
    lives in ``agent.utils.expr_validation._semantic_expr_key``.
    """
    if not expr:
        return expr

    # 1. A.attr == B.attr  (general dotted equality)
    m = re.match(r"(\w+)\.(\w+)\s*==\s*(\w+)\.(\w+)\s*$", expr)
    if m:
        a = f"{m.group(1)}.{m.group(2)}"
        b = f"{m.group(3)}.{m.group(4)}"
        if a > b:
            a, b = b, a
        return f"{a} == {b}"

    # 2. len(A) == len(B)
    m = re.match(r"len\((\w+)\)\s*==\s*len\((\w+)\)\s*$", expr)
    if m:
        a, b = m.group(1), m.group(2)
        if a > b:
            a, b = b, a
        return f"len({a}) == len({b})"

    # 3. A.shape == B.shape
    m = re.match(r"(\w+)\.shape\s*==\s*(\w+)\.shape\s*$", expr)
    if m:
        a, b = m.group(1), m.group(2)
        if a > b:
            a, b = b, a
        return f"{a}.shape == {b}.shape"

    # 4. A % B == 0  (divisibility) — non-commutative, keep order
    m = re.match(r"(\w+)\s*%\s*(\w+)\s*==\s*0\s*$", expr)
    if m:
        return f"{m.group(1)} % {m.group(2)} == 0"

    # 5. A == N * B  (multiple equality) — non-commutative, keep order
    m = re.match(r"(\w+)\s*==\s*(\d+)\s*\*\s*(\w+)\s*$", expr)
    if m:
        return f"{m.group(1)} == {m.group(2)} * {m.group(3)}"

    return expr


def _expr_exists(existing: list[dict], expr: str) -> bool:
    """Check if an expr string already exists in param_relations.

    Uses _normalize_expr so that commutative equivalents (A==B vs B==A)
    are recognized as duplicates.
    """
    target = _normalize_expr(expr)
    for rel in existing:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(obj, dict):
            existing_expr = obj.get("expr", "")
            if existing_expr and _normalize_expr(existing_expr) == target:
                return True
    return False


# ---------------------------------------------------------------------------
# Phase 3 Item 7: cross-source / cross-expr_type semantic dedup
# ---------------------------------------------------------------------------
# expr_type specificity priority (lower = more specific = preferred survivor).
# When the same semantic key appears multiple times, the most specific
# expr_type wins (e.g. value_dependency beats self_string_length).
_EXPR_TYPE_PRIORITY: dict[str, int] = {
    "value_dependency": 1, "self_value_range": 1, "self_value_enum": 1,
    "type_dependency": 2, "type_equality": 2, "shape_value_dependency": 2,
    "presence_dependency": 2, "shape_equality": 2, "format_equality": 2,
    "self_string_length": 3, "cross_param_constraint": 3,
    "shape_broadcast": 3, "shape_dependency": 3, "self_shape_dim_range": 3,
    "self_constraint": 4,
}


def _deduplicate_relations(
    existing: list[dict], new_rels: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Platform-aware semantic dedup of ``existing + new_rels``.

    Dedup key = ``_semantic_expr_key(expr) + "::" + platform`` so that
    relations on different platforms are NEVER merged (compatible with
    ``expand_common_in_relations`` which expands ``platform="common"`` to
    per-platform rows — merging across platforms would lose coverage).

    Only ``new_rels`` duplicates are removed; ``existing`` (already in DB)
    is returned untouched to avoid accidental data loss. Within a group
    sharing the same key, the "best" (most specific) relation survives:
    priority = expr_type specificity → has real src_text → has guard →
    generic platform (``""``/``"common"``) preferred over platform-specific.
    """
    all_rels = existing + new_rels
    groups: dict[str, list[dict]] = {}
    for rel in all_rels:
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
        key = _semantic_expr_key(expr)
        if not key:
            continue
        plat = str(rel.get("platform", "") or "")
        groups.setdefault(f"{key}::{plat}", []).append(rel)

    removed: list[dict] = []
    for _gkey, group in groups.items():
        if len(group) <= 1:
            continue

        def _score(rel: dict) -> tuple:
            obj = rel.get("relation_object", {})
            if isinstance(obj, str):
                try:
                    obj = json.loads(obj)
                except Exception:  # noqa: BLE001
                    obj = {}
            et = obj.get("expr_type", "")
            src = obj.get("src_text", "") or rel.get("source_citation", "")
            expr_s = obj.get("expr", "")
            plat = str(rel.get("platform", "") or "")
            return (
                _EXPR_TYPE_PRIORITY.get(et, 5),
                0 if src and "正则" not in src and "YAML" not in src else 1,
                0 if "if" in expr_s else 1,
                0 if plat in ("", "common") else 1,  # generic platform preferred
            )

        best = min(group, key=_score)
        for rel in group:
            if rel is best:
                continue
            # Use identity membership to correctly handle the case where
            # multiple new_rels entries are value-equal dicts (e.g. the same
            # expr repeated 3 times). Value-based ``rel in new_rels`` would
            # match ALL equal dicts, and ``r not in removed`` would then
            # drop the survivor too — identity avoids that.
            if any(rel is r for r in new_rels):
                removed.append(rel)
                logger.info(
                    "ConstraintDedup: 移除语义重复 expr=%s platform=%s",
                    str(rel.get("relation_object", ""))[:60],
                    rel.get("platform", ""),
                )
    removed_ids = {id(r) for r in removed}
    deduped_new = [r for r in new_rels if id(r) not in removed_ids]
    return existing, deduped_new


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
    platform: str = "",
) -> dict:
    """Build a single-parameter relation dict."""
    return {
        "function_name": fn,
        "relation_type": "self_constraint",
        "platform": platform,
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
                # context_before check: verify pn is the subject (appears
                # before the "与/和/同" keyword).  Without this, patterns
                # like "deqScale2的数据类型与offsetOptional一致" in
                # offsetOptional's description would incorrectly produce
                # offsetOptional.dtype == deqScale2.dtype, when actually
                # offsetOptional's dtype is fixed and the constraint is
                # about deqScale2 matching it.
                context_before = text[max(0, m.start() - 60):m.start()]
                if pn not in context_before:
                    continue  # pn is not the subject, skip
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


def _collect_seen_exprs(existing: list[dict]) -> set[str]:
    """Collect expr strings from existing relations for dedup.

    Extracted from the inline dedup logic of ``_pass3_implicit_value`` so
    that Pass 3 and the new Pass 3b (dimension equalities) share one
    canonical dedup source. Handles ``relation_object`` stored as either a
    dict or a JSON string.
    """
    seen: set[str] = set()
    for rel in existing:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(obj, dict):
            seen.add(obj.get("expr", ""))
    return seen


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
    seen_exprs = _collect_seen_exprs(existing)

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
            src_text = _strip_html(sections_text[start:end].strip())

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
# Pass 3b: Implicit variable dimension equalities (deterministic regex)
# Captures inter-variable equalities like K1==N2, N1==2*K2 that Pass 1
# (param-vs-param) and Pass 3 (var-vs-numeric) both miss.
# ===================================================================

# Order-sensitive: N*Y must be matched before X=Y so that "N1=2*K2" is not
# truncated to "N1=2" by the equality pattern.
_P3B_DIM_NMUL_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)"
)
_P3B_DIM_EQ_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)"
)
# Chinese equality: "X等于Y" / "X与Y相等/一致"
_P3B_DIM_EQ_CN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:等于|与)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:相等|一致)"
)
# Chinese multiple: "X是Y的N倍"
_P3B_DIM_NMUL_CN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*是\s*([A-Za-z_][A-Za-z0-9_]*)\s*的\s*(\d+)\s*(?:倍|整数倍)"
)
# Condition clause detection: "激活层为X时" / "当X时" / "若X时"
_P3B_COND_RE = re.compile(
    r"(?:激活层为|当|若)\s*([^\s，。；,;]{1,40}?)\s*时"
)


def _detect_condition_clause(text: str, pos: int) -> str:
    """Look back up to 80 chars before *pos* for a ``...时`` condition.

    Returns a Python condition fragment (e.g.
    ``activation in [geglu, swiglu, reglu]``) or an empty string when no
    encodable condition is found. Only ``激活层为X时`` clauses are encoded
    into a Python ``in`` test today; other ``当X时``/``若X时`` clauses are
    detected but not yet encoded (returns "" so the equality is emitted
    unconditionally — conservative, never drops the constraint).
    """
    window = text[max(0, pos - 80):pos]
    last = None
    for m in _P3B_COND_RE.finditer(window):
        last = m  # take the condition nearest to the equality
    if not last:
        return ""
    parts = [
        p.strip() for p in re.split(r"[/、，,]", last.group(1)) if p.strip()
    ]
    if parts and "激活" in window:
        return f"activation in [{', '.join(parts)}]"
    return ""


def _pass3b_dim_equalities(
    sections_text: str,
    implicit_params: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Pass 3b: Extract equality constraints between implicit dimension vars.

    Captures inter-variable equalities such as ``K1==N2`` and
    ``N1==2*K2`` that Pass 1 (param-vs-param, scans param descriptions) and
    Pass 3 (var-vs-numeric, requires a numeric RHS) both miss. These appear
    in the constraint *section* text where both sides are implicit
    dimension variables, so this pass reuses the Pass 3 context
    (``sections_text`` + ``var_names``) and runs alongside it.

    Only generates an equality when **both** sides are known implicit
    variables, avoiding ``a=b`` descriptive prose. Conditional equalities
    (``激活层为geglu时，N1=2*K2``) are wrapped in a guard:
    ``((N1 == 2 * K2) if activation in [geglu] else True)``.

    Uses ``expr_type="cross_variable_equality"`` (descriptive metadata;
    downstream consumers only branch on ``value_dependency``, so this is
    safe — see assemble_result._group_constraints_by_platform).
    """
    if not sections_text.strip() or not implicit_params:
        return []
    var_names = {
        m["var_name"] for m in implicit_params
        if m.get("var_name")
        and not m.get("is_constant")
        and not m.get("is_external_constant")
        and not m.get("is_quantization_type")
    }
    if not var_names:
        return []

    seen_exprs = _collect_seen_exprs(existing)
    constraints: list[dict] = []

    def _emit(expr: str, params: list[str], src: str, m: re.Match) -> None:
        cond = _detect_condition_clause(sections_text, m.start())
        full = f"({expr}) if {cond} else True" if cond else expr
        if full in seen_exprs:
            return
        seen_exprs.add(full)
        s = max(0, m.start() - 50)
        e = min(len(sections_text), m.end() + 50)
        src_text = _strip_html(sections_text[s:e].strip())
        constraints.append({
            "function_name": "",
            "relation_type": "cross_param_constraint",
            "platform": "",
            "description": full,
            "params": params,
            "param_optional": {p: False for p in params},
            "source_citation": src_text,
            "relation_object": {
                "expr_type": "cross_variable_equality",
                "expr": full,
                "relation_params": params,
                "src_text": src_text,
            },
        })

    # 1) X = N * Y  (matched first so X=Y does not truncate "N1=2*K2")
    for m in _P3B_DIM_NMUL_RE.finditer(sections_text):
        x, n, y = m.group(1), m.group(2), m.group(3)
        if x in var_names and y in var_names:
            _emit(f"{x} == {n} * {y}", [x, y], m.group(0), m)

    # 2) X = Y  (both sides implicit vars; skip X=X and numeric RHS)
    for m in _P3B_DIM_EQ_RE.finditer(sections_text):
        x, y = m.group(1), m.group(2)
        if x == y or y.isdigit():
            continue
        if x in var_names and y in var_names:
            _emit(f"{x} == {y}", [x, y], m.group(0), m)

    # 3) Chinese equality / multiple (same var-membership check)
    for m in _P3B_DIM_EQ_CN_RE.finditer(sections_text):
        x, y = m.group(1), m.group(2)
        if x in var_names and y in var_names:
            _emit(f"{x} == {y}", [x, y], m.group(0), m)
    for m in _P3B_DIM_NMUL_CN_RE.finditer(sections_text):
        x, y, n = m.group(1), m.group(2), m.group(3)
        if x in var_names and y in var_names:
            _emit(f"{x} == {n} * {y}", [x, y], m.group(0), m)

    return constraints


# ===================================================================
# Pass 4: Value range extraction (deterministic regex + YAML + char* enum)
# Merges: allowed_range_build Phase 1 + Phase 1b + single_param_constraint Layer 0
# ===================================================================

# Reuse patterns from allowed_range_build (lazy import to avoid circular deps)


def _try_deterministic_range(text: str, return_match: bool = False):
    """Try deterministic extraction of range from text.

    Returns ``(value_list, ar_type)`` or, when *return_match* is True,
    ``(value_list, ar_type, match)`` where *match* is the ``re.Match``
    object — letting the caller extract document-original context for
    ``src_text`` traceability (Item 6).

    *return_match* defaults to ``False`` for backward compatibility.
    Note: ``allowed_range_build.py`` has its own independent
    ``_try_deterministic_range``; this change does not affect it.
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
                        return (value_list, ar_type, m) if return_match \
                               else (value_list, ar_type)
                elif isinstance(result, list) and result:
                    return (result, "range", m) if return_match \
                           else (result, "range")
            except (ValueError, IndexError):
                continue
    return None


def _extract_src_context(text: str, match: re.Match | None, radius: int = 50) -> str:
    """Extract +-radius chars of document-original context around a match.

    Reuses :func:`_strip_html` to remove residual HTML tags so ``src_text``
    stays clean and traceable to the source document (Item 6).
    """
    if not text or not match:
        return ""
    s = max(0, match.start() - radius)
    e = min(len(text), match.end() + radius)
    return _strip_html(text[s:e].strip())


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


async def _pass4_value_range(
    params: list[dict],
    existing: list[dict],
    constraints_text: str,
    sections_text: str,
) -> list[dict]:
    """Pass 4: Extract value range constraints for scalar parameters.

    Merges:
    - allowed_range_build Phase 1 (deterministic regex)
    - allowed_range_build Phase 1b (YAML semantic rules for scalars)
    - single_param_constraint Layer 0 (YAML semantic rules for tensors)
    - char* enum extraction via DeepAgent (replaces deleted allowed_range_extract node 4h)
    """
    from agent.nodes.build_param_constraint._helpers import _normalize_type, _split_csv
    from agent.utils.param_validators import is_bool_type, is_tensor_type
    from agent.utils.semantic_rules import (
        get_allowed_range_for_scalar,
        get_expr_for_tensor,
    )

    new_rels: list[dict] = []
    char_params: list[dict] = []  # collected for batch Agent processing

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

        # --- char* params: collect for batch DeepAgent extraction ---
        if normalized in ("char", "const char"):
            char_params.append(param)
            continue

        # --- Skip aclIntArray (LLM handles) ---
        if "aclIntArray" in ptype:
            continue

        # --- Phase 1: Deterministic regex ---
        # return_match=True so we can extract document-original context (Item 6).
        det = _try_deterministic_range(combined_text, return_match=True)
        det_src_text = combined_text
        if det is None:
            param_sentences = _extract_param_sentences(constraints_text, pn)
            if param_sentences:
                det = _try_deterministic_range(param_sentences, return_match=True)
                det_src_text = param_sentences

        if det is not None:
            det_value, det_type, det_match = det
            expr = _range_to_expr(pn, det_value, det_type)
            if expr:
                # Item 6: use document-original context instead of placeholder.
                src_text = _extract_src_context(det_src_text, det_match) \
                           or f"正则提取: {det_type}"
                new_rels.append(_make_self_rel(
                    fn, pn, expr,
                    f"{pn} 取值范围({det_type}): {det_value}",
                    src_text,
                    "self_value_enum" if det_type == "enum" else "self_value_range",
                ))
                continue

        # --- Phase 1b: YAML semantic rules fallback for scalar params ---
        # Only search the parameter's OWN description text, NOT the entire
        # constraints_text.  Searching constraints_text causes false-positive
        # matches: e.g. innerPrecise's YAML search picks up "scale" from
        # scaleOptional/deqScale mentions in the shared constraints section,
        # triggering the positive_scale rule (>= 1) which is wrong for a
        # mode parameter that can be 0 or 1.
        yaml_ar = None
        is_tensor = "aclTensor" in ptype
        if not is_tensor:
            yaml_ar = get_allowed_range_for_scalar(combined_text, pn)
        if yaml_ar:
            expr = _range_to_expr(pn, yaml_ar, "range")
            if expr:
                # Item 6: use sentences mentioning param_name as src_text.
                yaml_src = _extract_param_sentences(combined_text, pn)
                yaml_src = _strip_html(yaml_src[:200]) if yaml_src else ""
                new_rels.append(_make_self_rel(
                    fn, pn, expr,
                    f"{pn} 取值范围(YAML): {yaml_ar}",
                    yaml_src or f"YAML语义规则: {pn}",
                    "self_value_range",
                ))
                continue

        # --- Layer 0: YAML for Tensor element-level ---
        if "aclTensor" in ptype:
            result = get_expr_for_tensor(combined_text, pn)
            if result and result.get("confidence") == "high":
                expr = result.get("expr", "")
                if expr:
                    desc = result.get("description", "")
                    # Item 6: use sentences mentioning param_name as src_text.
                    tensor_src = _extract_param_sentences(combined_text, pn)
                    tensor_src = _strip_html(tensor_src[:200]) if tensor_src else ""
                    new_rels.append(_make_self_rel(
                        fn, pn, expr, desc,
                        tensor_src or desc or f"YAML Tensor规则: {pn}",
                        result.get("expr_type", "self_value_range"),
                    ))

    # --- char* enum extraction via DeepAgent ---
    if char_params and sections_text.strip():
        try:
            from agent.nodes.allowed_range_agent import _extract_batch_via_agent

            # Item 6: helper to find document-original context for enum values.
            def _find_char_src(vals: list[str]) -> str:
                for v in vals[:3]:
                    idx = sections_text.find(v)
                    if idx >= 0:
                        s = max(0, idx - 30)
                        e = min(len(sections_text), idx + len(v) + 30)
                        return _strip_html(sections_text[s:e].strip())
                return ""

            batch_result = await _extract_batch_via_agent(char_params, sections_text)
            for param in char_params:
                pn = param.get("param_name", "")
                fn = param.get("function_name", "")
                entries = batch_result.get(pn, [])
                if not entries:
                    continue

                # Group entries by platform to preserve per-platform enum values.
                # Each entry: {"platform": "...", "allowed_range_value": "...", "type": "enum"}
                # If all entries share the same (or empty) platform, merge into one.
                # If entries have different platforms, create one relation per platform.
                platforms = {
                    e.get("platform", "") for e in entries
                    if isinstance(e, dict) and e.get("allowed_range_value", "")
                }
                if len(platforms) <= 1:
                    # Single platform (or all empty) — merge all values, dedup
                    enum_vals: list[str] = []
                    for entry in entries:
                        if isinstance(entry, dict):
                            raw = entry.get("allowed_range_value", "")
                            if raw:
                                enum_vals.extend(_split_csv(raw))
                    enum_vals = sorted(set(enum_vals))
                    if enum_vals:
                        expr = f"{pn}.range_value in [{', '.join(repr(v) for v in enum_vals)}]"
                        # Item 6: use document context instead of placeholder.
                        char_src = _find_char_src(enum_vals)
                        new_rels.append(_make_self_rel(
                            fn, pn, expr,
                            f"{pn} 取值范围: {', '.join(enum_vals)}",
                            char_src or f"char*枚举值: {', '.join(enum_vals)}",
                            "self_value_enum",
                        ))
                else:
                    # Multiple platforms — create one relation per platform
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        raw = entry.get("allowed_range_value", "")
                        platform = entry.get("platform", "")
                        if not raw:
                            continue
                        vals = _split_csv(raw)  # already deduped and sorted
                        if not vals:
                            continue
                        expr = f"{pn}.range_value in [{', '.join(repr(v) for v in vals)}]"
                        # Item 6: use document context instead of placeholder.
                        char_src = _find_char_src(vals)
                        new_rels.append(_make_self_rel(
                            fn, pn, expr,
                            f"{pn} 取值范围: {', '.join(vals)}",
                            char_src or f"char*枚举值: {', '.join(vals)}",
                            "self_value_enum",
                            platform=platform,
                        ))
        except Exception:
            logger.exception("ConstraintExtract Pass 4: char* enum Agent failed")

    return new_rels


# ===================================================================
# Pass 4b: Per-platform scalar enum extraction
# Extracts per-platform allowed values for int64_t/integer params whose
# usage_notes contain platform-specific restrictions (e.g. "只支持传1").
# ===================================================================

# "只支持传1" / "只能为0" / "仅支持传1" / "只能配置为0" → [value]
_SCALAR_RESTRICTED_RE = re.compile(
    r"(?:只支持|仅支持|只能|仅能)\s*(?:传|为|配置为|配为|置为|是)?\s*(\d+)"
)

# "可以配置为0或者1" / "可配置为0或1" → [v1, v2]
_SCALAR_MULTI_CONFIG_RE = re.compile(
    r"(?:可以|可)\s*(?:配置|配)?\s*(?:为|置为)?\s*(\d+)\s*(?:或者|或|和)\s*(\d+)"
)

# "0和1都可配置" / "0或1均可配置" → [v1, v2]
_SCALAR_ALL_CONFIG_RE = re.compile(
    r"(\d+)\s*(?:和|或|或者)\s*(\d+)\s*(?:都可|都可以|均可)\s*(?:配置|设|设置)?"
)

# "innerPrecise为0时" / "innerPrecise为1代表" → collect mentioned values
_SCALAR_VALUE_MENTION_RE = re.compile(
    r"(?:为|传|配置为|配为)\s*(\d+)\s*(?:时|代表|，|。|的)"
)


def _extract_scalar_enum_values(text: str, pname: str = "") -> list[int]:
    """Extract discrete integer enum values from text.

    Scans for patterns like "只支持传1", "可以配置为0或者1",
    "0和1都可配置", and "X为0时 / X为1时".

    Returns a sorted list of unique integers, or [] if no pattern matches.
    """
    if not text:
        return []
    values: set[int] = set()

    # "只支持传N" / "只能为N" → [N]
    for m in _SCALAR_RESTRICTED_RE.finditer(text):
        # Check context: only accept if the param name is nearby
        # or the value is 0/1 (mode-type values)
        val = int(m.group(1))
        start = max(0, m.start() - 60)
        context = text[start:m.end() + 20]
        if pname and pname in context:
            values.add(val)
        elif val in (0, 1):
            # Be conservative: only accept 0/1 without param name context
            # if the surrounding text looks like a mode/flag description
            if any(kw in context for kw in ("模式", "模式", "高性能", "高精度", "传")):
                values.add(val)

    # "可以配置为N或者M" → [N, M]
    for m in _SCALAR_MULTI_CONFIG_RE.finditer(text):
        values.add(int(m.group(1)))
        values.add(int(m.group(2)))

    # "N和M都可配置" → [N, M]
    for m in _SCALAR_ALL_CONFIG_RE.finditer(text):
        values.add(int(m.group(1)))
        values.add(int(m.group(2)))

    # "X为N时" / "X为N代表" → collect N (only if pname is mentioned nearby)
    if pname:
        for m in _SCALAR_VALUE_MENTION_RE.finditer(text):
            start = max(0, m.start() - 40)
            context = text[start:m.end() + 10]
            if pname in context:
                values.add(int(m.group(1)))

    return sorted(values)


def _pass4b_per_platform_scalar_enum(
    params: list[dict],
    existing: list[dict],
    supported_platforms: list[str],
    constraints_text: str,
) -> list[dict]:
    """Extract per-platform enum values for integer scalar params.

    For params like innerPrecise where different platforms support different
    values (e.g. Atlas 推理系列加速卡: only 1, Atlas A2: 0 or 1), this
    function parses per-platform usage_notes and the param description to
    generate per-platform self_value_enum relations.

    This runs after Pass 4 and supplements it for params that Pass 4
    couldn't handle (no deterministic regex match, no YAML match).
    """
    from agent.nodes.build_param_constraint._helpers import (
        _normalize_type,
        _parse_json_field,
    )
    from agent.utils.table_parser import resolve_platform_value
    from agent.utils.param_validators import is_bool_type, is_tensor_type

    new_rels: list[dict] = []

    for param in params:
        pn = param.get("param_name", "")
        fn = param.get("function_name", "")
        ptype = param.get("param_type", "")
        if not pn or not fn:
            continue

        # Only handle integer scalar types (int64_t, int, etc.)
        if is_bool_type(ptype) or is_tensor_type(ptype):
            continue
        normalized = _normalize_type(ptype)
        if normalized not in ("int64_t", "int", "int32_t", "uint64_t", "int8_t"):
            continue

        # Skip if already has a self_value_enum constraint
        already_has = False
        for r in existing + new_rels:
            if pn in r.get("params", []):
                obj = r.get("relation_object", {})
                if isinstance(obj, dict) and obj.get("expr_type") == "self_value_enum":
                    already_has = True
                    break
        if already_has:
            continue

        # Parse usage_notes as per-platform JSON
        usage_json = _parse_json_field(param.get("usage_notes", ""))

        # Collect per-platform enum values
        platform_values: dict[str, list[int]] = {}
        for plat in supported_platforms:
            usage_text = resolve_platform_value(usage_json, plat)
            if not usage_text:
                continue
            vals = _extract_scalar_enum_values(usage_text, pn)
            if vals:
                platform_values[plat] = vals

        # For platforms without specific values, try param_desc + constraints
        general_values: list[int] = []
        if not all(plat in platform_values for plat in supported_platforms):
            param_desc = param.get("param_desc", "") or ""
            llm_desc = param.get("llm_description", "") or ""
            combined = f"{param_desc}\n{llm_desc}"
            general_values = _extract_scalar_enum_values(combined, pn)
            if not general_values and constraints_text:
                # Extract sentences mentioning this param from constraints text
                from agent.nodes.build_param_constraint.allowed_range_build import (
                    _extract_param_sentences,
                )
                param_sentences = _extract_param_sentences(constraints_text, pn)
                general_values = _extract_scalar_enum_values(param_sentences, pn)

        if not platform_values and not general_values:
            continue

        # Create per-platform relations
        for plat in supported_platforms:
            vals = platform_values.get(plat, general_values)
            if not vals:
                continue
            expr = f"{pn}.range_value in [{', '.join(str(v) for v in vals)}]"
            src = f"标量枚举提取({plat}): {vals}"
            new_rels.append(_make_self_rel(
                fn, pn, expr,
                f"{pn} 取值范围: {vals}",
                src, "self_value_enum",
                platform=plat,
            ))

        logger.info(
            "Pass4b: extracted per-platform enum for %s: %s",
            pn,
            {p: platform_values.get(p, general_values) for p in supported_platforms
             if platform_values.get(p, general_values)},
        )

    return new_rels


# ===================================================================
# Pass 6: Conditional constraint extraction from constraints text
# Extracts multi-param conditional constraints like:
#   "innerPrecise参数在BFLOAT16非量化场景，只能配置为0；..."
# These are too complex for regex expr generation — the relation is
# created with empty expr and routed to the complex relation Agent
# (via complexity_classify) in build_param_relations.
# ===================================================================

# "X参数在...场景" — indicates a conditional constraint
_PARAM_CONDITIONAL_RE = re.compile(
    r"(\w+)\s*参数\s*在\s*.+?场景"
)


def _pass6_conditional_constraints(
    params: list[dict],
    existing: list[dict],
    constraints_text: str,
    all_param_names: set[str],
) -> list[dict]:
    """Extract conditional constraints from the constraints section text.

    Scans for patterns like "X参数在...场景，只能配置为..." and creates
    multi-param value_dependency relations with empty expr.  The expr will
    be generated by the complex relation Agent in build_param_relations.

    The relation_params include the constrained param plus all other param
    names mentioned in the constraint sentence, so the Agent has enough
    context to generate the conditional expression.
    """
    if not constraints_text.strip():
        return []

    new_rels: list[dict] = []

    # Split constraints text into sentences
    sentences = re.split(r"[。；;\n]", constraints_text)

    for sentence in sentences:
        sentence = _strip_html(sentence.strip())
        if not sentence:
            continue

        # Check for conditional pattern: "X参数在...场景"
        m = _PARAM_CONDITIONAL_RE.search(sentence)
        if not m:
            continue

        constrained_param = m.group(1)
        if constrained_param not in all_param_names:
            continue

        # Check if this constraint already exists (by source_citation)
        already_exists = any(
            r.get("source_citation", "") == sentence
            for r in existing + new_rels
        )
        if already_exists:
            continue

        # Find all param names mentioned in the sentence
        mentioned_params: list[str] = [constrained_param]
        for name in all_param_names:
            if name != constrained_param and name in sentence:
                mentioned_params.append(name)

        # If only the constrained param is mentioned, the conditional
        # likely references dtypes/presence of other params indirectly
        # (e.g. "BFLOAT16非量化场景" implies checking x.dtype and
        # scaleOptional presence).  Include all non-infrastructure params
        # from the same function as relation_params so the Agent can
        # reference them.
        if len(mentioned_params) < 3:
            # Add quantization-related params that might be referenced
            # implicitly by "量化场景" / "非量化场景" / "伪量化场景"
            quant_params = [
                p.get("param_name", "") for p in params
                if p.get("param_name", "") not in mentioned_params
                and any(kw in (p.get("param_name", "") + p.get("param_desc", "") + p.get("llm_description", ""))
                        for kw in ("scale", "Scale", "antiquant", "Antiquant", "deqScale", "DeqScale", "offset", "Offset"))
            ]
            mentioned_params.extend(quant_params)

        # Also include x, weight1, weight2, y (dtype-bearing mandatory params)
        # since "BFLOAT16场景" / "FLOAT16场景" references their dtypes
        for must_name in ("x", "weight1", "weight2", "y"):
            if must_name in all_param_names and must_name not in mentioned_params:
                mentioned_params.append(must_name)

        # Build the relation
        fn = ""
        for p in params:
            if p.get("param_name") == constrained_param:
                fn = p.get("function_name", "")
                break

        new_rels.append({
            "function_name": fn,
            "relation_type": "value_dependency",
            "platform": "",
            "description": sentence,
            "params": mentioned_params,
            "param_optional": {p: False for p in mentioned_params},
            "source_citation": sentence,
            "relation_object": {
                "expr_type": "value_dependency",
                "expr": "",  # to be generated by complex relation Agent
                "relation_params": mentioned_params,
                "src_text": sentence,
            },
        })

        logger.info(
            "Pass6: detected conditional constraint for %s: %s...",
            constrained_param,
            sentence[:60],
        )

    return new_rels


# ===================================================================
# Pass 6b: Conditional shape extraction from parameter descriptions
# Scans param_desc / llm_description for conditional shape patterns like
#   "[E,K1,N1]/[K1,N1]" or "有专家...无专家..." or "per-channel/per-tensor"
# that are NOT in the constraints section but in the parameter table.
# Creates shape_value_dependency relations with empty expr for the complex
# relation Agent to generate in build_param_relations.
# ===================================================================

# Multi-shape candidate: "[A,B]/[C]" or "[A,B]或[C]" or "[A,B]，[C]"
_MULTI_SHAPE_RE = re.compile(
    r"\[[A-Za-z0-9_,\s]+\]\s*(?:[/、，,]|或)\s*\[[A-Za-z0-9_,\s]+\]"
)
# Expert-conditional: "有专家" / "无专家"
_EXPERT_COND_RE = re.compile(r"有专家|无专家")
# Quantization-conditional: "per-channel" / "per-tensor" / "per-group"
_QUANT_COND_RE = re.compile(r"per-channel|per-tensor|per-group", re.IGNORECASE)


def _pass6b_conditional_shapes_from_params(
    params: list[dict],
    existing: list[dict],
    all_param_names: set[str],
) -> list[dict]:
    """Extract conditional shape constraints from parameter descriptions.

    Scans each parameter's param_desc and llm_description for conditional
    shape patterns (multi-shape candidates, expert-conditional, quantization-
    conditional).  These patterns indicate the parameter's shape depends on
    other parameters' values, which is too complex for regex expr generation.

    Creates shape_value_dependency relations with empty expr, to be routed
    to the complex_relation_agent via complexity_classify.
    """
    new_rels: list[dict] = []

    for param in params:
        pn = param.get("param_name", "")
        fn = param.get("function_name", "")
        if not pn:
            continue

        text = _strip_html(_collect_param_text(param))
        if not text.strip():
            continue

        # Check for any conditional shape pattern
        has_multi_shape = bool(_MULTI_SHAPE_RE.search(text))
        has_expert_cond = bool(_EXPERT_COND_RE.search(text))
        has_quant_cond = bool(_QUANT_COND_RE.search(text))

        if not (has_multi_shape or has_expert_cond or has_quant_cond):
            continue

        # Check if a shape constraint for this param already exists
        already_has = False
        for r in existing + new_rels:
            if pn in r.get("params", []):
                obj = r.get("relation_object", {})
                if isinstance(obj, dict):
                    et = obj.get("expr_type", "")
                    if et in ("shape_value_dependency", "shape_choice"):
                        already_has = True
                        break
        if already_has:
            continue

        # Find other params mentioned in the text for relation_params
        mentioned_params: list[str] = [pn]
        for name in all_param_names:
            if name != pn and name in text:
                mentioned_params.append(name)

        # Add quantization-related params if quantization-conditional
        if has_quant_cond:
            quant_params = [
                p.get("param_name", "") for p in params
                if p.get("param_name", "") not in mentioned_params
                and any(kw in (p.get("param_name", "") + p.get("param_desc", ""))
                        for kw in ("scale", "Scale", "antiquant", "Antiquant",
                                   "deqScale", "DeqScale", "offset", "Offset"))
            ]
            mentioned_params.extend(quant_params)

        # Add expert-related params if expert-conditional
        if has_expert_cond:
            for name in all_param_names:
                if "expert" in name.lower() and name not in mentioned_params:
                    mentioned_params.append(name)

        # Also include dtype-bearing mandatory params
        for must_name in ("x", "weight1", "weight2", "y"):
            if must_name in all_param_names and must_name not in mentioned_params:
                mentioned_params.append(must_name)

        # Extract source citation (the sentence containing the pattern)
        src_text = ""
        for pattern in (_MULTI_SHAPE_RE, _EXPERT_COND_RE, _QUANT_COND_RE):
            m = pattern.search(text)
            if m:
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                src_text = text[start:end].strip()
                break

        if not src_text:
            continue

        new_rels.append({
            "function_name": fn,
            "relation_type": "shape_value_dependency",
            "platform": "",
            "description": f"{pn} 的 shape 依赖条件（来自参数描述）",
            "params": mentioned_params,
            "param_optional": {p: False for p in mentioned_params},
            "source_citation": src_text,
            "relation_object": {
                "expr_type": "shape_value_dependency",
                "expr": "",  # to be generated by complex relation Agent
                "relation_params": mentioned_params,
                "src_text": src_text,
            },
        })

        logger.info(
            "Pass6b: detected conditional shape for %s: %s...",
            pn,
            src_text[:60],
        )

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
# Pass 7: Dtype constraints from dtype_combinations
# Generates type_equality / type_dependency relations from the
# dtype_combos table (populated by dtype_combo_extract_node).
# Zero LLM calls — pure deterministic analysis of combo data.
# ===================================================================


async def _pass7_dtype_constraints(
    params: list[dict],
    existing: list[dict],
    dtype_combos: list[dict],
) -> list[dict]:
    """Pass 7: Generate dtype constraints from dtype_combinations.

    Analyzes the dtype_combos table to extract:
      1. Fixed dtype: a param always has the same dtype across all combos
         -> type_equality (param.dtype == 'DTYPE')
      2. Same-dtype pairs: two params always share the same dtype
         -> type_equality (paramA.dtype == paramB.dtype)

    Notes:
    - Mutual-exclusion and derivation (e.g. promote) are too complex for
      deterministic extraction and are left to the LLM passes.
    - Only generates constraints for params that appear in the combos.
    """
    if not dtype_combos:
        return []

    # Build per-function combo lists: {fn: [combo_dict, ...]}
    fn_combos: dict[str, list[dict]] = {}
    for dc in dtype_combos:
        fn = dc.get("function_name", "")
        combo = dc.get("combo", {})
        if fn and combo:
            fn_combos.setdefault(fn, []).append(combo)

    if not fn_combos:
        return []

    new_rels: list[dict] = []

    for fn, combos in fn_combos.items():
        if len(combos) < 1:
            continue

        # Collect all param names that appear in combos
        combo_params: set[str] = set()
        for combo in combos:
            combo_params.update(combo.keys())

        # --- 1. Fixed dtype detection ---
        # If a param has the same dtype value in ALL combos, it's fixed.
        param_fixed_dtype: dict[str, str] = {}
        for pn in combo_params:
            dtypes_seen: set[str] = set()
            for combo in combos:
                dt = combo.get(pn)
                if dt:
                    dtypes_seen.add(str(dt).strip())
            if len(dtypes_seen) == 1:
                param_fixed_dtype[pn] = dtypes_seen.pop()

        for pn, dt in param_fixed_dtype.items():
            expr = f"{pn}.dtype == '{dt}'"
            if _expr_exists(existing + new_rels, expr):
                continue
            # Check if an existing dtype constraint already covers this param
            already_has = False
            for r in existing + new_rels:
                if pn in r.get("params", []):
                    obj = r.get("relation_object", {})
                    if isinstance(obj, dict) and obj.get("expr_type", "") in (
                        "type_equality", "type_dependency"
                    ) and obj.get("expr", ""):
                        already_has = True
                        break
            if already_has:
                continue

            new_rels.append(_make_self_rel(
                fn, pn, expr,
                f"{pn} 的数据类型固定为 {dt}",
                f"dtype_combos分析: {pn}在所有组合中均为{dt}",
                "type_equality",
            ))

        # --- 2. Same-dtype pair detection ---
        # If two params always share the same dtype across all combos
        # where both are present, they have a type_equality constraint.
        param_list = sorted(combo_params)
        for i, pa in enumerate(param_list):
            for pb in param_list[i + 1:]:
                # Skip if either has a fixed dtype (already covered)
                if pa in param_fixed_dtype or pb in param_fixed_dtype:
                    continue
                # Check if they always match
                always_match = True
                both_present = False
                for combo in combos:
                    dta = combo.get(pa)
                    dtb = combo.get(pb)
                    if dta and dtb:
                        both_present = True
                        if str(dta).strip() != str(dtb).strip():
                            always_match = False
                            break
                if both_present and always_match:
                    expr = f"{pa}.dtype == {pb}.dtype"
                    if _expr_exists(existing + new_rels, expr):
                        continue
                    new_rels.append(_make_cross_rel(
                        fn, "dtype", pa, pb, expr,
                        f"{pa} 与 {pb} 的数据类型一致",
                        f"dtype_combos分析: {pa}与{pb}在所有组合中数据类型相同",
                    ))

    return new_rels


# ===================================================================
# Pass 7b: Conditional dtype constraints (type_dependency)
# Generates conditional dtype constraints from structured dtype JSON
# (e.g. {"量化": "INT8", "*": "FLOAT16"}) written by dtype_extract_node.
# ===================================================================

def _pass7b_conditional_dtype(
    params: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Pass 7b: Generate ``type_dependency`` constraints from conditional dtype.

    Reads each parameter's dtype field (the MCP server
    ``query_params_by_doc_id`` returns it as ``data_type``, mapped from the
    DB ``dtype_desc`` column; ``dtype_desc`` is also accepted for the
    agent's own ``db.py`` fallback). When the value is a structured JSON
    dict with a ``"*"`` default branch plus one or more condition branches
    (e.g. ``{"量化": "INT8", "*": "FLOAT16"}``), emit a conditional dtype
    constraint::

        (param.dtype == 'INT8') if quantization else (param.dtype == 'FLOAT16')

    Silently skips params with empty / non-JSON / non-conditional dtype so
    the pipeline never breaks.
    """
    new_rels: list[dict] = []
    for param in params:
        pn = param.get("param_name", "")
        fn = param.get("function_name", "")
        # MCP server returns "data_type"; agent db.py returns "dtype_desc".
        dtype_desc = (
            param.get("data_type", "")
            or param.get("dtype_desc", "")
            or ""
        )
        if not pn or not dtype_desc:
            continue

        try:
            dt_map = (
                json.loads(dtype_desc) if dtype_desc.startswith("{") else None
            )
        except (json.JSONDecodeError, TypeError):
            dt_map = None
        if not isinstance(dt_map, dict) or "*" not in dt_map:
            continue  # not a conditional dtype

        cond_branches = {k: v for k, v in dt_map.items() if k != "*"}
        if not cond_branches:
            continue

        default_dtype = dt_map["*"]
        # Skip if an existing type_equality / type_dependency already covers
        # this param — Pass 7 may have emitted a fixed-dtype constraint.
        already_has = False
        for r in existing + new_rels:
            if pn in r.get("params", []):
                obj = r.get("relation_object", {})
                if isinstance(obj, dict) and obj.get("expr_type", "") in (
                    "type_equality", "type_dependency"
                ) and obj.get("expr", ""):
                    already_has = True
                    break
        if already_has:
            continue

        cond_clauses: list[str] = []
        for cond, dt in cond_branches.items():
            cond_var = "quantization" if "量化" in cond else cond
            cond_clauses.append(
                f"({pn}.dtype == '{dt}') if {cond_var} "
                f"else ({pn}.dtype == '{default_dtype}')"
            )
        expr = " and ".join(cond_clauses) if cond_clauses else ""
        if not expr or _expr_exists(existing + new_rels, expr):
            continue

        new_rels.append(_make_self_rel(
            fn, pn, expr,
            f"{pn} 的数据类型依赖量化场景",
            f"data_type条件分析: {dtype_desc}",
            "type_dependency",
        ))
    return new_rels


# ===================================================================
# Node entry point
# ===================================================================


async def constraint_extract_node(state: PipelineState) -> dict[str, Any]:
    """Unified constraint extraction node (Pass 1-7).

    Runs independent Passes:
      Pass 1: cross-parameter (dtype/shape equality, length, divisibility)
      Pass 2: single-parameter (empty tensor, tensorlist, shape bound, bool, string length)
      Pass 3: implicit variable value (K1<65536, H=32/64)
      Pass 3b: implicit variable dimension equalities (K1==N2, N1==2*K2)
      Pass 4: value range (deterministic regex + YAML + char* enum)
      Pass 4b: per-platform scalar enum (只支持传1 / 可以配置为0或者1)
      Pass 5: LLM fallback (agent-based relation discovery + expr generation)
      Pass 6: conditional constraints from constraints text (量化/非量化场景)
      Pass 6b: conditional shapes from param descriptions (有/无专家, per-channel/tensor)
      Pass 7: dtype constraints from dtype_combinations table
      Pass 7b: conditional dtype (type_dependency) from structured dtype JSON

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
        all_params = await _mcp_client.query_params_by_doc_id(doc_id)
        existing = await _mcp_client.query_param_relations(doc_id)

        if not all_params:
            logger.info("ConstraintExtract: no parameters, skipping")
            return {"error": None}

        # Filter out infrastructure params (workspace, workspaceSize, executor, stream)
        # — these are not real operator parameters and should never appear in constraints.
        from agent.utils.param_validators import EXCLUDED_PARAMS
        params = [p for p in all_params if p.get("param_name", "") not in EXCLUDED_PARAMS]
        skipped = len(all_params) - len(params)
        if skipped:
            logger.info(
                "ConstraintExtract: filtered out %d infrastructure params: %s",
                skipped,
                [p["param_name"] for p in all_params if p.get("param_name", "") in EXCLUDED_PARAMS],
            )

        _emit(EventType.NODE_PROGRESS, {
            "message": f"已加载 {len(params)} 个参数, {len(existing)} 条已有约束",
            "phase": "data_ready",
            "params_count": len(params),
            "existing_count": len(existing),
        })

        # Fetch supported platforms early — needed by Pass 4b (per-platform
        # enum extraction) and the final expand_common_in_relations step.
        platforms_data = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        supported_platforms = [
            p["platform_name"] for p in platforms_data if p.get("is_supported") == 1
        ]

        all_new: list[dict] = []
        pass1_count = 0
        pass2_count = 0
        pass3_count = 0
        pass3b_count = 0
        pass4_count = 0
        pass4b_count = 0
        pass5_count = 0
        pass6_count = 0
        pass6b_count = 0
        pass7_count = 0
        pass7b_count = 0

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
        sections_text = ""
        try:
            from agent.utils.section_utils import resolve_ws_exe_content

            # resolve_ws_exe_content centralises the single-function exe->ws
            # promotion (params_get_workspace empty -> use params_execute) and
            # the constraints append, shared with fetch_sections. Ordering
            # invariant: promote first, then append constraints to ws_content.
            ws_content, exe_content, constraints_text = await resolve_ws_exe_content(
                _mcp_client, doc_id,
            )
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

        # --- Pass 3b: Implicit variable dimension equalities ---
        # Captures inter-variable equalities (K1==N2, N1==2*K2) that Pass 1
        # (param-vs-param) and Pass 3 (var-vs-numeric) both miss. Runs in an
        # independent try/except so Pass 3 failure (which blanks
        # sections_text) does not block it — in that case Pass 3b naturally
        # returns [] without breaking the pipeline. implicit_params is
        # re-read from state in case Pass 3 set it.
        try:
            implicit_params = state.get("implicit_params", [])
            pass3b_results = _pass3b_dim_equalities(
                sections_text, implicit_params, existing + all_new,
            )
            all_new.extend(pass3b_results)
            pass3b_count = len(pass3b_results)
            logger.info(
                "ConstraintExtract Pass 3b (dim-equality): %d new constraints",
                pass3b_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 3b failed")

        # --- Pass 4: Value range (deterministic regex + YAML + char* enum) ---
        try:
            pass4_results = await _pass4_value_range(
                params, existing + all_new, constraints_text, sections_text,
            )
            all_new.extend(pass4_results)
            pass4_count = len(pass4_results)
            logger.info(
                "ConstraintExtract Pass 4 (value-range): %d new constraints",
                pass4_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 4 failed")

        # --- Pass 4b: Per-platform scalar enum extraction ---
        # Extracts per-platform allowed values for int64_t params like
        # innerPrecise where usage_notes says "只支持传1" for one platform
        # but "0或1" for another.
        try:
            pass4b_results = _pass4b_per_platform_scalar_enum(
                params, existing + all_new, supported_platforms, constraints_text,
            )
            all_new.extend(pass4b_results)
            pass4b_count = len(pass4b_results)
            logger.info(
                "ConstraintExtract Pass 4b (per-platform scalar enum): %d new constraints",
                pass4b_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 4b failed")

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

        # --- Pass 6: Conditional constraints from constraints text ---
        # Extracts multi-param conditional constraints like
        # "innerPrecise参数在BFLOAT16非量化场景，只能配置为0；..."
        # Creates relations with empty expr for the complex relation Agent
        # to generate in build_param_relations.
        try:
            all_param_names = {p.get("param_name", "") for p in params if p.get("param_name")}
            pass6_results = _pass6_conditional_constraints(
                params, existing + all_new, constraints_text, all_param_names,
            )
            all_new.extend(pass6_results)
            pass6_count = len(pass6_results)
            logger.info(
                "ConstraintExtract Pass 6 (conditional constraints): %d new constraints",
                pass6_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 6 failed")

        # --- Pass 6b: Conditional shapes from param descriptions ---
        # Scans param_desc/llm_description for conditional shape patterns
        # (multi-shape candidates, expert-conditional, quantization-conditional)
        # that are NOT in the constraints section but in the parameter table.
        try:
            pass6b_results = _pass6b_conditional_shapes_from_params(
                params, existing + all_new, all_param_names,
            )
            all_new.extend(pass6b_results)
            pass6b_count = len(pass6b_results)
            logger.info(
                "ConstraintExtract Pass 6b (conditional shapes from params): %d new constraints",
                pass6b_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 6b failed")

        # --- Pass 7: Dtype constraints from dtype_combinations ---
        # Generates type_equality relations from the dtype_combos table.
        try:
            dtype_combos = await _mcp_client.query_dtype_combos_by_doc_id(doc_id)
            pass7_results = await _pass7_dtype_constraints(
                params, existing + all_new, dtype_combos or [],
            )
            all_new.extend(pass7_results)
            pass7_count = len(pass7_results)
            logger.info(
                "ConstraintExtract Pass 7 (dtype constraints): %d new constraints",
                pass7_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 7 failed")

        # --- Pass 7b: Conditional dtype (type_dependency) ---
        # Reads structured dtype JSON ({"量化": "INT8", "*": "FLOAT16"})
        # written by dtype_extract_node and generates type_dependency
        # constraints. Silently skips non-conditional dtypes.
        try:
            pass7b_results = _pass7b_conditional_dtype(
                params, existing + all_new,
            )
            all_new.extend(pass7b_results)
            pass7b_count = len(pass7b_results)
            logger.info(
                "ConstraintExtract Pass 7b (conditional-dtype): %d new constraints",
                pass7b_count,
            )
        except Exception:
            logger.exception("ConstraintExtract Pass 7b failed")

        # --- Phase 3 Item 7: cross-source semantic dedup ---
        # Runs after ALL Passes have extended ``all_new``, BEFORE the
        # ``_emit`` progress event so ``new_count`` reflects the post-dedup
        # net increment (pass_count = raw extract count, new_count = net).
        # Platform-aware: dedup key includes platform so common/A2/A3 rows
        # are never merged across platforms. Only ``all_new`` (this run's
        # additions) can be removed; ``existing`` is untouched.
        if settings.constraint_semantic_dedup:
            try:
                _pre_dedup = len(all_new)
                _, all_new = _deduplicate_relations(existing, all_new)
                if len(all_new) != _pre_dedup:
                    logger.info(
                        "ConstraintExtract dedup: %d -> %d new after semantic dedup",
                        _pre_dedup, len(all_new),
                    )
            except Exception:
                logger.exception("ConstraintExtract dedup failed (non-blocking)")

        _emit(EventType.NODE_PROGRESS, {
            "message": (
                f"P1:{pass1_count} P2:{pass2_count} P3:{pass3_count} "
                f"P3b:{pass3b_count} "
                f"P4:{pass4_count} P4b:{pass4b_count} P5:{pass5_count} "
                f"P6:{pass6_count} P6b:{pass6b_count} P7:{pass7_count} "
                f"P7b:{pass7b_count} "
                f"合计新增 {len(all_new)} 条"
            ),
            "phase": "extract_done",
            "pass1_count": pass1_count,
            "pass2_count": pass2_count,
            "pass3_count": pass3_count,
            "pass3b_count": pass3b_count,
            "pass4_count": pass4_count,
            "pass4b_count": pass4b_count,
            "pass5_count": pass5_count,
            "pass6_count": pass6_count,
            "pass6b_count": pass6b_count,
            "pass7_count": pass7_count,
            "pass7b_count": pass7b_count,
            "new_count": len(all_new),
        })

        # Step 2: Merge and save
        if all_new:
            merged = existing + all_new
            # Expand platform="common" to per-platform rows before DB save
            # supported_platforms was fetched earlier (before Pass 4b)
            merged = expand_common_in_relations(merged, supported_platforms)
            result = await _mcp_client.save_param_relations(doc_id, merged)
            logger.info(
                "ConstraintExtract: saved %d total relations "
                "(%d existing + %d new [P1=%d, P2=%d, P3=%d, P4=%d, P4b=%d, "
                "P5=%d, P6=%d, P6b=%d, P7=%d, P7b=%d])",
                result.get("saved", 0),
                len(existing),
                len(all_new),
                pass1_count, pass2_count, pass3_count,
                pass4_count, pass4b_count, pass5_count,
                pass6_count, pass6b_count, pass7_count, pass7b_count,
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