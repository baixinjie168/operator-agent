"""ParameterRepresentationBuild node: deterministically generate
``parameter_representation`` constraint records.

These records declare how each operator-parameter shape dimension is
expressed through named (non-operator) variables, and how platform
external constants are constrained.

They are *not* extracted by the LLM — they are derived deterministically
from the ``implicit_params`` mappings (produced by
``implicit_param_extract``) and ``platform_constants``. Keeping this
deterministic avoids LLM hallucination on what is essentially a
mechanical transcription of shape tuples.

Examples (aclnnAlltoAllMatmul):
    - "BS is represented by x1.shape[0]"
        → expr: ``x1.shape[0] == BS``
    - "x2.shape[0] is represented by H*rankSize"
        → expr: ``x2.shape[0] == H*rankSize.range_value``
    - "rankSize on Atlas A2 ∈ {2, 4, 8}"
        → expr: ``rankSize.range_value in [2, 4, 8]``

Dimension variables (BS, H, b, m, k, ...) are referenced directly by name —
they *are* the symbolic dimension size.  External constants (rankSize, ...)
keep ``.range_value`` because they have a real platform-dependent value
range.  Platform-constant membership checks also use ``.range_value``.

Zero LLM calls. Position in subgraph::

    implicit_param_extract ─┬→ extract_ws     ─┐
                            ├→ extract_exe     ─┤→ merge_relations → save_relations → END
                            └→ param_repr_build ─────────────────────────────────────→ END
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.implicit_param_extract import (
    _DIM_VAR_RE,
    _EXCLUDE_WORDS,
)
from agent.nodes.param_relation_extract.state import RelationExtractState
from agent.utils.expr_validation import validate_expr_syntax
from agent.utils.param_validators import is_tensor_type

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


def _build_tensor_param_names(param_types: dict[str, str]) -> set[str]:
    """Return the set of parameter names whose type is ``aclTensor*``.

    Used as the downstream defense line (Item 0): even if upstream
    ``implicit_param_extract`` misclassifies a non-Tensor parameter
    (e.g. ``transposeX2(bool)``, ``alltoAllAxesOptional(aclIntArray)``)
    as a ``tensor_param``, this filter prevents generating a bogus
    ``{non_tensor}.shape[{dim}] == {expr}`` constraint.
    """
    return {n for n, t in param_types.items() if is_tensor_type(t)}


# ---------------------------------------------------------------------------
# Condition guard detection (Item 5)
# ---------------------------------------------------------------------------

# Condition keywords that indicate a shape is conditional.  Each entry maps
# a document keyword to a Python condition fragment used as the guard.
_SHAPE_CONDITION_KEYWORDS: list[tuple[str, str]] = [
    ("per-channel", "per_channel"),
    ("per channel", "per_channel"),
    ("per-group", "per_group"),
    ("per group", "per_group"),
    ("per-tensor", "per_tensor"),
    ("per tensor", "per_tensor"),
    ("有专家", "expertTokens is not None"),
    ("无专家", "expertTokens is None"),
]


def _detect_shape_guard(
    tensor_param: str,
    shape_text: str,
    param_desc: str,
    is_optional: bool = False,
) -> str:
    """Detect the condition guard a shape representation needs (Item 5).

    Returns a Python condition fragment (e.g. ``"scaleOptional is not None"``,
    ``"per_channel"``), or an empty string when no guard is needed.

    Detection order (most reliable first):
      1. Optional parameter — name contains "Optional" or is_optional=True
         → ``{tensor} is not None``.
      2. Condition keywords (per-channel/per-group/有专家/无专家) in
         shape_text + param_desc.  ``per-tensor`` is the "else" branch of
         per-channel, so it is skipped here (covered by the per-channel
         guard).
    """
    # 1. Optional-parameter guard (mechanical, most reliable)
    if (
        is_optional
        or "Optional" in tensor_param
        or "optional" in tensor_param.lower()
    ):
        return f"{tensor_param} is not None"

    # 2. Condition-keyword detection in shape_text + param_desc
    combined = " ".join(filter(None, [shape_text, param_desc])).lower()
    for keyword, cond in _SHAPE_CONDITION_KEYWORDS:
        if keyword.lower() in combined:
            # per-tensor is the "else" branch — skip; per-channel guard
            # covers it.
            if cond == "per_tensor":
                continue
            return cond
    return ""


def _build_constant_values(mappings: list[dict]) -> dict[str, int]:
    """Build {var_name: constant_value} for known constants (e.g. k0=16)."""
    out: dict[str, int] = {}
    for m in mappings:
        if m.get("is_constant") and m.get("constant_value") is not None:
            out[m["var_name"]] = m["constant_value"]
    return out


def _slot_expr_to_python(
    slot_expr: str,
    constant_values: dict[str, int],
    external_const_names: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Convert a shape slot expression into a Python expression fragment.

    Each identifier is rewritten:
    - excluded words (``int``, ``shape``, ...) → kept as-is
    - known constant → its numeric value (e.g. ``k0`` → ``16``)
    - external constant → ``var.range_value`` (e.g. ``rankSize`` →
      ``rankSize.range_value``) — external constants have a real
      platform-dependent range_value attribute
    - dimension variable → ``var`` (e.g. ``b`` → ``b``, ``H`` → ``H``)
      — a dimension variable *is* the symbolic dimension size, so it is
      referenced directly by name without ``.range_value``

    Returns ``(python_expr, ordered_var_names)``.
    """
    ext_set = external_const_names or set()
    var_names: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(0)
        if name in _EXCLUDE_WORDS:
            return name
        if name in constant_values:
            return str(constant_values[name])
        if name not in var_names:
            var_names.append(name)
        # External constants have a real range_value (platform-dependent
        # values); dimension variables are used directly by name.
        if name in ext_set:
            return f"{name}.range_value"
        return name

    python_expr = _DIM_VAR_RE.sub(_replace, slot_expr)
    return python_expr, var_names


def _split_top_level_commas(slot_expr: str) -> list[str]:
    """Split a shape slot expression on top-level commas only.

    FIX-3: Chinese commas (，) and colons (：) are normalized first.
    Nested commas inside function calls like ``func(a, b)`` are NOT split,
    because only the top-level Tuple/List elements are separated.

    Examples:
        "m, k"       → ["m", "k"]        (two dim variables)
        "func(a, b)" → ["func(a, b)"]   (single expr, nested comma preserved)
        "[m, k]"     → ["m", "k"]        (list literal → elements)
    """
    normalized = slot_expr.replace("，", ",").replace("：", ":")
    normalized = normalized.replace("　", " ")  # full-width space
    try:
        tree = ast.parse(normalized, mode="eval")
        body = tree.body
        if isinstance(body, (ast.Tuple, ast.List)):
            parts = [ast.unparse(elt) for elt in body.elts]
            return [p.strip() for p in parts if p.strip()]
    except SyntaxError:
        pass
    return [normalized.strip()]


def _build_tensor_representations(
    mappings: list[dict],
    tensor_param_names: set[str] | None = None,
    param_descs: dict[str, str] | None = None,
    param_optional: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Generate ``parameter_representation`` records for tensor shape dims.

    For each (tensor_param, dim_index) slot that contains at least one
    named variable, emit one record::

        {tensor}.shape[{dim}] == {python_expr}

    When a condition guard is detected (Item 5) — the parameter is
    optional, or its description mentions per-channel/per-group/有专家 —
    the expression is wrapped::

        ({tensor}.shape[{dim}] == {python_expr}) if {guard} else True

    This eliminates over-broad constraints that trigger WARNs when the
    condition does not hold (e.g. per-tensor scenario).

    Args:
        mappings: implicit_params mappings from implicit_param_extract.
        tensor_param_names: authoritative set of aclTensor* parameter
            names. When provided (not None), non-Tensor ``tensor_param``
            values are skipped with a warning — this is the downstream
            defense against upstream misclassification. When None (type
            query failed), degrades to the original permissive behavior
            so the pipeline never breaks.
        param_descs: ``{param_name: param_desc}`` map from DB, used by
            condition-guard detection. Defaults to ``{}`` (degrades to
            name-only optional detection).
        param_optional: ``{param_name: is_optional}`` map from DB, used
            by condition-guard detection. Defaults to ``{}``.
    """
    constant_values = _build_constant_values(mappings)
    external_const_names = {
        m["var_name"] for m in mappings if m.get("is_external_constant")
    }
    desc_map = param_descs or {}
    opt_map = param_optional or {}
    reps: list[dict[str, Any]] = []
    seen_slots: set[tuple[str, int]] = set()

    for m in mappings:
        if m.get("is_external_constant"):
            continue  # handled via platform_constants
        if m.get("is_constant"):
            continue  # constants are substituted as values, not represented
        if m.get("is_quantization_type"):
            continue  # char-typed enum, no tensor shape to represent

        tensor = m.get("tensor_param") or ""
        dim = m.get("dim_index")
        slot_expr = m.get("slot_expr") or ""

        if not tensor or dim is None or not slot_expr.strip():
            continue

        # Downstream defense (Item 0): skip non-Tensor parameters to avoid
        # emitting bogus shape constraints like ``transposeX2.shape[0] == N``
        # when upstream misclassified a bool / aclIntArray param as a
        # tensor_param. Degrades to permissive behavior when the type set
        # is unavailable (None).
        if tensor_param_names is not None and tensor not in tensor_param_names:
            logger.warning(
                "ParameterRepresentationBuild: 跳过非 Tensor 参数 %s "
                "(类型非 aclTensor*)，避免生成非法 shape 约束", tensor,
            )
            continue

        slot_key = (tensor, dim)
        if slot_key in seen_slots:
            continue
        seen_slots.add(slot_key)

        # FIX-3: split top-level commas (e.g. "m, k" → ["m", "k"]).
        # Each part corresponds to a consecutive dimension starting at *dim*.
        parts = _split_top_level_commas(slot_expr)

        cur_dim = dim
        for part in parts:
            python_expr, var_names = _slot_expr_to_python(
                part, constant_values, external_const_names,
            )

            # Skip parts that reduced to a pure number (no variable references)
            if not var_names:
                cur_dim += 1
                continue

            # Item 5: detect condition guard and wrap the expression.
            guard = _detect_shape_guard(
                tensor,
                m.get("shape_text", ""),
                desc_map.get(tensor, ""),
                opt_map.get(tensor, False),
            )
            base_expr = f"{tensor}.shape[{cur_dim}] == {python_expr}"
            final_expr = (
                f"({base_expr}) if {guard} else True" if guard else base_expr
            )

            # FIX-4: validate syntax on the deterministic path. R10: keep
            # the constraint with a _syntax_warning flag instead of dropping.
            is_valid, error = validate_expr_syntax(final_expr)
            rep: dict[str, Any] = {
                "expr_type": "parameter_representation",
                "expr": final_expr,
                "relation_params": [tensor, *var_names],
                # FIX-12: prefer source_section_text (document原文) over
                # shape_text (internal label), falling back to shape_text.
                "src_text": m.get("source_section_text", m.get("shape_text", "")),
            }
            if not is_valid:
                logger.warning(
                    "ParameterRepresentationBuild: syntax error in '%s': %s",
                    final_expr, error,
                )
                rep["_syntax_warning"] = error
            reps.append(rep)
            cur_dim += 1

    return reps


def _build_platform_constant_representations(
    platform_constants: list[dict],
) -> dict[str, list[dict[str, Any]]]:
    """Generate platform-specific ``parameter_representation`` records.

    For each external constant (e.g. ``rankSize``) and each platform's
    allowed value set, emit::

        {const_name}.range_value in [v1, v2, ...]

    Returns ``{platform: [records]}``.
    """
    by_platform: dict[str, list[dict[str, Any]]] = {}

    for pc in platform_constants:
        const_name = pc.get("const_name", "")
        if not const_name:
            continue
        for pv in pc.get("platform_values", []):
            platform = pv.get("platform", "")
            values = pv.get("values", [])
            if not platform or not values:
                continue
            values_str = ", ".join(str(v) for v in values)
            by_platform.setdefault(platform, []).append({
                "expr_type": "parameter_representation",
                "expr": f"{const_name}.range_value in [{values_str}]",
                "relation_params": [const_name],
                "src_text": pv.get("source_citation", ""),
            })

    return by_platform


async def parameter_representation_build_node(
    state: RelationExtractState,
) -> dict[str, Any]:
    """Generate ``parameter_representation`` records and persist to DB.

    Reads ``implicit_params`` mappings and ``platform_constants`` from
    subgraph state (both produced by ``implicit_param_extract_node``).
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info(
        "ParameterRepresentationBuild: doc_id=%s for %s",
        doc_id, operator_name,
    )

    if not doc_id:
        logger.warning("ParameterRepresentationBuild: no doc_id, skipping")
        return {"error": None}

    mappings = state.get("implicit_params", [])
    platform_constants = state.get("platform_constants", [])

    if not mappings and not platform_constants:
        logger.info(
            "ParameterRepresentationBuild: nothing to build for %s",
            operator_name,
        )
        return {"error": None}

    # Build the authoritative Tensor parameter-name set (Item 0 downstream
    # defense). Falls back to None (permissive) when the type query fails so
    # the pipeline never breaks.
    tensor_param_names: set[str] | None = None
    param_descs: dict[str, str] = {}        # Item 5: condition-guard detection
    param_optional: dict[str, bool] = {}    # Item 5: condition-guard detection
    try:
        all_params = await _mcp_client.query_params_by_doc_id(doc_id)
        param_types: dict[str, str] = {}
        for p in all_params:
            pn = p.get("param_name", "")
            if not pn:
                continue
            param_types[pn] = p.get("param_type", "")
            param_descs[pn] = p.get("param_desc", "") or ""
            param_optional[pn] = bool(p.get("is_optional", False))
        if param_types:
            tensor_param_names = _build_tensor_param_names(param_types)
            logger.debug(
                "ParameterRepresentationBuild: %d Tensor params for %s: %s",
                len(tensor_param_names), operator_name,
                sorted(tensor_param_names),
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "ParameterRepresentationBuild: 查询参数类型失败，"
            "Tensor 过滤+条件守卫降级关闭", exc_info=True,
        )

    try:
        tensor_reps = _build_tensor_representations(
            mappings, tensor_param_names, param_descs, param_optional,
        )
        platform_reps = _build_platform_constant_representations(
            platform_constants,
        )

        payload = {
            "representations": tensor_reps,
            "platform_representations": platform_reps,
        }

        await _mcp_client.save_parameter_representations(
            doc_id=doc_id,
            representations_json=json.dumps(payload, ensure_ascii=False),
        )

        platform_reps_count = sum(len(v) for v in platform_reps.values())
        logger.info(
            "ParameterRepresentationBuild: built %d tensor reps + %d "
            "platform-specific reps for %s (doc_id=%s)",
            len(tensor_reps),
            platform_reps_count,
            operator_name,
            doc_id,
        )

        return {"error": None}

    except Exception as e:  # noqa: BLE001
        logger.exception(
            "ParameterRepresentationBuild failed for %s", operator_name,
        )
        return {"error": str(e)}
