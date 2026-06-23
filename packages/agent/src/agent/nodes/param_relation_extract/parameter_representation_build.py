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
        → expr: ``BS.range_value == x1.shape[0]``
    - "x2.shape[0] is represented by H*rankSize"
        → expr: ``x2.shape[0] == H.range_value*rankSize.range_value``
    - "rankSize on Atlas A2 ∈ {2, 4, 8}"
        → expr: ``rankSize.range_value in [2, 4, 8]``

Zero LLM calls. Position in subgraph::

    implicit_param_extract ─┬→ extract_ws     ─┐
                            ├→ extract_exe     ─┤→ merge_relations → save_relations → END
                            └→ param_repr_build ─────────────────────────────────────→ END
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


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
) -> tuple[str, list[str]]:
    """Convert a shape slot expression into a Python expression fragment.

    Each identifier is rewritten:
    - excluded words (``int``, ``shape``, ...) → kept as-is
    - known constant → its numeric value (e.g. ``k0`` → ``16``)
    - named variable → ``var.range_value`` (e.g. ``BS`` → ``BS.range_value``)

    Returns ``(python_expr, ordered_var_names)``.
    """
    var_names: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(0)
        if name in _EXCLUDE_WORDS:
            return name
        if name in constant_values:
            return str(constant_values[name])
        if name not in var_names:
            var_names.append(name)
        return f"{name}.range_value"

    python_expr = _DIM_VAR_RE.sub(_replace, slot_expr)
    return python_expr, var_names


def _build_tensor_representations(
    mappings: list[dict],
) -> list[dict[str, Any]]:
    """Generate ``parameter_representation`` records for tensor shape dims.

    For each (tensor_param, dim_index) slot that contains at least one
    named variable, emit one record::

        {tensor}.shape[{dim}] == {python_expr}

    where ``python_expr`` is the slot expression rewritten with
    ``var.range_value`` substitutions and constant values.
    """
    constant_values = _build_constant_values(mappings)
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

        slot_key = (tensor, dim)
        if slot_key in seen_slots:
            continue
        seen_slots.add(slot_key)

        python_expr, var_names = _slot_expr_to_python(slot_expr, constant_values)

        # Skip slots that reduced to a pure number (no variable references)
        if not var_names:
            continue

        reps.append({
            "expr_type": "parameter_representation",
            "expr": f"{tensor}.shape[{dim}] == {python_expr}",
            "relation_params": [tensor, *var_names],
            "src_text": m.get("shape_text", ""),
        })

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

    try:
        tensor_reps = _build_tensor_representations(mappings)
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
