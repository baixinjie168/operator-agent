"""Derive shape-equality groups and fixed-value map from constraints.

Adapted from the legacy ``_build_shape_equal_groups`` and
``_build_fixed_values`` methods in
``OperatorTestProject/src/test_case_generator.py``.

The new format uses ``expr_type`` on each constraint entry; the legacy format
relied on free-form ``type: "shape_unification" / "shape_equality" / "fixed_value"``
strings.  Only ``shape_equality`` and ``shape_unification`` produce equality
groups; ``fixed_value`` populates a separate fixed-value map.
"""

from __future__ import annotations

from typing import Any

from shared.models.enums import ConstraintExprType

_FALLBACK_FALLBACK_PLATFORM = "default"


def build_shape_equal_groups(context: dict[str, Any]) -> list[list[str]]:
    """Return equivalence groups of parameter names that must share the same shape.

    Args:
        context: Mapping containing ``constraints_in_parameters`` keyed by platform.
                 Each constraint has ``expr_type`` and ``relation_params``.

    Returns:
        List of groups, each a list of parameter names.  Groups with fewer
        than 2 params are omitted.  Returns an empty list if no shape
        constraints are present.
    """
    cip = context.get("constraints_in_parameters") or {}
    if not isinstance(cip, dict):
        return []

    groups: list[list[str]] = []
    for constraints in cip.values():
        if not isinstance(constraints, list):
            continue
        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            expr_type = constraint.get("expr_type", "")
            if expr_type not in (
                ConstraintExprType.SHAPE_EQUALITY,
                ConstraintExprType.SHAPE_UNIFICATION,
            ):
                continue
            params = constraint.get("relation_params") or []
            valid = [p for p in params if isinstance(p, str)]
            if len(valid) > 1:
                groups.append(valid)
    return groups


def build_fixed_values(context: dict[str, Any]) -> dict[str, float]:
    """Return a mapping of parameter name → fixed scalar value.

    Parses ``fixed_value`` expressions of the form ``"<name> == <number>"``.
    Non-numeric RHS or expressions without ``==`` are silently skipped.
    """
    cip = context.get("constraints_in_parameters") or {}
    if not isinstance(cip, dict):
        return {}

    fixed_map: dict[str, float] = {}
    for constraints in cip.values():
        if not isinstance(constraints, list):
            continue
        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            if constraint.get("expr_type") != ConstraintExprType.FIXED_VALUE:
                continue
            expr = constraint.get("expr", "")
            if not isinstance(expr, str) or "==" not in expr:
                continue
            parts = expr.split("==", maxsplit=1)
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            value_str = parts[1].strip()
            try:
                fixed_map[name] = float(value_str)
            except ValueError:
                # Non-numeric fixed value (e.g. mode == 'relu'); skip silently.
                continue
    return fixed_map
