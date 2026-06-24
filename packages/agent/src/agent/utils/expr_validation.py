"""Expression validation utilities (Phase 0).

Extracted from ``build_param_relations.py`` into a shared module so that
both ``build_param_relations.py`` and ``complex_relation_agent.py`` can
import them without circular dependency.

Phase 0 provides zero-LLM-cost validation:
- Syntax check: ``ast.parse(expr, mode="eval")``
- Reference check: all Name/Attribute nodes must be valid
"""

from __future__ import annotations

import ast

# ---------------------------------------------------------------------------
# Constants (moved from build_param_relations.py)
# ---------------------------------------------------------------------------

_ALLOWED_ATTRS = {"shape", "dtype", "format", "range_value"}
_BUILTIN_NAMES = {
    "True", "False", "None", "len", "range",
    "all", "any", "int", "float", "str", "bool", "set",
    "min", "max", "list",
}


# ---------------------------------------------------------------------------
# Phase 0a: Syntax validation
# ---------------------------------------------------------------------------


def validate_expr_syntax(expr: str) -> tuple[bool, str]:
    """Validate expr is a legal Python expression.

    Returns:
        (is_valid, error_message)
    """
    if not expr:
        return True, ""  # Empty expression is allowed
    try:
        ast.parse(expr, mode="eval")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


# ---------------------------------------------------------------------------
# Phase 0b: Reference validation
# ---------------------------------------------------------------------------


def validate_expr_refs(
    expr: str,
    params: list[str],
    external_constants: set[str] | None = None,
    implicit_param_names: set[str] | None = None,
) -> tuple[bool, str]:
    """Validate parameter names and attributes in expr.

    Checks:
    1. All Name nodes must be in params, Python builtins,
       external_constants, or implicit_param_names
    2. All Attribute nodes must be in _ALLOWED_ATTRS
    3. Comprehension variables (e.g., 'd' in 'all(d > 0 for d in x.shape)')
       are allowed
    """
    if not expr:
        return True, ""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False, "Invalid syntax"

    param_set = set(params)
    ext_set = external_constants or set()
    implicit_set = implicit_param_names or set()

    # Collect all comprehension variables
    comprehension_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            for generator in node.generators:
                if isinstance(generator.target, ast.Name):
                    comprehension_vars.add(generator.target.id)
                elif isinstance(generator.target, ast.Tuple):
                    for elt in generator.target.elts:
                        if isinstance(elt, ast.Name):
                            comprehension_vars.add(elt.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if (
                node.id not in param_set
                and node.id not in _BUILTIN_NAMES
                and node.id not in comprehension_vars
                and node.id not in ext_set
                and node.id not in implicit_set
            ):
                return False, f"Unknown parameter: '{node.id}'"
        if isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_ATTRS:
                return False, f"Unknown attribute: '.{node.attr}'"
    return True, ""


# ---------------------------------------------------------------------------
# Phase 0: Comprehensive validation (syntax + references)
# ---------------------------------------------------------------------------


def validate_expr(
    expr: str,
    params: list[str],
    external_constants: set[str] | None = None,
    implicit_param_names: set[str] | None = None,
) -> tuple[bool, str]:
    """Phase 0: Comprehensive validation (syntax + references)."""
    is_valid, error = validate_expr_syntax(expr)
    if not is_valid:
        return False, error
    is_valid, error = validate_expr_refs(
        expr, params, external_constants, implicit_param_names,
    )
    if not is_valid:
        return False, error
    return True, ""
