"""Tests for expr_validation.py: Phase 0 syntax + reference + semantic checks.

Covers Phase 0c (validate_expr_semantic) — tautology / empty-bool /
excessive-redundancy detection — and its wiring into validate_expr.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure agent package is importable
ROOT = Path(__file__).resolve().parents[2]
for pkg in ("packages/shared/src", "packages/mcp-server/src", "packages/agent/src"):
    p = str(ROOT / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.utils.expr_validation import (  # noqa: E402
    _semantic_expr_key,
    _simplify_expr,
    validate_expr,
    validate_expr_refs,
    validate_expr_semantic,
)

# ---------------------------------------------------------------------------
# Phase 0c: validate_expr_semantic — tautology / empty-bool / redundancy
# ---------------------------------------------------------------------------


class TestValidateExprSemantic:
    def test_literal_true_tautology(self):
        """T2-1: literal True is a no-op constraint."""
        ok, err = validate_expr_semantic("True")
        assert ok is False
        assert "恒真" in err

    def test_empty_bool_all_true(self):
        """T2-2: BoolOp whose operands are all True."""
        ok, err = validate_expr_semantic("True and True and True")
        assert ok is False
        assert "空布尔" in err

    def test_empty_bool_or(self):
        ok, err = validate_expr_semantic("True or True")
        assert ok is False
        assert "空布尔" in err

    def test_excessive_redundancy(self):
        """T2-3: same Compare sub-expression repeated 3+ times."""
        ok, err = validate_expr_semantic(
            "a is None and a is None and a is None"
        )
        assert ok is False
        assert "冗余" in err
        assert "3" in err

    def test_redundancy_threshold_exactly_two_ok(self):
        """Two repetitions must NOT trigger redundancy (threshold is 3)."""
        ok, _ = validate_expr_semantic("a is None and a is None")
        assert ok is True

    def test_valid_equality(self):
        """T2-4: a legitimate non-tautological expression passes."""
        ok, _ = validate_expr_semantic("(a is None) != (b is None)")
        assert ok is True

    def test_valid_multibranch_any(self):
        """T2-5: multi-branch any() with differing sub-items passes."""
        ok, _ = validate_expr_semantic("any(p is None for p in [a, b, c])")
        assert ok is True

    def test_empty_expr_ok(self):
        ok, _ = validate_expr_semantic("")
        assert ok is True

    def test_syntax_error_delegated(self):
        """Syntax errors are delegated to Phase 0a (returns valid here)."""
        ok, _ = validate_expr_semantic("a == (")
        assert ok is True

    def test_ffnv3_redundancy_pattern(self):
        """T2-6: FFNV3 [50]-style copy-paste redundancy is caught."""
        # Simulate the deqScale1Optional is None ... repeated pattern
        expr = " and ".join(["deqScale1Optional is None"] * 5)
        ok, err = validate_expr_semantic(expr)
        assert ok is False
        assert "冗余" in err


# ---------------------------------------------------------------------------
# Phase 0: validate_expr wiring (syntax + refs + semantic)
# ---------------------------------------------------------------------------


class TestValidateExprWiring:
    def test_literal_true_rejected_through_validate_expr(self):
        """validate_expr must reject tautological True via Phase 0c."""
        ok, err = validate_expr("True", params=["x"])
        assert ok is False
        assert "恒真" in err

    def test_redundancy_rejected_through_validate_expr(self):
        ok, err = validate_expr(
            "a is None and a is None and a is None", params=["a"],
        )
        assert ok is False
        assert "冗余" in err

    def test_valid_expr_passes_all_phases(self):
        ok, err = validate_expr("x.shape[0] == y.shape[0]", params=["x", "y"])
        assert ok is True
        assert err == ""

    def test_syntax_error_caught_by_phase_0a(self):
        ok, err = validate_expr("x ==", params=["x"])
        assert ok is False
        assert "SyntaxError" in err

    def test_unknown_param_caught_by_phase_0b(self):
        ok, err = validate_expr("z.shape[0] == 1", params=["x"])
        assert ok is False
        assert "Unknown parameter" in err


# ---------------------------------------------------------------------------
# Fix 0: _ALLOWED_MODULES — math.ceil/floor/... no longer rejected
# ---------------------------------------------------------------------------

class TestAllowedModules:
    """Fix 0: whitelist module attributes (math.ceil etc.) pass Phase 0b.

    RELATION_OBJECT_BUILD_PROMPT 示例 7 自身使用 math.ceil；扩 _BUILTIN_NAMES
    + _ALLOWED_MODULES 前会被误判 "Unknown attribute: '.ceil'"。
    """

    def test_math_ceil_passes(self):
        ok, err = validate_expr(
            "math.ceil(x.shape[0] / 16) == y.shape[2]", params=["x", "y"],
        )
        assert ok is True, err
        assert err == ""

    def test_math_floor_passes(self):
        ok, err = validate_expr(
            "math.floor(x.range_value / 16) == 0", params=["x"],
        )
        assert ok is True, err

    def test_math_attr_direct_refs(self):
        ok, err = validate_expr_refs("math.ceil(x.shape[0])", params=["x"])
        assert ok is True, err

    def test_non_allowed_attribute_still_rejected(self):
        """x.unknown_attr 仍被拒（不在 _ALLOWED_ATTRS）。"""
        ok, err = validate_expr("x.unknown_attr == 1", params=["x"])
        assert ok is False
        assert "Unknown attribute" in err
        assert "unknown_attr" in err

    def test_unknown_module_name_rejected(self):
        """未白名单的模块名仍被拒（os.path 不被放行）。"""
        ok, err = validate_expr("os.path == 'a'", params=["x"])
        assert ok is False
        assert "Unknown" in err  # Unknown attribute(.path) 或 Unknown parameter(os)


# ---------------------------------------------------------------------------
# Phase 3 Item 8: _semantic_expr_key + _simplify_expr
# ---------------------------------------------------------------------------

class TestSemanticExprKey:
    """T7 / shared: AST-canonical key for semantic-equivalence dedup."""

    def test_commutative_equality_same_key(self):
        k1 = _semantic_expr_key("x.dtype == y.dtype")
        k2 = _semantic_expr_key("y.dtype == x.dtype")
        assert k1 == k2 and k1.startswith("eq(")

    def test_commutative_and_same_key(self):
        k1 = _semantic_expr_key("a == 1 and b == 2")
        k2 = _semantic_expr_key("b == 2 and a == 1")
        assert k1 == k2

    def test_guarded_vs_unguarded_different_key(self):
        """Guarded (X if cond else True) and bare X must NOT share a key."""
        kg = _semantic_expr_key("(x.shape[0] == N) if x is not None else True")
        kn = _semantic_expr_key("x.shape[0] == N")
        assert kg != kn
        assert kg.startswith("guarded(")

    def test_empty_expr_returns_empty(self):
        assert _semantic_expr_key("") == ""

    def test_syntax_error_fallback_to_raw(self):
        """On AST failure, return the raw string (exact-match fallback)."""
        assert _semantic_expr_key("a == (") == "a == ("


class TestSimplifyExpr:
    """T8-*: Phase 3 Item 8 — post-generation factoring."""

    def test_t8_2_factors_repeated_subexpr(self):
        """3+ repeated Compare sub-expr factored out."""
        expr = (
            "deqScale1Optional is None and "
            "deqScale1Optional is None and "
            "deqScale1Optional is None and "
            "antiquantScale1Optional is not None and "
            "innerPrecise.range_value == True"
        )
        simplified = _simplify_expr(expr)
        assert simplified != expr
        # common factor appears once now
        assert simplified.count("deqScale1Optional is None") == 1

    def test_t8_3_valid_multibranch_not_simplified(self):
        """Multi-branch any() with differing items is NOT touched."""
        expr = "any(p is None for p in [a, b, c, d])"
        # length < 100 so returns unchanged regardless
        assert _simplify_expr(expr) == expr

    def test_t8_5_simplify_invalid_syntax_returns_original(self):
        """If simplified form has invalid syntax, original is returned."""
        expr = "a == ( and b" + " x" * 100
        assert _simplify_expr(expr) == expr

    def test_t8_9_short_expr_untouched(self):
        """Short expr (<100 chars) is never simplified."""
        assert _simplify_expr("a is None and a is None and a is None") == \
            "a is None and a is None and a is None"

    def test_t8_nested_ifexp_simplify(self):
        """R3: nested IfExp with repeated And sub-expr is simplified."""
        # Build a >100 char nested IfExp with 3+ repeats in the body
        body = " and ".join(["deqScale1Optional is None"] * 3) + " and b == 1"
        expr = f"({body}) if quantization else True"
        simplified = _simplify_expr(expr)
        assert simplified != expr
        assert simplified.count("deqScale1Optional is None") == 1
