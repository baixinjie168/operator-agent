"""Tests for constraint_extract.py: dedup, HTML cleaning, dtype context, Pass 7.

Covers:
- _normalize_expr: commutative equivalence for A==B / len()==len() / shape==shape
- _expr_exists: dedup with normalization
- _strip_html: HTML tag removal from section content
- _pass6b_conditional_shapes_from_params: conditional shape pattern detection
- _pass7_dtype_constraints: dtype constraint generation from dtype_combos
"""

from __future__ import annotations

import json

import pytest

from agent.nodes.constraint_extract import (
    _expr_exists,
    _normalize_expr,
    _strip_html,
    _pass6b_conditional_shapes_from_params,
)


# ── _normalize_expr ──────────────────────────────────────────────────────────

class TestNormalizeExpr:
    """Test that commutative expressions normalize to the same form."""

    def test_dtype_equality_commutive(self):
        """A.dtype == B.dtype and B.dtype == A.dtype should be equal."""
        a = _normalize_expr("x.dtype == y.dtype")
        b = _normalize_expr("y.dtype == x.dtype")
        assert a == b
        assert a == "x.dtype == y.dtype"

    def test_len_equality_commutive(self):
        a = _normalize_expr("len(x) == len(y)")
        b = _normalize_expr("len(y) == len(x)")
        assert a == b
        assert a == "len(x) == len(y)"

    def test_shape_equality_commutive(self):
        a = _normalize_expr("x.shape == y.shape")
        b = _normalize_expr("y.shape == x.shape")
        assert a == b
        assert a == "x.shape == y.shape"

    def test_non_commutative_unchanged(self):
        """Non-equality expressions should pass through unchanged."""
        assert _normalize_expr("x % 2 == 0") == "x % 2 == 0"
        assert _normalize_expr("x.range_value in [1, 2]") == "x.range_value in [1, 2]"

    def test_empty_expr(self):
        assert _normalize_expr("") == ""

    def test_three_part_equality_not_normalized(self):
        """Complex expressions with != or other ops are not normalized."""
        expr = "x.shape[0] == y.shape[0]"
        assert _normalize_expr(expr) == expr  # shape[i] not handled, passes through


# ── _expr_exists ─────────────────────────────────────────────────────────────

class TestExprExists:
    """Test dedup logic with normalization."""

    def test_detects_communicative_duplicate(self):
        """Should detect B==A as duplicate of A==B."""
        existing = [
            {"relation_object": json.dumps({"expr": "y.dtype == x.dtype"})},
        ]
        assert _expr_exists(existing, "x.dtype == y.dtype") is True

    def test_detects_len_communicative(self):
        existing = [
            {"relation_object": json.dumps({"expr": "len(y) == len(x)"})},
        ]
        assert _expr_exists(existing, "len(x) == len(y)") is True

    def test_no_false_positive(self):
        existing = [
            {"relation_object": json.dumps({"expr": "x.dtype == y.dtype"})},
        ]
        assert _expr_exists(existing, "x.dtype == z.dtype") is False

    def test_handles_string_relation_object(self):
        existing = [
            {"relation_object": {"expr": "y.shape == x.shape"}},
        ]
        assert _expr_exists(existing, "x.shape == y.shape") is True

    def test_handles_invalid_json(self):
        existing = [
            {"relation_object": "not json"},
        ]
        assert _expr_exists(existing, "x.dtype == y.dtype") is False

    def test_empty_existing(self):
        assert _expr_exists([], "x.dtype == y.dtype") is False


# ── _strip_html ──────────────────────────────────────────────────────────────

class TestStripHtml:
    """Test HTML tag removal."""

    def test_removes_table_tags(self):
        assert _strip_html("<td>hello</td>") == "hello"

    def test_removes_closing_tags(self):
        assert _strip_html("</table>text<br/>") == "text"

    def test_preserves_inner_text(self):
        assert _strip_html("<p>line1</p><p>line2</p>") == "line1line2"

    def test_no_html_unchanged(self):
        assert _strip_html("no html here") == "no html here"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_none_passthrough(self):
        assert _strip_html(None) is None

    def test_nested_tags(self):
        assert _strip_html("<div><span>nested</span></div>") == "nested"

    def test_strips_with_attributes(self):
        assert _strip_html('<a href="link">text</a>') == "text"


# ── _pass6b_conditional_shapes_from_params ───────────────────────────────────

class TestPass6bConditionalShapes:
    """Test conditional shape detection from parameter descriptions."""

    def test_detects_multi_shape_candidate(self):
        """Should detect [E,K1,N1]/[K1,N1] pattern."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "shape: [E,K1,N1]/[K1,N1]", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"weight1"})
        assert len(results) == 1
        assert results[0]["relation_object"]["expr_type"] == "shape_value_dependency"
        assert results[0]["relation_object"]["expr"] == ""  # empty for Agent

    def test_detects_expert_conditional(self):
        """Should detect 有专家/无专家 pattern."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "有专家时3维，无专家时2维", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"weight1"})
        assert len(results) == 1

    def test_detects_quantization_conditional(self):
        """Should detect per-channel/per-tensor pattern."""
        params = [
            {"param_name": "scale", "function_name": "fn",
             "param_desc": "per-channel时为[E,N1]，per-tensor时为[N1]", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"scale"})
        assert len(results) == 1

    def test_no_pattern_no_result(self):
        """Should return empty list for params without conditional shapes."""
        params = [
            {"param_name": "x", "function_name": "fn",
             "param_desc": "普通参数描述", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"x"})
        assert len(results) == 0

    def test_dedup_against_existing(self):
        """Should skip params that already have shape_value_dependency."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "[E,K1,N1]/[K1,N1]", "llm_description": ""},
        ]
        existing = [
            {"params": ["weight1"],
             "relation_object": {"expr_type": "shape_value_dependency", "expr": "x"}},
        ]
        results = _pass6b_conditional_shapes_from_params(params, existing, {"weight1"})
        assert len(results) == 0

    def test_strips_html_from_text(self):
        """Should strip HTML from param descriptions before pattern matching."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "<td>[E,K1,N1]/[K1,N1]</td>", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"weight1"})
        assert len(results) == 1
        # src_text should not contain HTML tags
        src = results[0]["source_citation"]
        assert "<td>" not in src
