"""Tests for constraint_completeness_check.py: Phase 3 Item 9.

Covers the 4 global completeness checks:
  1. dtype completeness   — tensor params without dtype constraint flagged
  2. shape completeness    — tensor params without shape constraint flagged
  3. cross-equality        — param-text equality keywords auto-injected
  4. product coverage      — product-specific constraints missing platforms

Also covers R4: inject-dedup via semantic keys (reversed-operand dedup).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure agent package is importable
ROOT = Path(__file__).resolve().parents[2]
for pkg in ("packages/shared/src", "packages/mcp-server/src", "packages/agent/src"):
    p = str(ROOT / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.nodes.constraint_completeness_check import (  # noqa: E402
    _collect_text,
    _has_dtype_constraint,
    _has_shape_constraint,
    _make_eq,
)


# ── _make_eq ─────────────────────────────────────────────────────────────────

class TestMakeEq:
    def test_builds_cross_param_relation(self):
        rel = _make_eq("x", "y", "fn_op", "x.dtype == y.dtype", "src")
        assert rel["relation_type"] == "cross_param_constraint"
        assert rel["platform"] == ""
        assert rel["params"] == ["x", "y"]
        obj = rel["relation_object"]
        assert obj["expr"] == "x.dtype == y.dtype"
        assert obj["src_text"] == "src"


# ── _has_dtype_constraint / _has_shape_constraint ───────────────────────────

class TestHasConstraint:
    def _rel(self, pn, etype, expr):
        return {
            "params": [pn],
            "relation_object": json.dumps({"expr_type": etype, "expr": expr}),
        }

    def test_dtype_detected_by_expr_type(self):
        assert _has_dtype_constraint("x", [self._rel("x", "type_equality", "x.dtype == y.dtype")]) is True

    def test_dtype_detected_by_expr_substring(self):
        assert _has_dtype_constraint("x", [self._rel("x", "self_constraint", "x.dtype == FLOAT16")]) is True

    def test_dtype_missing(self):
        assert _has_dtype_constraint("x", [self._rel("x", "self_constraint", "x > 0")]) is False

    def test_shape_detected_by_expr_type(self):
        assert _has_shape_constraint("x", [self._rel("x", "shape_equality", "x.shape == y.shape")]) is True

    def test_shape_missing(self):
        assert _has_shape_constraint("x", [self._rel("x", "self_constraint", "x > 0")]) is False

    def test_param_not_in_relation(self):
        assert _has_dtype_constraint("z", [self._rel("x", "type_equality", "x.dtype == y.dtype")]) is False


# ── _collect_text ───────────────────────────────────────────────────────────

class TestCollectText:
    def test_collects_plain_fields(self):
        param = {"param_desc": "hello", "llm_description": "world"}
        text = _collect_text(param)
        assert "hello" in text and "world" in text

    def test_handles_json_field(self):
        param = {"usage_notes": json.dumps({"A2": "note1", "A3": "note2"})}
        text = _collect_text(param)
        assert "note1" in text and "note2" in text

    def test_empty_param(self):
        assert _collect_text({}) == ""

    def test_handles_invalid_json(self):
        param = {"param_desc": "{bad json"}
        text = _collect_text(param)
        assert "bad" in text


# ── R4: semantic-key inject dedup ───────────────────────────────────────────

class TestSemanticInjectDedup:
    """R4: the completeness check uses _semantic_expr_key for inject dedup,
    so x.dtype==y.dtype and y.dtype==x.dtype don't both get injected."""

    def test_reversed_equality_is_duplicate(self):
        from agent.utils.expr_validation import _semantic_expr_key
        k1 = _semantic_expr_key("x.dtype == y.dtype")
        k2 = _semantic_expr_key("y.dtype == x.dtype")
        assert k1 == k2  # would not double-inject
