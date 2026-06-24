"""Tests for single_param_constraint.py: Layer 1 rules, dedup, coverage."""

import pytest

from agent.nodes.single_param_constraint import (
    RULES,
    SingleParamRule,
    _dedup,
    _extract_numeric_group,
    _is_already_covered,
    _match_rules,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_param(
    name="x1",
    ptype="aclTensorList *",
    fn="aclnnFooGetWorkspaceSize",
    desc="",
    llm_desc="",
    src="",
):
    return {
        "param_name": name,
        "param_type": ptype,
        "function_name": fn,
        "param_desc": desc,
        "llm_description": llm_desc,
        "src_content": src,
    }


# ---------------------------------------------------------------------------
# SingleParamRule
# ---------------------------------------------------------------------------


class TestSingleParamRule:
    def test_compiled_pattern(self):
        rule = SingleParamRule(
            pattern=r"hello\s+world",
            expr_template="x",
            expr_type="test",
            description_template="test",
        )
        assert rule.search("hello  world") is not None
        assert rule.search("helloworld") is None

    def test_param_type_filter_default_empty(self):
        rule = SingleParamRule(
            pattern="x", expr_template="x",
            expr_type="t", description_template="d",
        )
        assert rule.param_type_filter == ""


# ---------------------------------------------------------------------------
# Layer 1: Rule matching
# ---------------------------------------------------------------------------


class TestMatchRules:
    def test_empty_tensor_single_tensor(self):
        param = _make_param(
            name="x", ptype="aclTensor *",
            desc="不支持空Tensor。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        r = results[0]
        assert r["relation_type"] == "self_constraint"
        assert r["params"] == ["x"]
        assert r["relation_object"]["expr_type"] == "self_shape_nonempty"
        assert "all(d > 0 for d in x.shape)" == r["relation_object"]["expr"]

    def test_empty_tensor_tensorlist(self):
        param = _make_param(
            name="x1", ptype="aclTensorList *",
            desc="不支持空Tensor。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        assert results[0]["relation_object"]["expr"] == (
            "all(d > 0 for d in x1.shape)"
        )

    def test_support_empty_tensor_not_extracted(self):
        """'支持空Tensor' should NOT generate a constraint."""
        param = _make_param(
            name="out", ptype="aclTensor *",
            desc="支持空Tensor。",
        )
        results = _match_rules(param, [])
        assert len(results) == 0

    def test_dtype_consistency_tensorlist(self):
        param = _make_param(
            name="x1", ptype="aclTensorList *",
            desc="该参数中所有Tensor的数据类型保持一致。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        expr = results[0]["relation_object"]["expr"]
        assert "x1[i].dtype == x1[0].dtype" in expr

    def test_dtype_consistency_skipped_for_single_tensor(self):
        """dtype consistency rule should only apply to TensorList."""
        param = _make_param(
            name="x", ptype="aclTensor *",
            desc="该参数中所有Tensor的数据类型保持一致。",
        )
        results = _match_rules(param, [])
        assert len(results) == 0

    def test_format_consistency(self):
        param = _make_param(
            name="x2", ptype="aclTensorList *",
            desc="该参数中所有Tensor的数据格式保持一致。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        assert results[0]["relation_object"]["expr_type"] == (
            "self_format_consistency"
        )

    def test_shape_consistency(self):
        param = _make_param(
            name="out", ptype="aclTensorList *",
            desc="该参数中所有Tensor的shape保持一致。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        assert "out[i].shape == out[0].shape" in (
            results[0]["relation_object"]["expr"]
        )

    def test_shape_upper_bound_from_error_code(self):
        param = _make_param(
            name="x", ptype="aclTensor *",
            desc="x1、x2或out中的Tensor维度超过8维。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        expr = results[0]["relation_object"]["expr"]
        assert "len(x.shape) <= 8" == expr

    def test_shape_upper_bound_from_shape_support(self):
        param = _make_param(
            name="x", ptype="aclTensor *",
            desc="shape支持0-8维。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        assert "len(x.shape) <= 8" == results[0]["relation_object"]["expr"]

    def test_multiple_rules_matched(self):
        """A param can match multiple rules (e.g. empty + consistency)."""
        param = _make_param(
            name="x1", ptype="aclTensorList *",
            desc="不支持空Tensor。该参数中所有Tensor的数据类型保持一致。",
        )
        results = _match_rules(param, [])
        expr_types = {r["relation_object"]["expr_type"] for r in results}
        assert "self_shape_nonempty" in expr_types
        assert "self_dtype_consistency" in expr_types

    def test_no_text_returns_empty(self):
        param = _make_param(name="x", desc="", llm_desc="", src="")
        assert _match_rules(param, []) == []

    def test_source_citation_is_matched_text(self):
        param = _make_param(
            name="x", ptype="aclTensor *",
            desc="该参数不支持空Tensor。",
        )
        results = _match_rules(param, [])
        assert len(results) == 1
        assert "不支持空Tensor" in results[0]["source_citation"]


# ---------------------------------------------------------------------------
# Dedup against existing relations
# ---------------------------------------------------------------------------


class TestIsAlreadyCovered:
    def test_not_covered(self):
        rule = RULES[0]  # self_shape_nonempty
        assert not _is_already_covered("x", rule, [])

    def test_covered_by_multiParam(self):
        rule = RULES[0]  # self_shape_nonempty
        existing = [{
            "params": ["x", "y"],
            "relation_object": {
                "expr": "all(d > 0 for d in x.shape) and x.dtype == y.dtype",
            },
        }]
        assert _is_already_covered("x", rule, existing)

    def test_not_covered_other_param(self):
        rule = RULES[0]
        existing = [{
            "params": ["y"],
            "relation_object": {
                "expr": "all(d > 0 for d in y.shape)",
            },
        }]
        assert not _is_already_covered("x", rule, existing)


# ---------------------------------------------------------------------------
# Dedup logic
# ---------------------------------------------------------------------------


class TestDedup:
    def test_no_duplicates(self):
        rels = [
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["x"],
             "relation_object": {"expr_type": "self_shape_nonempty"}},
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["y"],
             "relation_object": {"expr_type": "self_shape_nonempty"}},
        ]
        assert len(_dedup(rels)) == 2

    def test_dedup_same_key(self):
        rels = [
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["x"], "_source": "llm_long_tail",
             "relation_object": {"expr_type": "self_shape_nonempty"}},
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["x"], "_source": "deterministic",
             "relation_object": {"expr_type": "self_shape_nonempty"}},
        ]
        result = _dedup(rels)
        assert len(result) == 1
        assert result[0]["_source"] == "deterministic"

    def test_deterministic_wins_over_llm(self):
        rels = [
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["x"], "_source": "deterministic",
             "relation_object": {"expr_type": "a"}},
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["x"], "_source": "llm_long_tail",
             "relation_object": {"expr_type": "a"}},
        ]
        result = _dedup(rels)
        assert len(result) == 1
        assert result[0]["_source"] == "deterministic"

    def test_dedup_different_expr_type_keeps_both(self):
        rels = [
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["x"],
             "relation_object": {"expr_type": "self_shape_nonempty"}},
            {"function_name": "f", "relation_type": "self_constraint",
             "params": ["x"],
             "relation_object": {"expr_type": "self_dtype_consistency"}},
        ]
        assert len(_dedup(rels)) == 2


# ---------------------------------------------------------------------------
# Numeric group extraction
# ---------------------------------------------------------------------------


class TestExtractNumericGroup:
    def test_single_group(self):
        import re
        m = re.match(r"超过(\d+)维", "超过8维")
        assert _extract_numeric_group(m) == "8"

    def test_multiple_groups_first_non_none(self):
        import re
        m = re.match(r"(a)|(\d+)", "42")
        assert _extract_numeric_group(m) == "42"

    def test_no_groups(self):
        import re
        m = re.match(r"hello", "hello")
        assert _extract_numeric_group(m) == ""

