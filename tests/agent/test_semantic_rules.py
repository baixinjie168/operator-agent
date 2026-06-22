"""Tests for agent.utils.semantic_rules (YAML semantic value range inference)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/agent/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/shared/src"))

from agent.utils.semantic_rules import (
    load_rules,
    match_rules,
    get_allowed_range_for_scalar,
    get_expr_for_tensor,
    build_prompt_context,
)


class TestLoadRules:
    def test_loads_non_empty_list(self):
        rules = load_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_each_rule_has_required_fields(self):
        rules = load_rules()
        required = {"id", "keywords", "description", "target"}
        for rule in rules:
            assert isinstance(rule, dict)
            missing = required - set(rule.keys())
            assert not missing, f"Rule {rule.get('id', '?')} missing: {missing}"


class TestMatchRules:
    def test_match_positive_count(self):
        matched = match_rules("代表各专家的token数", "expertTokensOptional")
        assert len(matched) >= 1
        assert matched[0]["id"] == "positive_count"

    def test_match_probability(self):
        matched = match_rules("dropout概率", "dropout_rate")
        assert len(matched) >= 1
        assert matched[0]["id"] == "probability_range"

    def test_match_index(self):
        matched = match_rules("轴的索引", "axis_index")
        assert len(matched) >= 1
        assert matched[0]["id"] == "non_negative_index"

    def test_no_match_for_offset(self):
        matched = match_rules("偏移量", "axis_offset")
        # offset rule has target "none", so it matches but is filtered later
        if matched:
            assert matched[0]["target"] == "none"

    def test_empty_description_returns_empty(self):
        matched = match_rules("", "")
        assert matched == []

    def test_case_insensitive(self):
        matched = match_rules("Count of tokens", "")
        assert len(matched) >= 1


class TestGetAllowedRangeForScalar:
    def test_positive_count(self):
        ar = get_allowed_range_for_scalar("总专家数", "numExperts")
        assert ar == [[1, None]]

    def test_probability(self):
        ar = get_allowed_range_for_scalar("概率值", "dropout_rate")
        assert ar == [[0, 1]]

    def test_index(self):
        ar = get_allowed_range_for_scalar("索引值", "axis")
        assert ar == [[0, None]]

    def test_no_match_returns_none(self):
        ar = get_allowed_range_for_scalar("未知参数", "unknown_param_xyz")
        assert ar is None


class TestGetExprForTensor:
    def test_positive_count_tensor(self):
        result = get_expr_for_tensor("代表各专家的token数", "expertTokensOptional")
        assert result is not None
        assert result["expr_type"] == "self_value_range"
        assert "expertTokensOptional" in result["expr"]
        assert ">= 1" in result["expr"]

    def test_empty_param_name_returns_none(self):
        result = get_expr_for_tensor("token数", "")
        assert result is None

    def test_no_match_returns_none(self):
        result = get_expr_for_tensor("未知描述", "unknown_param_xyz")
        assert result is None


class TestBuildPromptContext:
    def test_returns_non_empty_string(self):
        ctx = build_prompt_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_contains_key_sections(self):
        ctx = build_prompt_context()
        assert "语义推断参考规则" in ctx
        assert "allowed_range_value" in ctx
