"""Tests for coverage.py: find_uncovered_params + find_uncovered_context_mentions."""

import pytest

from agent.nodes.param_relation_extract.coverage import (
    find_uncovered_context_mentions,
    find_uncovered_params,
)


class TestFindUncoveredParams:
    def test_all_covered(self):
        params = ["x", "y", "z"]
        relations = [
            {"params": ["x", "y"]},
            {"params": ["y", "z"]},
        ]
        assert find_uncovered_params(params, relations) == []

    def test_partial_coverage(self):
        params = ["x", "y", "z", "w"]
        relations = [{"params": ["x", "y"]}]
        result = find_uncovered_params(params, relations)
        assert set(result) == {"z", "w"}

    def test_no_relations(self):
        params = ["x", "y"]
        assert find_uncovered_params(params, []) == ["x", "y"]

    def test_empty_params(self):
        relations = [{"params": ["x", "y"]}]
        assert find_uncovered_params([], relations) == []


class TestFindUncoveredContextMentions:
    def test_uncovered_paragraph_with_two_params(self):
        content = "x 和 y 之间存在 shape 约束关系，当 x 的维度为 2 时 y 必须为 3 维"
        params = ["x", "y"]
        relations = []
        result = find_uncovered_context_mentions(content, params, relations)
        assert len(result) == 1

    def test_covered_by_source_citation(self):
        content = "x 和 y 之间存在 shape 约束关系"
        params = ["x", "y"]
        relations = [
            {"params": ["x", "y"], "source_citation": "x 和 y 之间存在 shape 约束关系"},
        ]
        result = find_uncovered_context_mentions(content, params, relations)
        assert len(result) == 0

    def test_word_boundary_prevents_false_match(self):
        # "x" should not match "axis"
        content = "axis 和 scale 之间有约束"
        params = ["x", "scale"]
        relations = []
        result = find_uncovered_context_mentions(content, params, relations)
        # "x" is not mentioned (only "axis"), so only 1 param mentioned → not uncovered
        assert len(result) == 0

    def test_short_paragraphs_ignored(self):
        content = "x y"  # Too short (<20 chars)
        params = ["x", "y"]
        result = find_uncovered_context_mentions(content, params, [])
        assert len(result) == 0

    def test_single_param_paragraph_ignored(self):
        content = "x 是一个非常重要的参数，它的取值范围很广泛"
        params = ["x"]
        result = find_uncovered_context_mentions(content, params, [])
        assert len(result) == 0

    def test_fingerprint_matching(self):
        content = "x and y need broadcast relation and same dtype"
        params = ["x", "y"]
        # Source citation that contains the paragraph as substring
        relations = [{
            "params": ["x", "y"],
            "source_citation": "x and y need broadcast relation and same dtype indeed",
        }]
        result = find_uncovered_context_mentions(content, params, relations)
        assert len(result) == 0
