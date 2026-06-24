"""Tests for dimensions_agent: DeepAgent-based shape->dimensions conversion.

Covers:
- _try_deterministic_parse: regex-based shape parsing (reused from dimensions_build)
- _validate_dimensions_structure: dual format validation (rank + per-dimension)
- _parse_dimensions_response: JSON parsing from agent text output
- dimensions_agent_node: full node with deterministic + agent fallback
"""

import json
from typing import Any

import pytest

from agent.nodes.build_param_constraint.dimensions_agent import (
    _is_rank_format,
    _parse_dimensions_response,
    _try_deterministic_parse,
    _try_html_list_parse,
    _validate_dimensions_structure,
    dimensions_agent_node,
)


# ---------------------------------------------------------------------------
# _try_deterministic_parse
# ---------------------------------------------------------------------------


class TestTryDeterministicParse:
    """Test deterministic regex parsing."""

    def test_scalar(self):
        assert _try_deterministic_parse("标量") == []

    def test_zero_d(self):
        assert _try_deterministic_parse("0-D") == []
        assert _try_deterministic_parse("0D") == []

    def test_rank_range(self):
        assert _try_deterministic_parse("0-8") == [0, 8]
        assert _try_deterministic_parse("1-4") == [1, 4]
        assert _try_deterministic_parse("2~6") == [2, 6]

    def test_rank_exact(self):
        assert _try_deterministic_parse("2D") == [2, 2]
        assert _try_deterministic_parse("3-D") == [3, 3]
        assert _try_deterministic_parse("4D") == [4, 4]

    def test_symbolic_tuple(self):
        assert _try_deterministic_parse("(N,C,H,W)") == [4, 4]
        assert _try_deterministic_parse("(B, S, H)") == [3, 3]

    def test_numeric_array(self):
        assert _try_deterministic_parse("[2, 3, 4]") == [[2, 2], [3, 3], [4, 4]]
        assert _try_deterministic_parse("[1, 256]") == [[1, 1], [256, 256]]

    def test_same_as_input(self):
        assert _try_deterministic_parse("与输入相同") == []
        assert _try_deterministic_parse("same as input") == []

    def test_no_match(self):
        assert _try_deterministic_parse("1D~8D") is None
        assert _try_deterministic_parse("some weird shape") is None

    def test_empty_string(self):
        assert _try_deterministic_parse("") is None


# ---------------------------------------------------------------------------
# _validate_dimensions_structure
# ---------------------------------------------------------------------------


class TestValidateDimensionsStructure:
    """Test dimensions structure validation."""

    def test_empty_valid(self):
        is_valid, _ = _validate_dimensions_structure([])
        assert is_valid

    def test_rank_valid(self):
        assert _validate_dimensions_structure([4, 4])[0]
        assert _validate_dimensions_structure([0, 8])[0]
        assert _validate_dimensions_structure([1, 1])[0]

    def test_rank_invalid_min_gt_max(self):
        is_valid, msg = _validate_dimensions_structure([5, 3])
        assert not is_valid
        assert "min <= max" in msg

    def test_rank_invalid_negative(self):
        is_valid, _ = _validate_dimensions_structure([-1, 5])
        assert not is_valid

    def test_rank_invalid_too_many(self):
        is_valid, _ = _validate_dimensions_structure([0, 11])
        assert not is_valid

    def test_per_dimension_valid(self):
        assert _validate_dimensions_structure([[2, 2], [3, 3]])[0]
        assert _validate_dimensions_structure([[1, None]])[0]

    def test_per_dimension_invalid_min_gt_max(self):
        is_valid, _ = _validate_dimensions_structure([[5, 3]])
        assert not is_valid

    def test_not_a_list(self):
        is_valid, _ = _validate_dimensions_structure("not a list")
        assert not is_valid


# ---------------------------------------------------------------------------
# _is_rank_format
# ---------------------------------------------------------------------------


class TestIsRankFormat:

    def test_rank(self):
        assert _is_rank_format([4, 4])
        assert _is_rank_format([0, 8])

    def test_not_rank(self):
        assert not _is_rank_format([])
        assert not _is_rank_format([4])
        assert not _is_rank_format([4, 4, 4])
        assert not _is_rank_format([[2, 2]])


# ---------------------------------------------------------------------------
# _parse_dimensions_response
# ---------------------------------------------------------------------------


class TestParseDimensionsResponse:

    def test_plain_json(self):
        result = _parse_dimensions_response("[[4, 4], [0, 8]]")
        assert result == [[4, 4], [0, 8]]

    def test_code_block(self):
        text = "Results:\n```json\n[[4, 4]]\n```"
        result = _parse_dimensions_response(text)
        assert result == [[4, 4]]

    def test_empty_response(self):
        result = _parse_dimensions_response("")
        assert result == []

    def test_invalid_json(self):
        result = _parse_dimensions_response("not json at all")
        assert result == []

    def test_single_rank(self):
        result = _parse_dimensions_response("[4, 4]")
        assert result == [4, 4]


# ---------------------------------------------------------------------------
# _try_html_list_parse (Phase 1.5)
# ---------------------------------------------------------------------------


class TestTryHtmlListParse:
    """Test deterministic HTML-list shape parsing."""

    def test_per_channel_per_group_n1(self):
        shape = (
            "<ul><li>per-channel下输入在有/无专家时分别为[E, N1]/[N1]</li>"
            "<li>per-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]</li></ul>"
        )
        # brackets: [E,N1]=2, [N1]=1, [E,G,N1]=3, [G,N1]=2 → [1, 3]
        assert _try_html_list_parse(shape) == [1, 3]

    def test_per_channel_per_group_n2(self):
        shape = (
            "<ul><li>per-channel下输入在有/无专家时分别为[E, N2]/[N2]</li>"
            "<li>per-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]</li></ul>"
        )
        assert _try_html_list_parse(shape) == [1, 3]

    def test_per_tensor_per_channel(self):
        shape = (
            "<ul><li>per-tensor下输入在有/无专家时均为一维向量，"
            "输入元素个数在有/无专家时分别为[E]/[1]</li>"
            "<li>per-channel下输入在有/无专家时为二维向量/一维向量，"
            "输入元素个数在有/无专家时分别为[E, N1]/[N1]</li></ul>"
        )
        # brackets: [E]=1, [1]=1, [E,N1]=2, [N1]=1 → [1, 2]
        assert _try_html_list_parse(shape) == [1, 2]

    def test_single_bracket_in_html(self):
        """HTML with a single bracket still works."""
        shape = "<ul><li>shape is [M, K1, N1]</li></ul>"
        assert _try_html_list_parse(shape) == [3, 3]

    def test_no_html_returns_none(self):
        """Non-HTML shapes are not handled here."""
        assert _try_html_list_parse("[M, K1]") is None
        assert _try_html_list_parse("2D") is None
        assert _try_html_list_parse("(N, C, H, W)") is None

    def test_html_no_brackets_returns_none(self):
        """HTML without bracket groups falls through to agent."""
        assert _try_html_list_parse("<ul><li>标量</li></ul>") is None
        assert _try_html_list_parse("<p>no dimensions here</p>") is None

    def test_empty_string(self):
        assert _try_html_list_parse("") is None


# ---------------------------------------------------------------------------
# dimensions_agent_node (deterministic-only tests, no LLM needed)
# ---------------------------------------------------------------------------


class TestDimensionsAgentNodeDeterministic:
    """Test the node with shapes that can be fully handled by deterministic regex."""

    @pytest.mark.asyncio
    async def test_empty_params(self):
        result = await dimensions_agent_node({"params": []})
        assert result == {"dimensions_map": {}}

    @pytest.mark.asyncio
    async def test_no_state(self):
        result = await dimensions_agent_node({})
        assert result == {"dimensions_map": {}}

    @pytest.mark.asyncio
    async def test_all_deterministic(self):
        params = [
            {
                "function_name": "fn1",
                "param_name": "x1",
                "shape": json.dumps({"*": "(N,C,H,W)"}),
            },
            {
                "function_name": "fn1",
                "param_name": "x2",
                "shape": json.dumps({"*": "2D"}),
            },
            {
                "function_name": "fn1",
                "param_name": "x3",
                "shape": json.dumps({"*": "0-8"}),
            },
        ]
        result = await dimensions_agent_node({"params": params})
        dims_map = result["dimensions_map"]
        assert dims_map["fn1::x1::(N,C,H,W)"] == [4, 4]
        assert dims_map["fn1::x2::2D"] == [2, 2]
        assert dims_map["fn1::x3::0-8"] == [0, 8]

    @pytest.mark.asyncio
    async def test_dedup_same_shape(self):
        params = [
            {
                "function_name": "fn1",
                "param_name": "x1",
                "shape": json.dumps({"*": "2D"}),
            },
            {
                "function_name": "fn1",
                "param_name": "x2",
                "shape": json.dumps({"*": "2D"}),
            },
        ]
        result = await dimensions_agent_node({"params": params})
        dims_map = result["dimensions_map"]
        assert dims_map["fn1::x1::2D"] == [2, 2]
        assert dims_map["fn1::x2::2D"] == [2, 2]

    @pytest.mark.asyncio
    async def test_empty_shape_skipped(self):
        params = [
            {"function_name": "fn1", "param_name": "x1", "shape": ""},
            {
                "function_name": "fn1",
                "param_name": "x2",
                "shape": json.dumps({"*": "2D"}),
            },
        ]
        result = await dimensions_agent_node({"params": params})
        dims_map = result["dimensions_map"]
        assert "fn1::x2::2D" in dims_map
        assert len(dims_map) == 1

    @pytest.mark.asyncio
    async def test_platform_specific_shapes(self):
        params = [
            {
                "function_name": "fn1",
                "param_name": "x1",
                "shape": json.dumps({"Atlas A2": "(N,C,H,W)", "Atlas A3": "2D"}),
            },
        ]
        result = await dimensions_agent_node({"params": params})
        dims_map = result["dimensions_map"]
        assert dims_map["fn1::x1::(N,C,H,W)"] == [4, 4]
        assert dims_map["fn1::x1::2D"] == [2, 2]

    @pytest.mark.asyncio
    async def test_scalar_shape(self):
        params = [
            {
                "function_name": "fn1",
                "param_name": "bias",
                "shape": json.dumps({"*": "标量"}),
            },
        ]
        result = await dimensions_agent_node({"params": params})
        dims_map = result["dimensions_map"]
        assert dims_map["fn1::bias::标量"] == []
