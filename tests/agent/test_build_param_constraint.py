"""Tests for build_param_constraint.py: dimensions parsing with symbolic shapes.

Covers:
- _parse_dimensions_response: rank spec [N,N] vs per-dimension [[min,max],...]
- _try_deterministic_parse: regex-based shape parsing
- _validate_dimensions_structure: dual format validation
- _validate_dimensions_alignment: alignment validation
- _try_deterministic_range: regex-based range extraction
- _validate_range_structure: min/max/type validation
- _validate_range_source: source text validation
- Edge cases: empty, invalid JSON, markdown code blocks
"""

import pytest

from agent.nodes.build_param_constraint._helpers import _normalize_type
from agent.nodes.build_param_constraint.allowed_range_build import (
    _try_deterministic_range,
    _validate_range_source,
    _validate_range_structure,
)
from agent.nodes.build_param_constraint.attrs_build import (
    _ARRAY_TYPE_DTYPE_FALLBACK,
    attrs_build_node,
)
from agent.nodes.build_param_constraint.fetch_param_data import (
    _resolve_cross_references,
)
from agent.nodes.build_param_constraint.dimensions_agent import (
    _parse_dimensions_response,
    _try_deterministic_parse,
    _validate_dimensions_structure,
)


class TestParseDimensionsResponse:
    """Test the dimensions parser (new enumeration semantics).

    每个返回元素是一维 int 枚举列表；旧 [min,max] 嵌套会被兼容展平 + 去重。
    """

    # ── 平铺 int 列表 → 每个 int 包成单元素 list ──

    def test_rank_spec_simple(self):
        """[5, 5] → [[5], [5]]（每个 int 一个枚举）。"""
        assert _parse_dimensions_response("[5, 5]") == [[5], [5]]

    def test_rank_spec_4d(self):
        """[4, 4] → [[4], [4]]。"""
        assert _parse_dimensions_response("[4, 4]") == [[4], [4]]

    def test_rank_spec_1d(self):
        """[1, 1] → [[1], [1]]。"""
        assert _parse_dimensions_response("[1, 1]") == [[1], [1]]

    def test_rank_spec_in_code_block(self):
        """Handle markdown code block wrapping."""
        assert _parse_dimensions_response("```json\n[5, 5]\n```") == [[5], [5]]

    # ── 嵌套列表 → 兼容展平 + 去重 ──

    def test_per_dim_fixed_values(self):
        """[[2,2],[3,3],[4,4]] → [[2],[3],[4]]（去重）。"""
        assert _parse_dimensions_response("[[2,2],[3,3],[4,4]]") == [[2], [3], [4]]

    def test_per_dim_with_nulls(self):
        """[[null,null],[3,3]] → [[],[3]]（null 跳过 → 空）。"""
        assert _parse_dimensions_response("[[null,null],[3,3]]") == [[], [3]]

    def test_per_dim_range(self):
        """[[1,8]] → [[1,8]]（两 int 保留为枚举 {1,8}）。"""
        assert _parse_dimensions_response("[[1,8]]") == [[1, 8]]

    def test_per_dim_in_code_block(self):
        """Handle markdown code block wrapping."""
        text = "```json\n[[2,2],[3,3],[4,4]]\n```"
        assert _parse_dimensions_response(text) == [[2], [3], [4]]

    # ── Empty / invalid ──

    def test_empty_array(self):
        assert _parse_dimensions_response("[]") == []

    def test_empty_string(self):
        assert _parse_dimensions_response("") == []

    def test_invalid_json(self):
        assert _parse_dimensions_response("not json at all") == []

    # ── Mixed / fallback ──

    def test_non_list_items_become_empty(self):
        """Strings in nested list positions become []."""
        result = _parse_dimensions_response('["bad", [3,3]]')
        assert result == [[], [3]]

    def test_regex_fallback_rank_spec(self):
        """Regex fallback still works for rank spec."""
        text = "Here is the answer: [5, 5] done"
        assert _parse_dimensions_response(text) == [[5], [5]]

    def test_regex_fallback_per_dim(self):
        """Regex fallback still works for per-dimension."""
        text = "Answer: [[2,2],[3,3]]"
        assert _parse_dimensions_response(text) == [[2], [3]]


# ---------------------------------------------------------------------------
# _try_deterministic_parse
# ---------------------------------------------------------------------------


class TestTryDeterministicParse:
    def test_scalar_zh(self):
        assert _try_deterministic_parse("标量") == []

    def test_zero_d(self):
        assert _try_deterministic_parse("0-D") == []
        assert _try_deterministic_parse("0D") == []

    def test_one_d(self):
        assert _try_deterministic_parse("1-D") == [1]
        assert _try_deterministic_parse("1D") == [1]

    def test_n_d(self):
        assert _try_deterministic_parse("2D") == [2]
        assert _try_deterministic_parse("3-D") == [3]
        assert _try_deterministic_parse("5D") == [5]

    def test_rank_range(self):
        """'X-Y' 连续区间展开为枚举列表。"""
        assert _try_deterministic_parse("0-8") == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert _try_deterministic_parse("1-8") == [1, 2, 3, 4, 5, 6, 7, 8]
        assert _try_deterministic_parse("2~5") == [2, 3, 4, 5]
        assert _try_deterministic_parse("1 - 4") == [1, 2, 3, 4]

    def test_parentheses(self):
        # 符号元组按逗号计数 → rank
        assert _try_deterministic_parse("(N,C,H,W)") == [4]
        assert _try_deterministic_parse("(N, D, H, W, C)") == [5]
        assert _try_deterministic_parse("(batch, seq)") == [2]

    def test_brackets(self):
        # 方括号内按逗号计数 → rank
        assert _try_deterministic_parse("[2, 3, 4]") == [3]
        assert _try_deterministic_parse("[8]") == [1]
        assert _try_deterministic_parse("[16, 32]") == [2]

    def test_same_as_input(self):
        assert _try_deterministic_parse("与输入相同") == []
        assert _try_deterministic_parse("same as input") == []

    def test_unknown_patterns(self):
        assert _try_deterministic_parse("variable") is None
        assert _try_deterministic_parse("2D or 3D") is None
        assert _try_deterministic_parse("") is None

    def test_whitespace_handling(self):
        assert _try_deterministic_parse("  2D  ") == [2]
        assert _try_deterministic_parse("  0-8  ") == [0, 1, 2, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# _validate_dimensions_structure
# ---------------------------------------------------------------------------


class TestValidateDimensionsStructure:
    def test_empty_list(self):
        is_valid, error = _validate_dimensions_structure([])
        assert is_valid
        assert error == ""

    def test_rank_format_range(self):
        """枚举格式：升序去重的 int 列表均合法。"""
        is_valid, error = _validate_dimensions_structure([0, 8])
        assert is_valid
        assert error == ""

        is_valid, error = _validate_dimensions_structure([1, 8])
        assert is_valid
        assert error == ""

        is_valid, error = _validate_dimensions_structure([2, 5])
        assert is_valid
        assert error == ""

    def test_single_value_valid(self):
        """[N] 单值枚举合法。"""
        is_valid, error = _validate_dimensions_structure([2])
        assert is_valid
        assert error == ""

    def test_zero_scalar_valid(self):
        """[0] 表示 0 维（标量），合法。"""
        is_valid, error = _validate_dimensions_structure([0])
        assert is_valid
        assert error == ""

    def test_duplicate_rejected(self):
        """重复值非法。"""
        is_valid, error = _validate_dimensions_structure([2, 2])
        assert not is_valid
        assert "sorted" in error or "deduplicated" in error

    def test_not_sorted_rejected(self):
        """非升序非法。"""
        is_valid, error = _validate_dimensions_structure([3, 1])
        assert not is_valid
        assert "sorted" in error or "deduplicated" in error

    def test_negative_value_rejected(self):
        """负数非法。"""
        is_valid, error = _validate_dimensions_structure([-1, 5])
        assert not is_valid
        assert "out of" in error

    def test_value_too_large_rejected(self):
        """超 MAX_DIM 非法。"""
        is_valid, error = _validate_dimensions_structure([0, 11])
        assert not is_valid
        assert "out of" in error

    def test_nested_lists_rejected(self):
        """新枚举语义不接受嵌套 list（旧 per-dim 格式已废弃）。"""
        assert not _validate_dimensions_structure([[1, 8], [3, 3], [4, 4]])[0]
        assert not _validate_dimensions_structure([[None, None], [3, 5]])[0]
        assert not _validate_dimensions_structure([[5, 3]])[0]
        assert not _validate_dimensions_structure([[1, 2, 3]])[0]
        assert not _validate_dimensions_structure([["a", "b"]])[0]
        # 多个嵌套元素：第一个即被拒
        dims = [[i, i] for i in range(11)]
        assert not _validate_dimensions_structure(dims)[0]

    def test_not_a_list(self):
        is_valid, error = _validate_dimensions_structure("not a list")
        assert not is_valid
        assert "must be a list" in error


# ---------------------------------------------------------------------------
# _try_deterministic_range
# ---------------------------------------------------------------------------


class TestTryDeterministicRange:
    """_try_deterministic_range 现返回 (value_list, ar_type) 元组。"""

    def test_bracket_range(self):
        assert _try_deterministic_range("[0, 100]") == ([[0, 100]], "range")
        assert _try_deterministic_range("[1, 8]") == ([[1, 8]], "range")

    def test_dash_not_recognized(self):
        # 裸 dash（无括号/无 ~到至）不再识别为 range
        assert _try_deterministic_range("0-100") is None
        assert _try_deterministic_range("1-1024") is None

    def test_tilde_range(self):
        assert _try_deterministic_range("0~100") == ([[0, 100]], "range")

    def test_comma_not_recognized(self):
        # 裸逗号（无括号）不是 range
        assert _try_deterministic_range("0, 100") is None

    def test_enum_values(self):
        result = _try_deterministic_range("枚举值: 1, 2, 3")
        assert result == ([[1, 1], [2, 2], [3, 3]], "enum")

    def test_no_match(self):
        assert _try_deterministic_range("无限制") is None
        assert _try_deterministic_range("") is None

    def test_negative_numbers(self):
        assert _try_deterministic_range("[-1, 1]") == ([[-1, 1]], "range")


# ---------------------------------------------------------------------------
# _validate_range_structure
# ---------------------------------------------------------------------------


class TestValidateRangeStructure:
    def test_empty_list(self):
        is_valid, error = _validate_range_structure([])
        assert is_valid
        assert error == ""

    def test_valid_range(self):
        is_valid, error = _validate_range_structure([[0, 100]])
        assert is_valid
        assert error == ""

    def test_valid_range_with_null(self):
        is_valid, error = _validate_range_structure([[1, None]])
        assert is_valid
        assert error == ""

    def test_valid_multiple_ranges(self):
        is_valid, error = _validate_range_structure([[0, 10], [20, 30]])
        assert is_valid
        assert error == ""

    def test_min_greater_than_max(self):
        is_valid, error = _validate_range_structure([[100, 0]])
        assert not is_valid
        assert "min (100) > max (0)" in error

    def test_wrong_length(self):
        is_valid, error = _validate_range_structure([[1, 2, 3]])
        assert not is_valid
        assert "must be [min, max]" in error

    def test_wrong_type(self):
        is_valid, error = _validate_range_structure([["a", "b"]])
        assert not is_valid
        assert "must be int/float or null" in error

    def test_unsigned_negative(self):
        is_valid, error = _validate_range_structure([[-1, 10]], param_type="uint64_t")
        assert not is_valid
        assert "negative value" in error

    def test_unsigned_positive(self):
        is_valid, error = _validate_range_structure([[0, 10]], param_type="uint64_t")
        assert is_valid
        assert error == ""

    def test_unreasonably_large(self):
        is_valid, error = _validate_range_structure([[0, 1e10]])
        assert not is_valid
        assert "unreasonably large" in error

    def test_not_a_list(self):
        is_valid, error = _validate_range_structure("not a list")
        assert not is_valid
        assert "must be a list" in error


# ---------------------------------------------------------------------------
# _validate_range_source
# ---------------------------------------------------------------------------


class TestValidateRangeSource:
    def test_empty_range(self):
        is_valid, error = _validate_range_source([], "some text")
        assert is_valid
        assert error == ""

    def test_values_in_source(self):
        is_valid, error = _validate_range_source([[0, 100]], "范围0到100")
        assert is_valid
        assert error == ""

    def test_values_not_in_source(self):
        # This is a soft check, should still pass
        is_valid, error = _validate_range_source([[0, 100]], "无限制")
        assert is_valid
        assert error == ""

    def test_null_values(self):
        is_valid, error = _validate_range_source([[1, None]], "大于1")
        assert is_valid
        assert error == ""


class TestNormalizeType:
    """Test the type normalization helper for param_constraint.type.value."""

    def test_strips_const_modifier(self):
        assert _normalize_type("const aclTensor") == "aclTensor"

    def test_strips_pointer_asterisk(self):
        assert _normalize_type("aclTensor*") == "aclTensor"

    def test_strips_const_and_pointer(self):
        assert _normalize_type("const aclTensor *") == "aclTensor"

    def test_strips_reference_operator(self):
        assert _normalize_type("int&") == "int"

    def test_strips_all_modifiers(self):
        assert _normalize_type("const int * &") == "int"

    def test_preserves_clean_type(self):
        assert _normalize_type("uint64_t") == "uint64_t"

    def test_handles_empty_string(self):
        assert _normalize_type("") == ""

    def test_handles_extra_whitespace(self):
        assert _normalize_type("  const  aclTensor  ") == "aclTensor"

    def test_const_in_middle_of_type_name(self):
        # Should not strip "const" if it's part of a larger word
        assert _normalize_type("const_ptr") == "const_ptr"


# ---------------------------------------------------------------------------
# attrs_build_node: Level-3 dtype fallback (type -> dtype)
# ---------------------------------------------------------------------------


def _make_state(params, sig_type_map, *, dtype_by_platform=None, platforms=None):
    """Build minimal BuildParamConstraintState for attrs_build_node tests."""
    return {
        "params": params,
        "sig_type_map": sig_type_map,
        "all_sig_param_names": [p["param_name"] for p in params],
        "dtype_by_platform": dtype_by_platform or {},
        "supported_platforms": platforms or ["common"],
    }


def _get_dtype_value(attrs_map, fn, pn, plat="common"):
    """Extract dtype.value from attrs_map for a given param+platform."""
    key = f"{fn}::{pn}::{plat}"
    return attrs_map[key]["dtype"]["value"]


class TestDtypeFallbackMapping:
    """Test the _ARRAY_TYPE_DTYPE_FALLBACK constant."""

    def test_acl_int_array(self):
        assert _ARRAY_TYPE_DTYPE_FALLBACK["aclIntArray"] == "int"

    def test_acl_float_array(self):
        assert _ARRAY_TYPE_DTYPE_FALLBACK["aclFloatArray"] == "float"

    def test_acl_bool_array(self):
        assert _ARRAY_TYPE_DTYPE_FALLBACK["aclBoolArray"] == "bool"

    def test_acl_tensor_not_in_map(self):
        # aclTensor falls through to the default (type name itself)
        assert "aclTensor" not in _ARRAY_TYPE_DTYPE_FALLBACK

    def test_acl_scalar_not_in_map(self):
        assert "aclScalar" not in _ARRAY_TYPE_DTYPE_FALLBACK


class TestAttrsBuildDtypeLevel3:
    """Test Level-3 dtype fallback in attrs_build_node."""

    @pytest.mark.asyncio
    async def test_acl_int_array_empty_dtype(self):
        """aclIntArray with no dtype from Level 1/2 → ['int']."""
        params = [{"param_name": "pads", "function_name": "FusionOp"}]
        state = _make_state(params, {"FusionOp::pads": "aclIntArray"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "pads") == ["int"]

    @pytest.mark.asyncio
    async def test_acl_float_array_empty_dtype(self):
        """aclFloatArray with no dtype from Level 1/2 → ['float']."""
        params = [{"param_name": "scales", "function_name": "FusionOp"}]
        state = _make_state(params, {"FusionOp::scales": "aclFloatArray"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "scales") == ["float"]

    @pytest.mark.asyncio
    async def test_acl_bool_array_empty_dtype(self):
        """aclBoolArray with no dtype from Level 1/2 → ['bool']."""
        params = [{"param_name": "flags", "function_name": "FusionOp"}]
        state = _make_state(params, {"FusionOp::flags": "aclBoolArray"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "flags") == ["bool"]

    @pytest.mark.asyncio
    async def test_acl_tensor_empty_dtype_falls_back_to_type_name(self):
        """aclTensor with no dtype → [] (tensor type name is not a valid dtype)."""
        params = [{"param_name": "x", "function_name": "FusionOp"}]
        state = _make_state(params, {"FusionOp::x": "aclTensor"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "x") == []

    @pytest.mark.asyncio
    async def test_acl_scalar_empty_dtype_falls_back_to_type_name(self):
        """aclScalar with no dtype → ['aclScalar'] (default: type name itself)."""
        params = [{"param_name": "alpha", "function_name": "FusionOp"}]
        state = _make_state(params, {"FusionOp::alpha": "aclScalar"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "alpha") == ["aclScalar"]

    @pytest.mark.asyncio
    async def test_level1_takes_precedence_over_level3(self):
        """When dtype_desc has a value, Level 3 is not triggered."""
        params = [{
            "param_name": "pads",
            "function_name": "FusionOp",
            "dtype_desc": '{"*": "INT64"}',
        }]
        state = _make_state(params, {"FusionOp::pads": "aclIntArray"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "pads") == ["INT64"]

    @pytest.mark.asyncio
    async def test_level2_takes_precedence_over_level3(self):
        """When dtype_combinations has a value, Level 3 is not triggered."""
        params = [{"param_name": "pads", "function_name": "FusionOp"}]
        dtype_by_platform = {"common": {"pads": ["INT32", "INT64"]}}
        state = _make_state(
            params, {"FusionOp::pads": "aclIntArray"},
            dtype_by_platform=dtype_by_platform,
        )
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "pads") == ["INT32", "INT64"]

    @pytest.mark.asyncio
    async def test_null_pointer_keeps_empty_dtype(self):
        """Null-pointer-only param keeps dtype=[]; Level 3 skipped."""
        params = [{
            "param_name": "reserved",
            "function_name": "FusionOp",
            "usage_notes": '{"*": "仅支持空指针"}',
        }]
        state = _make_state(params, {"FusionOp::reserved": "aclIntArray"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "reserved") == []

    @pytest.mark.asyncio
    async def test_empty_type_keeps_empty_dtype(self):
        """When ptype itself is empty, Level 3 cannot fill → stays []."""
        params = [{"param_name": "unknown", "function_name": "FusionOp"}]
        state = _make_state(params, {})  # no sig_type_map entry, no param_type
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "unknown") == []

    @pytest.mark.asyncio
    async def test_const_pointer_normalized_before_fallback(self):
        """const aclIntArray* is normalized to aclIntArray, then mapped to 'int'."""
        params = [{"param_name": "pads", "function_name": "FusionOp"}]
        state = _make_state(params, {"FusionOp::pads": "const aclIntArray*"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "pads") == ["int"]

    @pytest.mark.asyncio
    async def test_multi_platform_independent_fallback(self):
        """Level 3 applies per-platform: one platform has dtype, another doesn't."""
        params = [{
            "param_name": "pads",
            "function_name": "FusionOp",
            "dtype_desc": '{"Atlas A2": "INT64"}',  # only A2 has dtype
        }]
        state = _make_state(
            params, {"FusionOp::pads": "aclIntArray"},
            platforms=["Atlas A2", "Atlas A3"],
        )
        result = await attrs_build_node(state)
        # Atlas A2: Level 1 provides INT64
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "pads", "Atlas A2") == ["INT64"]
        # Atlas A3: Level 1 empty, Level 2 empty → Level 3 fills 'int'
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "pads", "Atlas A3") == ["int"]

    @pytest.mark.asyncio
    async def test_bu_zhi_chi_not_treated_as_null_pointer(self):
        """usage_notes containing '不支持' (but not null-pointer) → Level 3 fills dtype.

        Regression test: the broad '不支持' pattern was removed from
        _NULL_POINTER_RE because it matched non-null-pointer contexts like
        '暂不支持设为True'.  Such params should still get their dtype via
        the Level-3 type-name fallback.
        """
        params = [{
            "param_name": "transposeX1",
            "function_name": "FusionOp",
            "usage_notes": '{"*": "暂不支持设为True。"}',
        }]
        state = _make_state(params, {"FusionOp::transposeX1": "bool"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "transposeX1") == ["bool"]

    @pytest.mark.asyncio
    async def test_explicit_null_pointer_still_keeps_empty_dtype(self):
        """Explicit null-pointer phrases ('只支持传空指针') → dtype stays []."""
        params = [{
            "param_name": "bias1Optional",
            "function_name": "FusionOp",
            "usage_notes": '{"*": "Atlas 推理系列加速卡产品：只支持传空指针。"}',
        }]
        state = _make_state(params, {"FusionOp::bias1Optional": "aclTensor"})
        result = await attrs_build_node(state)
        assert _get_dtype_value(result["attrs_map"], "FusionOp", "bias1Optional") == []


class TestConstraintBasedDtypeResolution:
    """Verify _resolve_cross_references uses type_equality constraints.

    Regression test for dead code: consistency_pairs was built from
    param_relations but never used.  Now constraint-based pairs supplement
    text-based detection to resolve params whose dtype_desc is empty.
    """

    def test_dtype_equality_constraint_resolves_empty_dtype(self):
        """gradOutput.dtype == self.dtype → copy self's dtype to gradOutput."""
        params = [
            {
                "param_name": "self",
                "function_name": "Op",
                "dtype_desc": '{"*": "BFLOAT16、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128"}',
            },
            {
                "param_name": "gradOutput",
                "function_name": "Op",
                # dtype_desc is empty — no cross-reference text either
            },
        ]
        relations = [
            {
                "function_name": "Op",
                "relation_object": '{"expr_type": "type_equality", "expr": "gradOutput.dtype == self.dtype", "relation_params": ["gradOutput", "self"], "src_text": ""}',
            },
        ]
        result = _resolve_cross_references(params, relations)
        index = {p["param_name"]: p for p in result}
        # gradOutput should now have self's dtype
        assert index["gradOutput"]["dtype_desc"] == index["self"]["dtype_desc"]

    def test_no_constraints_no_change(self):
        """Without any relations, params are returned as-is (shallow copy)."""
        params = [
            {"param_name": "x", "function_name": "Op", "dtype_desc": '{"*": "FLOAT16"}'},
            {"param_name": "y", "function_name": "Op"},
        ]
        result = _resolve_cross_references(params, [])
        assert len(result) == 2
        index = {p["param_name"]: p for p in result}
        assert index["x"]["dtype_desc"] == '{"*": "FLOAT16"}'
        assert index["y"].get("dtype_desc", "") == ""

    def test_chain_resolution_a_to_b_to_c(self):
        """Chained constraints A.dtype == B.dtype == C.dtype → all resolved."""
        params = [
            {"param_name": "a", "function_name": "Op", "dtype_desc": '{"*": "FLOAT16"}'},
            {"param_name": "b", "function_name": "Op"},
            {"param_name": "c", "function_name": "Op"},
        ]
        relations = [
            {
                "function_name": "Op",
                "relation_object": '{"expr": "b.dtype == a.dtype", "relation_params": ["b", "a"]}',
            },
            {
                "function_name": "Op",
                "relation_object": '{"expr": "c.dtype == b.dtype", "relation_params": ["c", "b"]}',
            },
        ]
        result = _resolve_cross_references(params, relations)
        index = {p["param_name"]: p for p in result}
        # All three should have the same dtype
        assert index["a"]["dtype_desc"] == '{"*": "FLOAT16"}'
        assert index["b"]["dtype_desc"] == '{"*": "FLOAT16"}'
        assert index["c"]["dtype_desc"] == '{"*": "FLOAT16"}'
