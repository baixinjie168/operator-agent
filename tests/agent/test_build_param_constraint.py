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

from agent.nodes.build_param_constraint import (
    _is_rank_format,
    _normalize_type,
    _parse_dimensions_response,
    _try_deterministic_parse,
    _try_deterministic_range,
    _validate_dimensions_alignment,
    _validate_dimensions_structure,
    _validate_range_source,
    _validate_range_structure,
)


class TestParseDimensionsResponse:
    """Test the dimensions parser handles both rank specs and per-dimension ranges."""

    # ── Rank specification (flat int list) ──

    def test_rank_spec_simple(self):
        """grid shape (N,Dout,Hout,Wout,3) → rank 5."""
        assert _parse_dimensions_response("[5, 5]") == [5, 5]

    def test_rank_spec_4d(self):
        """(N,C,H,W) → rank 4."""
        assert _parse_dimensions_response("[4, 4]") == [4, 4]

    def test_rank_spec_1d(self):
        """1-D tensor → rank 1."""
        assert _parse_dimensions_response("[1, 1]") == [1, 1]

    def test_rank_spec_in_code_block(self):
        """Handle markdown code block wrapping."""
        assert _parse_dimensions_response("```json\n[5, 5]\n```") == [5, 5]

    # ── Per-dimension ranges (nested list) ──

    def test_per_dim_fixed_values(self):
        """[2, 3, 4] → [[2,2],[3,3],[4,4]]."""
        assert _parse_dimensions_response("[[2,2],[3,3],[4,4]]") == [
            [2, 2], [3, 3], [4, 4],
        ]

    def test_per_dim_with_nulls(self):
        """Mixed null ranges."""
        assert _parse_dimensions_response("[[null,null],[3,3]]") == [
            [None, None], [3, 3],
        ]

    def test_per_dim_range(self):
        """1-8 → [[1,8]]."""
        assert _parse_dimensions_response("[[1,8]]") == [[1, 8]]

    def test_per_dim_in_code_block(self):
        """Handle markdown code block wrapping."""
        text = "```json\n[[2,2],[3,3],[4,4]]\n```"
        assert _parse_dimensions_response(text) == [[2, 2], [3, 3], [4, 4]]

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
        assert result == [[], [3, 3]]

    def test_regex_fallback_rank_spec(self):
        """Regex fallback still works for rank spec."""
        text = "Here is the answer: [5, 5] done"
        assert _parse_dimensions_response(text) == [5, 5]

    def test_regex_fallback_per_dim(self):
        """Regex fallback still works for per-dimension."""
        text = "Answer: [[2,2],[3,3]]"
        assert _parse_dimensions_response(text) == [[2, 2], [3, 3]]


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
        assert _try_deterministic_parse("1-D") == [1, 1]
        assert _try_deterministic_parse("1D") == [1, 1]

    def test_n_d(self):
        assert _try_deterministic_parse("2D") == [2, 2]
        assert _try_deterministic_parse("3-D") == [3, 3]
        assert _try_deterministic_parse("5D") == [5, 5]

    def test_rank_range(self):
        """'X-Y' rank range: supports X to Y dimensions."""
        assert _try_deterministic_parse("0-8") == [0, 8]
        assert _try_deterministic_parse("1-8") == [1, 8]
        assert _try_deterministic_parse("2~5") == [2, 5]
        assert _try_deterministic_parse("1 - 4") == [1, 4]

    def test_parentheses(self):
        assert _try_deterministic_parse("(N,C,H,W)") == [4, 4]
        assert _try_deterministic_parse("(N, D, H, W, C)") == [5, 5]
        assert _try_deterministic_parse("(batch, seq)") == [2, 2]

    def test_brackets(self):
        assert _try_deterministic_parse("[2, 3, 4]") == [[2, 2], [3, 3], [4, 4]]
        assert _try_deterministic_parse("[8]") == [[8, 8]]
        assert _try_deterministic_parse("[16, 32]") == [[16, 16], [32, 32]]

    def test_same_as_input(self):
        assert _try_deterministic_parse("与输入相同") == []
        assert _try_deterministic_parse("same as input") == []

    def test_unknown_patterns(self):
        assert _try_deterministic_parse("variable") is None
        assert _try_deterministic_parse("2D or 3D") is None
        assert _try_deterministic_parse("") is None

    def test_whitespace_handling(self):
        assert _try_deterministic_parse("  2D  ") == [2, 2]
        assert _try_deterministic_parse("  0-8  ") == [0, 8]


# ---------------------------------------------------------------------------
# _validate_dimensions_structure
# ---------------------------------------------------------------------------


class TestValidateDimensionsStructure:
    def test_empty_list(self):
        is_valid, error = _validate_dimensions_structure([])
        assert is_valid
        assert error == ""

    def test_rank_format_exact(self):
        """[N, N] exact rank is valid."""
        is_valid, error = _validate_dimensions_structure([2, 2])
        assert is_valid
        assert error == ""

    def test_rank_format_range(self):
        """[min_rank, max_rank] rank range is valid."""
        is_valid, error = _validate_dimensions_structure([0, 8])
        assert is_valid
        assert error == ""

        is_valid, error = _validate_dimensions_structure([1, 8])
        assert is_valid
        assert error == ""

        is_valid, error = _validate_dimensions_structure([2, 5])
        assert is_valid
        assert error == ""

    def test_rank_format_zero_scalar(self):
        """[0, 0] supports scalar (0-dimensional) tensors."""
        is_valid, error = _validate_dimensions_structure([0, 0])
        assert is_valid
        assert error == ""

    def test_rank_format_min_greater_than_max(self):
        """[3, 1] is invalid: min > max."""
        is_valid, error = _validate_dimensions_structure([3, 1])
        assert not is_valid
        assert "min <= max" in error

    def test_rank_format_negative_min(self):
        """[-1, 5] is invalid: negative min."""
        is_valid, error = _validate_dimensions_structure([-1, 5])
        assert not is_valid
        assert "must be >= 0" in error

    def test_rank_format_too_large(self):
        is_valid, error = _validate_dimensions_structure([0, 11])
        assert not is_valid
        assert "Too many dimensions" in error

    def test_per_dimension_valid(self):
        is_valid, error = _validate_dimensions_structure([[1, 8], [3, 3], [4, 4]])
        assert is_valid
        assert error == ""

    def test_per_dimension_with_null(self):
        is_valid, error = _validate_dimensions_structure([[None, None], [3, 5]])
        assert is_valid
        assert error == ""

    def test_per_dimension_min_greater_than_max(self):
        is_valid, error = _validate_dimensions_structure([[5, 3]])
        assert not is_valid
        assert "min (5) > max (3)" in error

    def test_per_dimension_wrong_length(self):
        is_valid, error = _validate_dimensions_structure([[1, 2, 3]])
        assert not is_valid
        assert "must be [min, max]" in error

    def test_per_dimension_wrong_type(self):
        is_valid, error = _validate_dimensions_structure([["a", "b"]])
        assert not is_valid
        assert "must be int/float or null" in error

    def test_per_dimension_too_many(self):
        dims = [[i, i] for i in range(11)]
        is_valid, error = _validate_dimensions_structure(dims)
        assert not is_valid
        assert "Too many dimensions" in error

    def test_not_a_list(self):
        is_valid, error = _validate_dimensions_structure("not a list")
        assert not is_valid
        assert "must be a list" in error


# ---------------------------------------------------------------------------
# _validate_dimensions_alignment
# ---------------------------------------------------------------------------


class TestValidateDimensionsAlignment:
    def test_aligned(self):
        is_valid, error = _validate_dimensions_alignment(3, [[1, 2], [3, 4], [5, 6]])
        assert is_valid
        assert error == ""

    def test_mismatch_fewer(self):
        is_valid, error = _validate_dimensions_alignment(3, [[1, 2], [3, 4]])
        assert not is_valid
        assert "expected 3 entries, got 2" in error

    def test_mismatch_more(self):
        is_valid, error = _validate_dimensions_alignment(2, [[1, 2], [3, 4], [5, 6]])
        assert not is_valid
        assert "expected 2 entries, got 3" in error


# ---------------------------------------------------------------------------
# _is_rank_format
# ---------------------------------------------------------------------------


class TestIsRankFormat:
    def test_valid_rank(self):
        assert _is_rank_format([2, 2]) is True
        assert _is_rank_format([5, 5]) is True

    def test_not_two_elements(self):
        assert _is_rank_format([2]) is False
        assert _is_rank_format([2, 2, 2]) is False

    def test_not_integers(self):
        assert _is_rank_format([2.0, 2.0]) is False
        assert _is_rank_format([[1, 2], [3, 4]]) is False

    def test_not_list(self):
        assert _is_rank_format("not a list") is False


# ---------------------------------------------------------------------------
# _try_deterministic_range
# ---------------------------------------------------------------------------


class TestTryDeterministicRange:
    def test_bracket_range(self):
        assert _try_deterministic_range("[0, 100]") == [[0, 100]]
        assert _try_deterministic_range("[1, 8]") == [[1, 8]]

    def test_dash_range(self):
        assert _try_deterministic_range("0-100") == [[0, 100]]
        assert _try_deterministic_range("1-1024") == [[1, 1024]]

    def test_tilde_range(self):
        assert _try_deterministic_range("0~100") == [[0, 100]]

    def test_comma_range(self):
        assert _try_deterministic_range("0, 100") == [[0, 100]]

    def test_enum_values(self):
        result = _try_deterministic_range("枚举值: 1, 2, 3")
        assert result == [[1, 1], [2, 2], [3, 3]]

    def test_no_match(self):
        assert _try_deterministic_range("无限制") is None
        assert _try_deterministic_range("") is None

    def test_negative_numbers(self):
        assert _try_deterministic_range("[-1, 1]") == [[-1, 1]]


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
