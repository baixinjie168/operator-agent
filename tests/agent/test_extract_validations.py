"""Unit tests for shape/dtype/dformat validation logic in extract nodes."""

import pytest

from agent.nodes.shape_extract import _is_shape_valid
from agent.nodes.dtype_extract import _is_dtype_valid, _VALID_DTYPES
from agent.nodes.dformat_extract import _is_dformat_valid, _VALID_DFORMATS


# ---------------------------------------------------------------------------
# _is_shape_valid
# ---------------------------------------------------------------------------


class TestShapeValid:
    """Tests for shape validity checking."""

    def test_empty_string_is_invalid(self):
        assert _is_shape_valid("") is False

    def test_whitespace_is_invalid(self):
        assert _is_shape_valid("   ") is False

    @pytest.mark.parametrize("dash", ["-", "—", "–", "－"])
    def test_dash_variants_are_invalid(self, dash):
        assert _is_shape_valid(dash) is False

    def test_relative_ref_with_yu_is_invalid(self):
        assert _is_shape_valid("与self一致") is False

    def test_relative_ref_with_tong_is_invalid(self):
        assert _is_shape_valid("与input相同") is False

    def test_relative_ref_with_backticks_is_invalid(self):
        assert _is_shape_valid("与`x`一样") is False

    def test_relative_ref_with_he_is_invalid(self):
        assert _is_shape_valid("和other保持一致") is False

    def test_relative_ref_with_gen_is_invalid(self):
        assert _is_shape_valid("跟y一致") is False

    def test_valid_shape_tuple(self):
        assert _is_shape_valid("(N,C,H,W)") is True

    def test_valid_shape_rank(self):
        assert _is_shape_valid("2D") is True

    def test_valid_shape_range(self):
        assert _is_shape_valid("1-8") is True

    def test_valid_shape_fixed(self):
        assert _is_shape_valid("[2,3,4]") is True

    def test_valid_shape_scalar_chinese(self):
        assert _is_shape_valid("标量") is True

    def test_valid_shape_with_text(self):
        assert _is_shape_valid("(batch_size, seq_len)") is True


# ---------------------------------------------------------------------------
# _is_dtype_valid
# ---------------------------------------------------------------------------


class TestDtypeValid:
    """Tests for dtype whitelist validation."""

    def test_empty_string_is_invalid(self):
        assert _is_dtype_valid("") is False

    def test_whitespace_is_invalid(self):
        assert _is_dtype_valid("   ") is False

    def test_dash_is_invalid(self):
        assert _is_dtype_valid("-") is False

    def test_relative_ref_is_invalid(self):
        assert _is_dtype_valid("与x一致") is False

    def test_single_valid_dtype_float32(self):
        assert _is_dtype_valid("FLOAT32") is True

    def test_single_valid_dtype_int8(self):
        assert _is_dtype_valid("INT8") is True

    def test_single_valid_dtype_bfloat16(self):
        assert _is_dtype_valid("BFLOAT16") is True

    def test_multiple_valid_dtypes_comma(self):
        assert _is_dtype_valid("FLOAT32,FLOAT16") is True

    def test_multiple_valid_dtypes_chinese_comma(self):
        assert _is_dtype_valid("INT8、INT16、INT32") is True

    def test_multiple_valid_dtypes_slash(self):
        assert _is_dtype_valid("UINT64/INT64") is True

    def test_mixed_case_is_valid(self):
        assert _is_dtype_valid("float32") is True

    def test_unknown_dtype_is_invalid(self):
        assert _is_dtype_valid("UNKNOWN_TYPE") is False

    def test_partial_unknown_is_invalid(self):
        assert _is_dtype_valid("FLOAT32,MYSTERY") is False

    def test_all_whitelist_entries_are_uppercase(self):
        for dt in _VALID_DTYPES:
            assert dt == dt.upper(), f"{dt} is not uppercase"

    @pytest.mark.parametrize("dt", [
        "FLOAT", "FLOAT32", "FLOAT16", "INT8", "INT32", "UINT8",
        "INT16", "UINT16", "UINT32", "INT64", "UINT64", "DOUBLE",
        "FLOAT64", "BOOL", "STRING", "COMPLEX64", "COMPLEX128",
        "BF16", "BFLOAT16", "INT", "UINT1", "COMPLEX32",
    ])
    def test_each_whitelist_entry_is_valid(self, dt):
        assert _is_dtype_valid(dt) is True


# ---------------------------------------------------------------------------
# _is_dformat_valid
# ---------------------------------------------------------------------------


class TestDformatValid:
    """Tests for dformat whitelist validation."""

    def test_empty_string_is_invalid(self):
        assert _is_dformat_valid("") is False

    def test_whitespace_is_invalid(self):
        assert _is_dformat_valid("   ") is False

    def test_dash_is_invalid(self):
        assert _is_dformat_valid("-") is False

    def test_relative_ref_is_invalid(self):
        assert _is_dformat_valid("与input一致") is False

    def test_single_valid_dformat_nchw(self):
        assert _is_dformat_valid("NCHW") is True

    def test_single_valid_dformat_nd(self):
        assert _is_dformat_valid("ND") is True

    def test_single_valid_dformat_fractal_z(self):
        assert _is_dformat_valid("FRACTAL_Z") is True

    def test_multiple_valid_dformats_comma(self):
        assert _is_dformat_valid("NCHW,NHWC") is True

    def test_multiple_valid_dformats_chinese_comma(self):
        assert _is_dformat_valid("ND、NCDHW") is True

    def test_unknown_dformat_is_invalid(self):
        assert _is_dformat_valid("CUSTOM_FORMAT") is False

    def test_partial_unknown_is_invalid(self):
        assert _is_dformat_valid("NCHW,BOGUS") is False

    def test_mixed_case_is_valid(self):
        assert _is_dformat_valid("nchw") is True

    @pytest.mark.parametrize("df", [
        "ND", "NCHW", "NHWC", "HWCN", "NDHWC", "NCDHW", "NC", "NCL",
        "NC1HWC0", "FRACTAL_Z", "NC1HWC0_C04", "FRACTAL_NZ",
        "NDC1HWC0", "FRACTAL_Z_3D",
    ])
    def test_each_whitelist_entry_is_valid(self, df):
        assert _is_dformat_valid(df) is True
