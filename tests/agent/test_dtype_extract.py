"""Tests for dtype_extract.py: regex fallback + conditional dtype (Item 4).

Covers T4-1..T4-8 from the Phase 2 plan:
- _regex_fallback_dtype: zero-LLM-cost dtype token extraction from text
- _detect_conditional_dtype: quantization-scenario conditional dtype
- _is_plain_dtype_valid: validation gating the fallback
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

from agent.nodes.dtype_extract import (  # noqa: E402
    _detect_conditional_dtype,
    _is_plain_dtype_valid,
    _regex_fallback_dtype,
)


# ---------------------------------------------------------------------------
# _regex_fallback_dtype
# ---------------------------------------------------------------------------

class TestRegexFallbackDtype:
    def test_t4_7_bf16_normalized(self):
        """T4-7: 'BF16' is normalized to BFLOAT16 (in VALID_DTYPES)."""
        result = _regex_fallback_dtype("数据类型为BF16")
        assert result == "BFLOAT16"

    def test_t4_2_extracts_from_param_desc(self):
        """T4-2: extracts FLOAT16 from a param_desc-style string."""
        result = _regex_fallback_dtype("数据类型: FLOAT16")
        assert result == "FLOAT16"

    def test_multiple_tokens_sorted(self):
        """Multiple dtype tokens are sorted and comma-joined."""
        result = _regex_fallback_dtype("支持INT8和FLOAT16")
        assert result == "FLOAT16, INT8"

    def test_double_normalized(self):
        """DOUBLE is normalized to FLOAT64."""
        result = _regex_fallback_dtype("类型 DOUBLE")
        assert result == "FLOAT64"

    def test_empty_description_returns_none(self):
        assert _regex_fallback_dtype("") is None
        assert _regex_fallback_dtype(None) is None  # type: ignore[arg-type]

    def test_no_token_returns_none(self):
        """T4-3: no dtype token → None (does not block pipeline)."""
        assert _regex_fallback_dtype("这是一个普通参数描述") is None

    def test_case_insensitive(self):
        result = _regex_fallback_dtype("dtype is float16")
        assert result == "FLOAT16"

    def test_bfloat16_kept(self):
        result = _regex_fallback_dtype("使用BFLOAT16")
        assert result == "BFLOAT16"


# ---------------------------------------------------------------------------
# _detect_conditional_dtype
# ---------------------------------------------------------------------------

class TestDetectConditionalDtype:
    def test_t4_4_quantization_scenario(self):
        """T4-4: 量化场景下x为INT8，非量化为FLOAT16."""
        desc = "量化场景下x为INT8，非量化为FLOAT16"
        result = _detect_conditional_dtype(desc)
        assert result is not None
        assert result["cond_dtype"] == "INT8"
        assert result["default_dtype"] == "FLOAT16"
        assert result["condition"] == "量化"

    def test_quant_no_default(self):
        """Quantization dtype found but no default mentioned."""
        result = _detect_conditional_dtype("量化时为INT8")
        assert result is not None
        assert result["cond_dtype"] == "INT8"
        assert result["default_dtype"] is None

    def test_no_quant_returns_none(self):
        assert _detect_conditional_dtype("数据类型为FLOAT16") is None

    def test_empty_returns_none(self):
        assert _detect_conditional_dtype("") is None
        assert _detect_conditional_dtype(None) is None  # type: ignore[arg-type]

    def test_bf16_in_quant(self):
        result = _detect_conditional_dtype("量化场景下为BF16，非量化为FLOAT32")
        assert result is not None
        assert result["cond_dtype"] == "BFLOAT16"
        assert result["default_dtype"] == "FLOAT32"

    def test_quant_english(self):
        result = _detect_conditional_dtype(
            "quantization mode: INT8, non-quant: FLOAT16"
        )
        assert result is not None
        assert result["cond_dtype"] == "INT8"


# ---------------------------------------------------------------------------
# _is_plain_dtype_valid (gating the fallback)
# ---------------------------------------------------------------------------

class TestIsPlainDtypeValid:
    def test_valid_single(self):
        assert _is_plain_dtype_valid("FLOAT16") is True

    def test_valid_csv(self):
        assert _is_plain_dtype_valid("FLOAT16, INT8") is True

    def test_invalid_token(self):
        assert _is_plain_dtype_valid("GARBAGE") is False

    def test_empty(self):
        assert _is_plain_dtype_valid("") is False

    def test_dash(self):
        assert _is_plain_dtype_valid("-") is False

    def test_cross_reference(self):
        assert _is_plain_dtype_valid("与query一致") is False
