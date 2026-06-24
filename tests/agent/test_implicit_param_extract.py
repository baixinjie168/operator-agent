"""Tests for quantization_type implicit parameter extraction."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure agent package is importable
ROOT = Path(__file__).resolve().parents[2]
for pkg in ("packages/shared/src", "packages/mcp-server/src", "packages/agent/src"):
    p = str(ROOT / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.nodes.param_relation_extract.implicit_param_extract import (  # noqa: E402
    _QUANTIZATION_CANDIDATES,
    _build_quantization_type_mapping,
    _extract_quantization_modes,
)


# ---------------------------------------------------------------------------
# _extract_quantization_modes
# ---------------------------------------------------------------------------


class TestExtractQuantizationModes:
    def test_full_hit_all_four(self):
        text = "支持 per-channel、per-group、per-tensor、per-token 四种模式"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-channel", "per-group", "per-tensor", "per-token"]

    def test_partial_hit_preserves_canonical_order(self):
        # Document mentions per-tensor before per-channel, but result keeps
        # the canonical _QUANTIZATION_CANDIDATES order.
        text = "支持per-tensor，per-channel"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-channel", "per-tensor"]

    def test_single_hit_per_group(self):
        text = "per-group下输入为二维向量"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-group"]

    def test_no_hit_returns_empty(self):
        text = "这是一个没有量化粒度关键词的普通算子说明文档。"
        modes, _ = _extract_quantization_modes(text)
        assert modes == []

    def test_empty_text_returns_empty(self):
        modes, _ = _extract_quantization_modes("")
        assert modes == []

    def test_deduplication_repeated(self):
        text = "per-channel ... per-channel ... per-channel"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-channel"]

    def test_word_boundary_no_false_positive_percent(self):
        """'percent-channel' must not match 'per-channel'."""
        text = "使用 percent-channel 编码"
        modes, _ = _extract_quantization_modes(text)
        assert modes == []

    def test_word_boundary_no_false_positive_substring(self):
        """'super-channel' must not match 'per-channel'."""
        text = "super-channel 模式"
        modes, _ = _extract_quantization_modes(text)
        assert modes == []

    def test_source_citation_returned(self):
        text = "前缀文字 per-tensor 后缀文字"
        _, citation = _extract_quantization_modes(text)
        assert "per-tensor" in citation

    def test_candidates_constant_has_four_values(self):
        assert _QUANTIZATION_CANDIDATES == (
            "per-channel", "per-group", "per-tensor", "per-token",
        )


# ---------------------------------------------------------------------------
# _build_quantization_type_mapping
# ---------------------------------------------------------------------------


class TestBuildQuantizationTypeMapping:
    def test_structure_with_hits(self):
        text = "支持 per-channel、per-tensor 两种"
        m = _build_quantization_type_mapping(text)
        assert m["var_name"] == "quantization_type"
        assert m["is_quantization_type"] is True
        assert m["param_type"] == "char"
        assert m["allowed_range_type"] == "enum"
        assert m["allowed_range_value"] == ["per-channel", "per-tensor"]
        assert m["is_constant"] is False
        assert m["is_external_constant"] is False
        assert m["is_compound"] is False
        assert m["tensor_param"] is None
        assert m["dim_index"] is None
        assert "per-channel" in m["source_citation"]

    def test_empty_allowed_range_when_no_hit(self):
        """No quantization terms → empty allowed_range_value, but still present."""
        m = _build_quantization_type_mapping("普通算子说明，无量化词")
        assert m["var_name"] == "quantization_type"
        assert m["allowed_range_value"] == []
        assert m["allowed_range_type"] == "enum"
        assert m["param_type"] == "char"
        assert m["source_citation"] == ""

    def test_full_four_modes(self):
        text = "per-channel per-group per-tensor per-token"
        m = _build_quantization_type_mapping(text)
        assert m["allowed_range_value"] == [
            "per-channel", "per-group", "per-tensor", "per-token",
        ]

    def test_empty_text_still_returns_mapping(self):
        m = _build_quantization_type_mapping("")
        assert m["var_name"] == "quantization_type"
        assert m["allowed_range_value"] == []
