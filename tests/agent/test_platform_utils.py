"""Tests for utils/platform_utils.py: platform normalization, splitting, and resolution."""

from __future__ import annotations

from agent.utils.platform_utils import (
    normalize_platform_name,
    resolve_target_platforms,
    split_platforms,
)

SUPPORTED_PLATFORMS = [
    "Atlas 训练系列产品",
    "Atlas 推理系列产品",
    "Atlas A2 训练系列产品/Atlas A2 推理系列产品",
    "Atlas A3 训练系列产品/Atlas A3 推理系列产品",
    "Atlas 200I/500 A2 推理产品",
]


# ---------------------------------------------------------------------------
# normalize_platform_name
# ---------------------------------------------------------------------------


class TestNormalizePlatformName:
    def test_standard_name_unchanged(self):
        assert normalize_platform_name("Atlas 推理系列产品") == "Atlas 推理系列产品"

    def test_missing_space_alias(self):
        assert normalize_platform_name("Atlas推理系列产品") == "Atlas 推理系列产品"

    def test_missing_space_training(self):
        assert normalize_platform_name("Atlas训练系列产品") == "Atlas 训练系列产品"

    def test_a2_alias(self):
        result = normalize_platform_name("Atlas A2训练系列产品/Atlas A2推理系列产品")
        assert result == "Atlas A2 训练系列产品/Atlas A2 推理系列产品"

    def test_a3_alias(self):
        result = normalize_platform_name("Atlas A3训练系列产品/Atlas A3推理系列产品")
        assert result == "Atlas A3 训练系列产品/Atlas A3 推理系列产品"

    def test_empty_string(self):
        assert normalize_platform_name("") == ""

    def test_whitespace_only(self):
        assert normalize_platform_name("   ") == ""

    def test_unknown_name_returned_as_is(self):
        assert normalize_platform_name("SomeUnknownPlatform") == "SomeUnknownPlatform"


# ---------------------------------------------------------------------------
# split_platforms
# ---------------------------------------------------------------------------


class TestSplitPlatforms:
    def test_empty_string(self):
        assert split_platforms("") == [""]

    def test_whitespace_only(self):
        assert split_platforms("   ") == [""]

    def test_single_platform(self):
        result = split_platforms("Atlas 推理系列产品")
        assert result == ["Atlas 推理系列产品"]

    def test_single_platform_with_alias(self):
        result = split_platforms("Atlas推理系列产品")
        assert result == ["Atlas 推理系列产品"]

    def test_chinese_comma_separator(self):
        result = split_platforms("Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品")
        assert len(result) == 2
        assert "Atlas A2 训练系列产品/Atlas A2 推理系列产品" in result
        assert "Atlas A3 训练系列产品/Atlas A3 推理系列产品" in result

    def test_or_separator(self):
        result = split_platforms("Atlas A2 训练系列产品/Atlas A2 推理系列产品或Atlas 推理系列产品")
        assert len(result) == 2
        assert "Atlas A2 训练系列产品/Atlas A2 推理系列产品" in result
        assert "Atlas 推理系列产品" in result

    def test_english_comma_separator(self):
        result = split_platforms("Atlas 训练系列产品,Atlas 推理系列产品")
        assert len(result) == 2
        assert "Atlas 训练系列产品" in result
        assert "Atlas 推理系列产品" in result

    def test_chinese_full_comma_separator(self):
        result = split_platforms("Atlas 训练系列产品，Atlas 推理系列产品")
        assert len(result) == 2

    def test_huozhe_separator(self):
        result = split_platforms("Atlas 训练系列产品或者Atlas 推理系列产品")
        assert len(result) == 2

    def test_yiji_separator(self):
        result = split_platforms("Atlas 训练系列产品以及Atlas 推理系列产品")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# resolve_target_platforms
# ---------------------------------------------------------------------------


class TestResolveTargetPlatforms:
    def test_empty_platform_means_all(self):
        result = resolve_target_platforms("", SUPPORTED_PLATFORMS)
        assert result == SUPPORTED_PLATFORMS

    def test_whitespace_platform_means_all(self):
        result = resolve_target_platforms("   ", SUPPORTED_PLATFORMS)
        assert result == SUPPORTED_PLATFORMS

    def test_single_supported_platform(self):
        result = resolve_target_platforms("Atlas 推理系列产品", SUPPORTED_PLATFORMS)
        assert result == ["Atlas 推理系列产品"]

    def test_single_unsupported_platform_filtered(self):
        result = resolve_target_platforms("x的数据格式为NZ", SUPPORTED_PLATFORMS)
        assert result == []

    def test_multiple_platforms(self):
        platform_str = "Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品"
        result = resolve_target_platforms(platform_str, SUPPORTED_PLATFORMS)
        assert len(result) == 2
        assert "Atlas A2 训练系列产品/Atlas A2 推理系列产品" in result
        assert "Atlas A3 训练系列产品/Atlas A3 推理系列产品" in result

    def test_mixed_valid_and_invalid(self):
        platform_str = "Atlas 推理系列产品、x的数据格式为NZ"
        result = resolve_target_platforms(platform_str, SUPPORTED_PLATFORMS)
        assert result == ["Atlas 推理系列产品"]

    def test_alias_normalized_and_matched(self):
        result = resolve_target_platforms("Atlas推理系列产品", SUPPORTED_PLATFORMS)
        assert result == ["Atlas 推理系列产品"]

    def test_non_platform_info_filtered(self):
        result = resolve_target_platforms("scale的数据格式为NZ", SUPPORTED_PLATFORMS)
        assert result == []

    def test_complex_non_platform_info_filtered(self):
        result = resolve_target_platforms("x的数据格式为ND且dstType取值为29", SUPPORTED_PLATFORMS)
        assert result == []

    def test_three_platforms_with_or(self):
        platform_str = "Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品或Atlas 推理系列产品"
        result = resolve_target_platforms(platform_str, SUPPORTED_PLATFORMS)
        assert len(result) == 3
