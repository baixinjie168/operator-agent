"""Tests for the table_column_extract node and table_parser utilities."""

from __future__ import annotations

import json

import pytest

from agent.utils.table_parser import (
    _extract_discontinuous,
    detect_table_columns,
    extract_4_columns_from_table,
    find_param_name_column,
    is_table_form,
    parse_html_tables,
)

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

BASIC_TABLE_HTML = """
<table>
<tr>
  <th>参数名</th><th>输入/输出</th><th>数据类型</th>
  <th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th>
</tr>
<tr>
  <td>x</td><td>输入</td><td>FLOAT32、FLOAT16</td>
  <td>ND</td><td>1-8</td><td>√</td>
</tr>
<tr>
  <td>y</td><td>输入</td><td>INT64</td>
  <td>ND、NZ</td><td>-</td><td>×</td>
</tr>
<tr>
  <td>offset</td><td>输入</td><td>uint64_t</td>
  <td>-</td><td>1</td><td>-</td>
</tr>
</table>
"""

SIMPLE_TABLE_HTML = """
<table>
<tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
<tr><td>handle</td><td>输入</td><td>tpu handle</td></tr>
</table>
"""

ROWSPAN_TABLE_HTML = """
<table>
<tr>
  <th>参数名</th><th>数据类型</th><th>数据格式</th>
  <th>维度(shape)</th><th>非连续Tensor</th>
</tr>
<tr>
  <td>x1</td><td rowspan="2">FLOAT32</td><td>ND</td><td>1-8</td><td>√</td>
</tr>
<tr>
  <td>x2</td><td>NZ</td><td>2-4</td><td>×</td>
</tr>
</table>
"""


# ---------------------------------------------------------------------------
# Test parse_html_tables
# ---------------------------------------------------------------------------

class TestParseHtmlTables:
    def test_basic_table(self):
        tables = parse_html_tables(BASIC_TABLE_HTML)
        assert len(tables) == 1
        grid = tables[0]
        assert len(grid) == 4  # 1 header + 3 data rows
        assert grid[0] == ["参数名", "输入/输出", "数据类型", "数据格式", "维度(shape)", "非连续Tensor"]

    def test_simple_table(self):
        tables = parse_html_tables(SIMPLE_TABLE_HTML)
        assert len(tables) == 1
        assert len(tables[0]) == 2  # 1 header + 1 data row

    def test_rowspan_expansion(self):
        tables = parse_html_tables(ROWSPAN_TABLE_HTML)
        grid = tables[0]
        # x2 row should have FLOAT32 in dtype column due to rowspan
        assert grid[2][1] == "FLOAT32"

    def test_empty_content(self):
        assert parse_html_tables("") == []
        assert parse_html_tables("no tables here") == []

    def test_multiple_tables(self):
        html = BASIC_TABLE_HTML + SIMPLE_TABLE_HTML
        tables = parse_html_tables(html)
        assert len(tables) == 2


# ---------------------------------------------------------------------------
# Test detect_table_columns
# ---------------------------------------------------------------------------

class TestDetectTableColumns:
    def test_full_header(self):
        header = ["参数名", "输入/输出", "数据类型", "数据格式", "维度(shape)", "非连续Tensor"]
        col_map = detect_table_columns(header)
        assert col_map["dtype_desc"] == 2
        assert col_map["dformat_desc"] == 3
        assert col_map["shape"] == 4
        assert col_map["is_support_discontinuous"] == 5

    def test_simple_header(self):
        header = ["参数名", "输入/输出", "描述"]
        col_map = detect_table_columns(header)
        assert col_map == {"direction": 1, "param_desc": 2}

    def test_partial_header(self):
        header = ["参数名", "数据类型", "描述"]
        col_map = detect_table_columns(header)
        assert col_map == {"dtype_desc": 1, "param_desc": 2}


class TestIsTableForm:
    def test_full_table(self):
        header = ["参数名", "输入/输出", "数据类型", "数据格式", "维度(shape)", "非连续Tensor"]
        assert is_table_form(header) is True

    def test_simple_table(self):
        header = ["参数名", "输入/输出", "描述"]
        assert is_table_form(header) is False

    def test_three_columns(self):
        header = ["参数名", "数据类型", "数据格式", "维度(shape)", "描述"]
        assert is_table_form(header) is True


class TestFindParamNameColumn:
    def test_chinese_header(self):
        assert find_param_name_column(["参数名", "数据类型"]) == 0

    def test_english_header(self):
        assert find_param_name_column(["Parameter", "Type"]) == 0

    def test_fallback(self):
        assert find_param_name_column(["foo", "bar"]) == 0

    def test_offset_column(self):
        assert find_param_name_column(["序号", "参数名", "类型"]) == 1


# ---------------------------------------------------------------------------
# Test extract_4_columns_from_table
# ---------------------------------------------------------------------------

class TestExtract4Columns:
    def _get_grid_and_map(self, html: str = BASIC_TABLE_HTML):
        tables = parse_html_tables(html)
        grid = tables[0]
        col_map = detect_table_columns(grid[0])
        name_idx = find_param_name_column(grid[0])
        return grid, col_map, name_idx

    def test_tensor_param_with_all_columns(self):
        grid, col_map, name_idx = self._get_grid_and_map()
        result = extract_4_columns_from_table(grid, col_map, "x", "aclTensor", name_idx)
        assert result["shape"] == "1-8"
        assert result["dtype_desc"] == "FLOAT32、FLOAT16"
        assert result["dformat_desc"] == "ND"
        disc = json.loads(result["is_support_discontinuous"])
        assert disc["value"] is True
        assert disc["src_text"] == "√"

    def test_dash_values_treated_as_empty(self):
        grid, col_map, name_idx = self._get_grid_and_map()
        result = extract_4_columns_from_table(grid, col_map, "y", "aclTensor", name_idx)
        assert result.get("shape") in (None, "")  # dash → not included or empty
        assert result["dtype_desc"] == "INT64"
        assert result["dformat_desc"] == "ND、NZ"
        disc = json.loads(result["is_support_discontinuous"])
        assert disc["value"] is False

    def test_non_tensor_param(self):
        grid, col_map, name_idx = self._get_grid_and_map()
        result = extract_4_columns_from_table(grid, col_map, "offset", "uint64_t", name_idx)
        assert result["shape"] == "1"
        assert result.get("dformat_desc") in (None, "")  # dash → not included or empty
        disc = json.loads(result["is_support_discontinuous"])
        assert disc["value"] == "N/A"

    def test_param_not_found(self):
        grid, col_map, name_idx = self._get_grid_and_map()
        result = extract_4_columns_from_table(grid, col_map, "nonexistent", "aclTensor", name_idx)
        assert result == {}

    def test_simple_table_no_target_columns(self):
        # Use a header with no target columns (no shape/dtype/dformat/disc/desc/direction)
        header = ["参数名", "类别", "备注"]
        grid = [header, ["handle", "必需", "tpu handle"]]
        col_map = detect_table_columns(header)
        assert col_map == {}
        result = extract_4_columns_from_table(grid, col_map, "handle", "aclTensor", 0)
        assert result == {}

    def test_rowspan_table(self):
        grid, col_map, name_idx = self._get_grid_and_map(ROWSPAN_TABLE_HTML)
        # x2 should inherit FLOAT32 from rowspan
        result = extract_4_columns_from_table(grid, col_map, "x2", "aclTensor", name_idx)
        assert result["dtype_desc"] == "FLOAT32"
        assert result["shape"] == "2-4"

    def test_param_desc_extraction(self):
        header = ["参数名", "数据类型", "描述"]
        grid = [header, ["x", "FLOAT32", "输入张量"], ["y", "INT64", "维度数量"]]
        col_map = detect_table_columns(header)
        assert col_map == {"dtype_desc": 1, "param_desc": 2}
        result = extract_4_columns_from_table(grid, col_map, "x", "aclTensor", 0)
        assert result["param_desc"] == "输入张量"
        assert result["dtype_desc"] == "FLOAT32"

    def test_param_desc_dash_treated_as_empty(self):
        header = ["参数名", "描述"]
        grid = [header, ["x", "-"]]
        col_map = detect_table_columns(header)
        result = extract_4_columns_from_table(grid, col_map, "x", "aclTensor", 0)
        assert result.get("param_desc") == ""  # dash → empty string

    def test_direction_input_chinese(self):
        header = ["参数名", "输入/输出", "数据类型"]
        grid = [header, ["x", "输入", "FLOAT32"]]
        col_map = detect_table_columns(header)
        assert col_map["direction"] == 1
        result = extract_4_columns_from_table(grid, col_map, "x", "aclTensor", 0)
        assert result["direction"] == "input"

    def test_direction_output_chinese(self):
        header = ["参数名", "输入/输出"]
        grid = [header, ["y", "输出"]]
        col_map = detect_table_columns(header)
        result = extract_4_columns_from_table(grid, col_map, "y", "aclTensor", 0)
        assert result["direction"] == "output"

    def test_direction_dash_ignored(self):
        header = ["参数名", "输入/输出"]
        grid = [header, ["z", "-"]]
        col_map = detect_table_columns(header)
        result = extract_4_columns_from_table(grid, col_map, "z", "aclTensor", 0)
        assert "direction" not in result  # dash → empty → not included


# ---------------------------------------------------------------------------
# Test _extract_discontinuous
# ---------------------------------------------------------------------------

class TestExtractDiscontinuous:
    def test_supported_checkmark(self):
        disc = json.loads(_extract_discontinuous("√", "aclTensor"))
        assert disc["value"] is True

    def test_supported_tick(self):
        disc = json.loads(_extract_discontinuous("✓", "aclTensor"))
        assert disc["value"] is True

    def test_supported_chinese(self):
        disc = json.loads(_extract_discontinuous("支持", "aclTensor"))
        assert disc["value"] is True

    def test_not_supported_cross(self):
        disc = json.loads(_extract_discontinuous("×", "aclTensor"))
        assert disc["value"] is False

    def test_not_supported_chinese(self):
        disc = json.loads(_extract_discontinuous("不支持", "aclTensor"))
        assert disc["value"] is False

    def test_dash_tensor_returns_false(self):
        disc = json.loads(_extract_discontinuous("-", "aclTensor"))
        assert disc["value"] is False

    def test_dash_non_tensor_returns_na(self):
        disc = json.loads(_extract_discontinuous("-", "uint64_t"))
        assert disc["value"] == "N/A"

    def test_empty_tensor_returns_false(self):
        disc = json.loads(_extract_discontinuous("", "aclTensor"))
        assert disc["value"] is False

    def test_empty_non_tensor_returns_na(self):
        disc = json.loads(_extract_discontinuous("", "int64_t"))
        assert disc["value"] == "N/A"

    def test_unrecognised_tensor_fallback_false(self):
        disc = json.loads(_extract_discontinuous("unknown", "aclTensor"))
        assert disc["value"] is False

    def test_unrecognised_non_tensor_fallback_na(self):
        disc = json.loads(_extract_discontinuous("unknown", "int64_t"))
        assert disc["value"] == "N/A"

    def test_src_text_preserved(self):
        disc = json.loads(_extract_discontinuous("√", "aclTensor"))
        assert disc["src_text"] == "√"


# ---------------------------------------------------------------------------
# Relative-reference handling (e.g. "与self一致")
# ---------------------------------------------------------------------------

from agent.utils.table_parser import _is_relative_ref


class TestIsRelativeRef:
    """Test _is_relative_ref detects cross-parameter references."""

    def test_yu_self_yizhi(self):
        assert _is_relative_ref("与self一致") is True

    def test_yu_backtick_self_yizhi(self):
        assert _is_relative_ref("与`self`一致") is True

    def test_tong_input_yizhi(self):
        assert _is_relative_ref("同input一致") is True

    def test_yu_x_xiangtong(self):
        assert _is_relative_ref("与x相同") is True

    def test_he_self_yiyang(self):
        assert _is_relative_ref("和self一样") is True

    def test_yu_self_baochi_yizhi(self):
        assert _is_relative_ref("与self保持一致") is True

    def test_valid_dtype_not_ref(self):
        assert _is_relative_ref("FLOAT32") is False
        assert _is_relative_ref("FLOAT16、BFLOAT16") is False

    def test_valid_shape_not_ref(self):
        assert _is_relative_ref("1-8") is False
        assert _is_relative_ref("(N,C,H,W)") is False

    def test_valid_format_not_ref(self):
        assert _is_relative_ref("ND") is False
        assert _is_relative_ref("ND、NZ") is False

    def test_empty_string(self):
        assert _is_relative_ref("") is False

    def test_dash(self):
        assert _is_relative_ref("-") is False


class TestExtract4ColumnsRelativeRef:
    """Ensure relative references in dtype/shape/dformat are cleared."""

    def test_dtype_relative_ref_cleared(self):
        header = ["参数名", "数据类型", "数据格式", "维度(shape)", "非连续Tensor"]
        grid = [header, ["gradOutput", "与`self`一致", "ND", "-", "√"]]
        col_map = detect_table_columns(header)
        name_idx = find_param_name_column(header)
        result = extract_4_columns_from_table(grid, col_map, "gradOutput", "aclTensor", name_idx)
        # dtype should be cleared (relative ref)
        assert result.get("dtype_desc") is None or result.get("dtype_desc") == ""
        # dformat "ND" is a real value — should be preserved
        assert result["dformat_desc"] == "ND"
        # shape "-" → empty
        assert result.get("shape") is None or result.get("shape") == ""
        # discontinuous should work normally
        disc = json.loads(result["is_support_discontinuous"])
        assert disc["value"] is True

    def test_shape_relative_ref_cleared(self):
        header = ["参数名", "数据类型", "维度(shape)"]
        grid = [header, ["target", "FLOAT32", "与self一致"]]
        col_map = detect_table_columns(header)
        name_idx = find_param_name_column(header)
        result = extract_4_columns_from_table(grid, col_map, "target", "aclTensor", name_idx)
        # dtype "FLOAT32" is real — preserved
        assert result["dtype_desc"] == "FLOAT32"
        # shape "与self一致" should be cleared
        assert result.get("shape") is None or result.get("shape") == ""

    def test_param_desc_relative_ref_preserved(self):
        """param_desc should NOT be cleared — it's useful context for LLM."""
        header = ["参数名", "数据类型", "描述"]
        grid = [header, ["out", "与self一致", "shape与self相同"]]
        col_map = detect_table_columns(header)
        name_idx = find_param_name_column(header)
        result = extract_4_columns_from_table(grid, col_map, "out", "aclTensor", name_idx)
        # dtype cleared (relative ref)
        assert result.get("dtype_desc") is None or result.get("dtype_desc") == ""
        # param_desc preserved (even if it contains relative ref language)
        assert result["param_desc"] == "shape与self相同"


# ---------------------------------------------------------------------------
# Platform-tagged value extraction (term + nested ul/li)
# ---------------------------------------------------------------------------

from agent.utils.table_parser import (
    extract_columns_as_json,
    extract_platform_tagged_values,
    parse_html_tables_with_raw,
)


class TestExtractPlatformTaggedValues:
    """Test extract_platform_tagged_values handles all HTML patterns."""

    def test_li_pattern_term_inside_li(self):
        """<li><term>PLATFORM</term>：VALUE</li> — the common list pattern."""
        html = (
            "<li><term>Atlas A2 训练系列产品</term>：[M, K1]</li>"
            "<li><term>Atlas 推理系列加速卡产品</term>：[M, K1]</li>"
        )
        result = extract_platform_tagged_values(html)
        assert result == {
            "Atlas A2 训练系列产品": "[M, K1]",
            "Atlas 推理系列加速卡产品": "[M, K1]",
        }

    def test_inline_pattern_simple_value(self):
        """<term>PLATFORM</term>：VALUE without nested list."""
        html = "<term>Atlas A2 训练系列产品</term>：输入在有/无专家时分别为[E, N2]/[N2]"
        result = extract_platform_tagged_values(html)
        assert result == {
            "Atlas A2 训练系列产品": "输入在有/无专家时分别为[E, N2]/[N2]",
        }

    def test_inline_pattern_nested_ul_multiple_li(self):
        """<term>PLATFORM</term>：<ul><li>..</li><li>..</li></ul> — the bug case.

        Regression: the value must include ALL <li> items, not just the first.
        """
        html = (
            "<term>Atlas A2 训练系列产品</term>："
            "<ul>"
            "<li>per-channel下输入在有/无专家时分别为[E, N2]/[N2]</li>"
            "<li>per-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]</li>"
            "</ul>"
        )
        result = extract_platform_tagged_values(html)
        assert result is not None
        assert "Atlas A2 训练系列产品" in result
        value = result["Atlas A2 训练系列产品"]
        # Both per-channel and per-group must be present
        assert "per-channel" in value
        assert "per-group" in value
        assert "[E, G, N2]" in value
        assert "[G, N2]" in value

    def test_inline_pattern_nested_ul_three_li(self):
        """Three nested <li> items must all be captured."""
        html = (
            "<term>Atlas A2</term>："
            "<ul>"
            "<li>per-tensor：[E]</li>"
            "<li>per-channel：[E, N1]</li>"
            "<li>per-group：[E, G, N1]</li>"
            "</ul>"
        )
        result = extract_platform_tagged_values(html)
        value = result["Atlas A2"]
        assert "per-tensor" in value
        assert "per-channel" in value
        assert "per-group" in value

    def test_no_platform_tag_returns_none(self):
        """Universal value (no <term> tags) returns None."""
        assert extract_platform_tagged_values("FLOAT16、BFLOAT16") is None
        assert extract_platform_tagged_values("") is None


class TestExtractColumnsJsonNestedList:
    """End-to-end: extract_columns_as_json with nested <ul><li> shape values."""

    NESTED_LIST_TABLE_HTML = """\
<table>
<tr>
  <th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th>
  <th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th>
</tr>
<tr>
  <td>antiquantOffset2Optional（aclTensor*）</td>
  <td>可选输入</td>
  <td>伪量化参数，第二个matmul的偏移量。</td>
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td>
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：<ul><li>per-channel下输入在有/无专家时分别为[E, N2]/[N2]</li><li>per-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]</li></ul></td>
  <td>√</td>
</tr>
</table>
"""

    def test_nested_ul_shape_not_truncated(self):
        """Regression: shape with nested <ul><li> must keep ALL list items."""
        tables, raw_tables = parse_html_tables_with_raw(self.NESTED_LIST_TABLE_HTML)
        grid = tables[0]
        raw_grid = raw_tables[0]
        col_map = detect_table_columns(grid[0])
        name_idx = find_param_name_column(grid[0])

        result = extract_columns_as_json(
            grid, raw_grid, col_map, "antiquantOffset2Optional", "aclTensor*", name_idx
        )

        # Shape must be platform-keyed with FULL content
        assert "shape" in result
        shape = result["shape"]
        platform = "Atlas A2 训练系列产品/Atlas A2 推理系列产品"
        assert platform in shape
        value = shape[platform]
        # Both per-channel and per-group present (the bug truncated at first </li>)
        assert "per-channel" in value
        assert "[E, N2]" in value and "[N2]" in value
        assert "per-group" in value
        assert "[E, G, N2]" in value and "[G, N2]" in value

    def test_dtype_and_usage_extracted_correctly(self):
        """dtype and usage_notes columns should also parse platform tags."""
        tables, raw_tables = parse_html_tables_with_raw(self.NESTED_LIST_TABLE_HTML)
        grid = tables[0]
        raw_grid = raw_tables[0]
        col_map = detect_table_columns(grid[0])
        name_idx = find_param_name_column(grid[0])

        result = extract_columns_as_json(
            grid, raw_grid, col_map, "antiquantOffset2Optional", "aclTensor*", name_idx
        )

        assert result["dtype_desc"] == {
            "Atlas A2 训练系列产品/Atlas A2 推理系列产品": "FLOAT16、BFLOAT16",
        }
        assert result["usage_notes"] == {
            "Atlas 推理系列加速卡产品": "只支持传空指针。",
        }
        assert result["dformat_desc"] == {"*": "ND"}
        assert result["direction"] == "input"
