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
        assert result["shape"] == ""  # dash → empty
        assert result["dtype_desc"] == "INT64"
        assert result["dformat_desc"] == "ND、NZ"
        disc = json.loads(result["is_support_discontinuous"])
        assert disc["value"] is False

    def test_non_tensor_param(self):
        grid, col_map, name_idx = self._get_grid_and_map()
        result = extract_4_columns_from_table(grid, col_map, "offset", "uint64_t", name_idx)
        assert result["shape"] == "1"
        assert result["dformat_desc"] == ""  # dash → empty
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
