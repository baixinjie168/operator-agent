"""Fix 1C: aclIntArray dtype extraction regression test.

Isolates parsing vs section delivery for the aclnnCalculateMatmulWeightSize
dtype=int defect. The buggy JSON had tensorShape (aclIntArray) dtype=["int"]
(Level-3 fallback) because dtype_desc (FLOAT16、BFLOAT16 from the table's
「数据类型」列) never reached Level-1. This test verifies the *parser* can
extract dtype_desc from an aclIntArray row — if it passes, the断点 is in
section delivery (Fix 1D: get_section returns only the first same-type
section, masking later ones); if it fails, the parser needs fixing.

Note: tests/agent/test_table_column_extract.py has a pre-existing stale
import (extract_4_columns_from_table, renamed to extract_columns_as_json)
and cannot be collected. This file is self-contained on the current API so
Fix 1C is verifiable independently.
"""

from __future__ import annotations

from agent.utils.table_parser import (
    detect_table_columns,
    extract_columns_as_json,
    find_param_name_column,
    parse_html_tables_with_raw,
)

# Mirrors the aclnnCalculateMatmulWeightSize tensorShape row: 8-column
# GetWorkspaceSize-style table, aclIntArray param with a FLOAT16、BFLOAT16
# 「数据类型」cell and a 2-6 「维度(shape)」cell.
INT_ARRAY_TABLE_HTML = """\
<table>
<tr>
  <th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th>
  <th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th>
</tr>
<tr>
  <td>tensorShape（aclIntArray*）</td><td>输入</td>
  <td>权重矩阵的shape。</td>
  <td>输入shape支持2-6维，即（batch，n，k），不支持空Array。</td>
  <td>FLOAT16、BFLOAT16</td><td>-</td><td>2-6</td><td>-</td>
</tr>
</table>
"""


def _extract(param_name: str = "tensorShape", ptype: str = "aclIntArray*") -> dict:
    tables, raw_tables = parse_html_tables_with_raw(INT_ARRAY_TABLE_HTML)
    grid = tables[0]
    raw_grid = raw_tables[0]
    col_map = detect_table_columns(grid[0])
    name_idx = find_param_name_column(grid[0])
    return extract_columns_as_json(
        grid, raw_grid, col_map, param_name, ptype, name_idx,
    )


class TestExtractIntArrayDtype:
    """Fix 1C: parser must extract dtype from an aclIntArray row."""

    def test_dtype_extracted_from_int_array_row(self):
        """aclIntArray 行的「数据类型」列 FLOAT16、BFLOAT16 能被解析。"""
        out = _extract()
        assert out.get("dtype_desc") == {"*": "FLOAT16、BFLOAT16"}, out

    def test_shape_extracted_from_int_array_row(self):
        """aclIntArray 行的「维度(shape)」列 2-6 能被解析。"""
        out = _extract()
        assert out.get("shape") == {"*": "2-6"}, out

    def test_usage_notes_extracted(self):
        """「使用说明」列（含 2-6维 描述）能被解析。"""
        out = _extract()
        assert "2-6维" in out.get("usage_notes", {}).get("*", ""), out
