"""Tests for chunked_extract.py: split_into_chunks + extract_relations_chunked."""

import pytest

from agent.nodes.param_relation_extract.chunked_extract import (
    _is_param_list_item,
    _is_nested_item,
    _merge_small_blocks,
    split_into_chunks,
)


class TestIsParamListItem:
    def test_matches_chinese_parens(self):
        line = "  - permutedTokenOutputGrad（aclTensor *，计算输入）：正向输出"
        assert _is_param_list_item(line)

    def test_matches_english_parens(self):
        line = "  - self(aclTensor*, input): target tensor"
        assert _is_param_list_item(line)

    def test_rejects_non_param_line(self):
        assert not _is_param_list_item("- **返回值：**")

    def test_rejects_nested_item(self):
        assert not _is_param_list_item("    - <term>Atlas A2</term>：支持")

    def test_rejects_plain_prose(self):
        assert not _is_param_list_item("这是一段普通的说明文字")

    def test_matches_int_type(self):
        line = "  - experts_num（int64_t，计算输入）：表示参与运算的专家个数。"
        assert _is_param_list_item(line)


class TestIsNestedItem:
    def test_nested_dash(self):
        line = "    - <term>Atlas A2</term>：数据类型支持 FLOAT16"
        assert _is_nested_item(line)

    def test_top_level_not_nested(self):
        line = "  - self(aclTensor*, input): target"
        assert not _is_nested_item(line)

    def test_plain_line_not_nested(self):
        line = "    数据类型支持 FLOAT16、FLOAT32"
        assert not _is_nested_item(line)


class TestSplitIntoChunks:
    def test_empty_text(self):
        assert split_into_chunks("") == []

    def test_plain_prose_split_by_blank_line(self):
        # Make paragraphs large enough to not be merged
        p1 = "第一段内容" + "x" * 1600
        p2 = "第二段内容" + "y" * 1600
        text = f"{p1}\n\n{p2}"
        chunks = split_into_chunks(text)
        assert len(chunks) == 2
        assert "第一段" in chunks[0]
        assert "第二段" in chunks[1]

    def test_html_table_as_single_block(self):
        text = "<table>\n<tr><th>A</th></tr>\n<tr><td>1</td></tr>\n</table>"
        chunks = split_into_chunks(text)
        assert len(chunks) == 1
        assert "<table>" in chunks[0]

    def test_markdown_table_as_single_block(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        chunks = split_into_chunks(text)
        assert len(chunks) == 1

    def test_param_list_items_split(self):
        # Pad with ASCII to exceed 1500 bytes (ASCII = 1 byte/char)
        text = (
            "  - self（aclTensor*，输入）：目标张量 " + "A" * 1500 + "\n"
            "    数据类型支持 FLOAT16\n"
            "  - size（aclIntArray*，输入）：广播尺寸 " + "B" * 1500 + "\n"
            "  - out（aclTensor*，输出）：输出张量 " + "C" * 1500 + "\n"
        )
        chunks = split_into_chunks(text)
        assert len(chunks) == 3
        assert "self" in chunks[0]
        assert "size" in chunks[1]
        assert "out" in chunks[2]

    def test_param_list_with_nested_items(self):
        text = (
            "  - self（aclTensor*，输入）：目标张量 " + "A" * 1500 + "\n"
            "    - <term>Atlas A2</term>：数据类型支持 FLOAT16\n"
            "    - <term>Atlas A3</term>：数据类型支持 BFLOAT16\n"
            "  - size（aclIntArray*，输入）：广播尺寸 " + "B" * 1500 + "\n"
        )
        chunks = split_into_chunks(text)
        assert len(chunks) == 2
        # First chunk should contain both nested items
        assert "Atlas A2" in chunks[0]
        assert "Atlas A3" in chunks[0]
        assert "size" in chunks[1]

    def test_mixed_html_table_and_prose(self):
        prose1 = "## 参数说明，" + "详细说明" * 200
        prose2 = "约束说明段落，" + "约束描述" * 200
        text = (
            f"{prose1}\n\n"
            "<table>\n<tr><th>A</th></tr>\n<tr><td>1</td></tr>\n</table>\n\n"
            f"{prose2}"
        )
        chunks = split_into_chunks(text)
        assert len(chunks) >= 2

    def test_small_blocks_merged(self):
        text = "短\n\n文\n\n本"
        chunks = split_into_chunks(text)
        # All small paragraphs should be merged into one
        assert len(chunks) == 1

    def test_long_html_table_split_with_header_preserved(self):
        """HTML table with >15 <tr> should split by 10-row groups, each keeping header."""
        header = "<table>\n<tr><th>参数名</th><th>说明</th></tr>"
        rows = [f"<tr><td>p{i}</td><td>desc{i}{'x' * 80}</td></tr>" for i in range(20)]
        footer = "</table>"
        text = header + "\n" + "\n".join(rows) + "\n" + footer
        chunks = split_into_chunks(text)
        # 20 rows → 2 groups (0-9, 10-19), each with header
        assert len(chunks) == 2
        # Both groups should contain the header
        for chunk in chunks:
            assert "<th>参数名</th>" in chunk
        # First group should contain p0..p9, second p10..p19
        assert "p0" in chunks[0]
        assert "p9" in chunks[0]
        assert "p10" in chunks[1]
        assert "p19" in chunks[1]

    def test_long_markdown_table_split_with_header_preserved(self):
        """MD table with >15 data rows should split by 10-row groups, each keeping header."""
        header = "| 参数名 | 说明 |\n| --- | --- |"
        data_rows = [f"| p{i} | desc{i}{'x' * 80} |" for i in range(20)]
        text = header + "\n" + "\n".join(data_rows)
        chunks = split_into_chunks(text)
        # 20 data rows → 2 groups, each with header (2 lines)
        assert len(chunks) == 2
        for chunk in chunks:
            assert "参数名" in chunk
            assert "| --- |" in chunk
        assert "p0" in chunks[0]
        assert "p10" in chunks[1]


class TestMergeSmallBlocks:
    def test_merges_small_blocks(self):
        blocks = [["hello"], ["world"]]
        result = _merge_small_blocks(blocks, max_size=1500)
        assert len(result) == 1
        assert "hello" in result[0] and "world" in result[0]

    def test_keeps_large_blocks_separate(self):
        big = "x" * 2000
        blocks = [["small"], [big]]
        result = _merge_small_blocks(blocks, max_size=1500)
        assert len(result) == 2

    def test_skips_empty_blocks(self):
        blocks = [["hello"], ["   "], ["world"]]
        result = _merge_small_blocks(blocks, max_size=1500)
        assert len(result) == 1
