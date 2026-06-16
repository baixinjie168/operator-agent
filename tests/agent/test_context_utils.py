"""Tests for nodes.context_utils — shared pre-filtering utilities."""

from agent.nodes.context_utils import (
    ParamRow,
    _is_ws_function,
    _param_matches,
    extract_param_context,
    format_param_context,
    parse_param_table,
)


# ---------------------------------------------------------------------------
# _param_matches
# ---------------------------------------------------------------------------


class TestParamMatches:
    def test_exact_match(self):
        assert _param_matches("param x is required", "x") is True

    def test_no_match(self):
        assert _param_matches("param y is required", "x") is False

    def test_word_boundary_prevents_partial(self):
        # "x_ref" should not match "x"
        assert _param_matches("param x_ref is required", "x") is False

    def test_match_at_start_of_line(self):
        assert _param_matches("x is the input", "x") is True

    def test_match_at_end_of_line(self):
        assert _param_matches("the parameter is x", "x") is True

    def test_match_in_table_row(self):
        assert _param_matches("| x | aclTensor | input tensor |", "x") is True

    def test_no_match_partial_in_table(self):
        assert _param_matches("| x_ref | aclTensor | ref tensor |", "x") is False

    def test_underscore_boundary(self):
        assert _param_matches("attr_name", "name") is False
        assert _param_matches("attr name", "name") is True

    def test_digit_boundary(self):
        assert _param_matches("dim1value", "dim") is False
        assert _param_matches("dim = 1", "dim") is True


# ---------------------------------------------------------------------------
# extract_param_context
# ---------------------------------------------------------------------------


class TestExtractParamContext:
    SAMPLE_SECTION = """\
## params_execute

| 参数名 | 数据类型 | 说明 |
| --- | --- | --- |
| x | aclTensor | 输入张量 |
| y | aclTensor | 第二个输入张量 |
| alpha | float32 | 缩放因子 |
| workspace | void * | 工作空间 |

Some extra paragraph about x that spans multiple lines.
The tensor x must be contiguous.

## return_codes_execute

| 返回码 | 说明 |
| --- | --- |
| ACL_SUCCESS | 成功 |
| ACL_ERROR | x 格式不正确 |
"""

    def test_filters_to_relevant_lines(self):
        ctx = extract_param_context(self.SAMPLE_SECTION, "alpha")
        # alpha's row should be present, x's paragraph should not
        assert "alpha" in ctx
        assert "缩放因子" in ctx

    def test_preserves_table_header(self):
        ctx = extract_param_context(self.SAMPLE_SECTION, "alpha")
        # Table header should be preserved
        assert "参数名" in ctx
        assert "| --- |" in ctx

    def test_adjacent_lines_included(self):
        ctx = extract_param_context(self.SAMPLE_SECTION, "alpha")
        # ±2 lines around alpha's row should be included
        assert "y" in ctx or "workspace" in ctx

    def test_fallback_to_full_text_when_no_match(self):
        ctx = extract_param_context(self.SAMPLE_SECTION, "nonexistent_param")
        assert ctx == self.SAMPLE_SECTION

    def test_fallback_when_result_too_short(self):
        short_section = "Just one line mentioning p"
        ctx = extract_param_context(short_section, "p")
        # Result < 100 chars → fallback to full text
        assert ctx == short_section

    def test_excludes_unrelated_content(self):
        ctx = extract_param_context(self.SAMPLE_SECTION, "alpha")
        # The paragraph about "x must be contiguous" should not appear
        # (unless alpha appears in it, which it doesn't)
        assert "must be contiguous" not in ctx

    def test_html_tr_row_boundary_expansion(self):
        """Phase 1.5: if matched line is inside <tr>, expand to full <tr>...</tr>."""
        # Pad alpha's <tr> with enough lines so ±2 window does NOT reach beta's <tr>
        alpha_row = (
            "<tr>\n"
            "  <td>alpha</td>\n"
            "  <td>first row mentioning alpha and some other info</td>\n"
            "  <td>extra col 1</td>\n"
            "  <td>extra col 2</td>\n"
            "  <td>extra col 3</td>\n"
            "</tr>"
        )
        section = (
            "<table>\n"
            + alpha_row + "\n"
            "<tr>\n"
            "  <td>beta</td>\n"
            "  <td>second row about beta only</td>\n"
            "</tr>\n"
            "</table>\n"
            "\n"
            "Some trailing paragraph about gamma."
        )
        ctx = extract_param_context(section, "alpha")
        # The entire <tr> containing alpha should be included
        assert "alpha" in ctx
        assert "<tr>" in ctx
        assert "</tr>" in ctx
        # beta's <tr> is far enough that ±2 window won't reach it;
        # Phase 1.5 should NOT pull beta in since it's in a separate <tr>
        assert "beta" not in ctx
        # Trailing paragraph should not appear
        assert "gamma" not in ctx


# ---------------------------------------------------------------------------
# parse_param_table
# ---------------------------------------------------------------------------


class TestParseParamTable:
    def test_parses_markdown_table(self):
        text = """\
| 参数名 | 数据类型 | 说明 |
| --- | --- | --- |
| x | aclTensor | 输入张量 |
| alpha | float32 | 缩放因子 |
"""
        rows = parse_param_table(text)
        assert len(rows) == 2
        assert rows[0].param_name == "x"
        assert rows[0].param_type == "aclTensor"
        assert rows[0].description == "输入张量"
        assert rows[1].param_name == "alpha"

    def test_skips_header_only(self):
        text = """\
| 参数名 | 数据类型 | 说明 |
| --- | --- | --- |
"""
        rows = parse_param_table(text)
        assert rows == []

    def test_handles_empty_text(self):
        rows = parse_param_table("")
        assert rows == []

    def test_handles_two_columns(self):
        text = """\
| 参数名 | 说明 |
| --- | --- |
| x | input |
"""
        rows = parse_param_table(text)
        assert len(rows) == 1
        assert rows[0].param_name == "x"
        assert rows[0].param_type == "input"
        assert rows[0].description == ""


# ---------------------------------------------------------------------------
# format_param_context
# ---------------------------------------------------------------------------


class TestFormatParamContext:
    def test_basic_format(self):
        row = ParamRow("x", "aclTensor", "input tensor", "| x | aclTensor | input tensor |")
        result = format_param_context(row, [])
        assert "参数: x" in result
        assert "类型: aclTensor" in result
        assert "描述: input tensor" in result

    def test_with_extra_paragraphs(self):
        row = ParamRow("x", "aclTensor", "input", "| x | aclTensor | input |")
        result = format_param_context(row, ["x must be contiguous"])
        assert "相关约束：" in result
        assert "x must be contiguous" in result


# ---------------------------------------------------------------------------
# _is_ws_function
# ---------------------------------------------------------------------------


class TestIsWsFunction:
    def test_get_workspace_size(self):
        assert _is_ws_function("aclnnFooGetWorkspaceSize") is True

    def test_execute_function(self):
        assert _is_ws_function("aclnnFoo") is False

    def test_empty(self):
        assert _is_ws_function("") is False
