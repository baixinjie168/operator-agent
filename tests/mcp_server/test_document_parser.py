"""Tests for the document_parser module."""

from __future__ import annotations

from mcp_server.parsers.document_parser import (
    classify_section,
    parse_function_signatures,
    parse_operator_file,
    parse_product_support_table,
    parse_table_rows,
)
from shared.models.enums import SectionType


class TestClassifySection:
    def test_product_support(self):
        assert classify_section("产品支持情况", "aclnnFoo") == SectionType.PRODUCT_SUPPORT

    def test_function_description(self):
        assert classify_section("功能说明", "aclnnFoo") == SectionType.FUNCTION_DESCRIPTION

    def test_function_prototype(self):
        assert classify_section("函数原型", "aclnnFoo") == SectionType.FUNCTION_PROTOTYPE

    def test_get_workspace_size(self):
        assert classify_section("aclnnFooGetWorkspaceSize", "aclnnFoo") == SectionType.GET_WORKSPACE_SIZE

    def test_execute_api(self):
        assert classify_section("aclnnFoo", "aclnnFoo") == SectionType.EXECUTE_API

    def test_constraints(self):
        assert classify_section("约束说明", "aclnnFoo") == SectionType.CONSTRAINTS

    def test_usage_example(self):
        assert classify_section("调用示例", "aclnnFoo") == SectionType.USAGE_EXAMPLE

    def test_unknown(self):
        assert classify_section("Some random heading", "aclnnFoo") == SectionType.UNKNOWN


class TestProductSupportTable:
    def test_parses_basic_table(self):
        text = (
            "| 产品 | 是否支持 |\n"
            "|---|---|\n"
            "| Atlas A2 训练系列产品 | √ |\n"
            "| Atlas 推理系列产品 | × |\n"
        )
        results = parse_product_support_table(text)
        assert len(results) == 2
        assert results[0].product == "Atlas A2 训练系列产品"
        assert results[0].supported is True
        assert results[1].product == "Atlas 推理系列产品"
        assert results[1].supported is False

    def test_empty_table(self):
        assert parse_product_support_table("") == []


class TestFunctionSignatures:
    def test_parses_c_signature_from_code_block(self):
        text = (
            "```\n"
            "aclnnStatus aclnnTestGetWorkspaceSize(\n"
            "  const aclTensor* input,\n"
            "  uint64_t* workspaceSize)\n"
            "```\n"
        )
        results = parse_function_signatures(text)
        assert len(results) == 1
        assert results[0].function_name == "aclnnTestGetWorkspaceSize"
        assert results[0].return_type == "aclnnStatus"
        assert len(results[0].parameters) == 2

    def test_parses_multiple_code_blocks(self):
        text = (
            "```\n"
            "aclnnStatus aclnnTestGetWorkspaceSize(\n"
            "  const aclTensor* input,\n"
            "  uint64_t* workspaceSize)\n"
            "```\n"
            "```\n"
            "aclnnStatus aclnnTest(\n"
            "  void* workspace,\n"
            "  uint64_t workspaceSize)\n"
            "```\n"
        )
        results = parse_function_signatures(text)
        assert len(results) == 2
        assert results[0].function_name == "aclnnTestGetWorkspaceSize"
        assert results[1].function_name == "aclnnTest"

    def test_no_code_blocks(self):
        assert parse_function_signatures("just text") == []


class TestTableRows:
    def test_parses_simple_table(self):
        text = (
            "| 参数名 | 输入/输出 | 描述 |\n"
            "|---|---|---|\n"
            "| input | 输入 | 测试输入 |\n"
            "| output | 输出 | 测试输出 |\n"
        )
        rows = parse_table_rows(text)
        assert len(rows) == 2
        assert rows[0]["参数名"] == "input"
        assert rows[1]["输入/输出"] == "输出"

    def test_handles_continuation_rows(self):
        text = (
            "| 参数名 | 描述 |\n"
            "|---|---|\n"
            "| input | 基本描述 |\n"
            "|  | 额外说明 |\n"
        )
        rows = parse_table_rows(text)
        assert len(rows) == 1
        assert "基本描述" in rows[0]["描述"]
        assert "额外说明" in rows[0]["描述"]


class TestFullDocumentParsing:
    def test_real_operator_document(self, sample_operator_path):
        result = parse_operator_file(sample_operator_path)

        assert result.operator_name == "aclnnBatchNormElemt"
        assert result.cann_version == "9.0.0"
        assert result.source_url is not None
        assert "hiascend.com" in result.source_url
        assert result.saved_date == "2026-05-13"

        # H2 sections + H3 sub-sections (params/return_codes)
        assert len(result.sections) == 12
        section_types = [s.section_type for s in result.sections]
        assert SectionType.PRODUCT_SUPPORT in section_types
        assert SectionType.FUNCTION_PROTOTYPE in section_types
        assert SectionType.GET_WORKSPACE_SIZE in section_types
        assert SectionType.PARAMS_GET_WORKSPACE in section_types
        assert SectionType.RETURN_CODES_GET_WORKSPACE in section_types
        assert SectionType.PARAMS_EXECUTE in section_types
        assert SectionType.CONSTRAINTS in section_types
        assert SectionType.USAGE_EXAMPLE in section_types

    def test_minimal_operator_document(self, minimal_operator_path):
        result = parse_operator_file(minimal_operator_path)

        assert result.operator_name == "aclnnTestOp"
        assert result.cann_version == "8.0.0"

        assert len(result.sections) == 11

    def test_serialization_roundtrip(self, sample_operator_path):
        import json

        result = parse_operator_file(sample_operator_path)
        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["operator_name"] == "aclnnBatchNormElemt"
        assert len(data["sections"]) == 12

        from shared.models.operator import ParsedOperatorDocument

        restored = ParsedOperatorDocument.model_validate(data)
        assert restored.operator_name == result.operator_name
        assert len(restored.sections) == len(result.sections)
