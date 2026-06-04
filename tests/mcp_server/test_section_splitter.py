"""Tests for the section_splitter module."""

from __future__ import annotations

import pytest

from mcp_server.parsers.section_splitter import split_sections
from shared.exceptions import DocumentParsingError
from shared.models.enums import SectionType


class TestTitleParsing:
    def test_extracts_operator_name_and_version(self):
        md = "# aclnnBatchNormElemt-CANN社区版9.0.0-昇腾社区\n\n## 产品支持情况\n\ncontent"
        header, _ = split_sections(md)
        assert header.operator_name == "aclnnBatchNormElemt"
        assert header.cann_version == "9.0.0"

    def test_rejects_invalid_title(self):
        md = "# Some random title\n\n## Section\n\ncontent"
        with pytest.raises(DocumentParsingError, match="Cannot parse operator title"):
            split_sections(md)


class TestMetadataExtraction:
    def test_extracts_url_and_saved_date(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n"
            "\n"
            "**URL:** https://example.com/foo.md\n"
            "\n"
            "**Saved:** 2026-05-14\n"
            "\n"
            "---\n"
            "\n"
            "## Section\ncontent"
        )
        header, _ = split_sections(md)
        assert header.source_url == "https://example.com/foo.md"
        assert header.saved_date == "2026-05-14"

    def test_works_without_metadata(self):
        md = "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n## Section\ncontent"
        header, _ = split_sections(md)
        assert header.source_url is None
        assert header.saved_date is None


class TestH2Splitting:
    def test_splits_by_h2_headings(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n"
            "## Section A\n\ncontent a\n\n"
            "## Section B\n\ncontent b"
        )
        _, sections = split_sections(md)
        assert len(sections) == 2
        assert sections[0].heading == "Section A"
        assert sections[1].heading == "Section B"

    def test_handles_detached_h2_heading(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n"
            "## \n\n产品支持情况\n\ncontent"
        )
        _, sections = split_sections(md)
        assert len(sections) == 1
        assert sections[0].heading == "产品支持情况"

    def test_tracks_line_numbers(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n"
            "---\n"
            "## Alpha\n\nalpha content\n\n"
            "## Beta\n\nbeta content"
        )
        _, sections = split_sections(md)
        assert sections[0].line_start >= 0
        assert sections[0].line_end >= sections[0].line_start
        assert sections[1].line_start > sections[0].line_start


class TestFooterStripping:
    def test_strips_markdown_extracted_by(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n"
            "## Section\ncontent\n\n"
            "---\n\n"
            "*Markdown extracted by Some Extension*\n"
        )
        _, sections = split_sections(md)
        assert len(sections) == 1
        assert "extracted by" not in sections[0].body_text


class TestH3SubSections:
    def test_splits_params_and_return_codes_in_get_workspace(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n"
            "## aclnnFooGetWorkspaceSize\n\nintro\n\n"
            "-   **参数说明**\n\nparam content here\n\n"
            "-   **返回值**\n\nreturn code content here\n"
        )
        _, sections = split_sections(md)
        types = [s.section_type for s in sections]
        assert SectionType.GET_WORKSPACE_SIZE in types
        assert SectionType.PARAMS_GET_WORKSPACE in types
        assert SectionType.RETURN_CODES_GET_WORKSPACE in types

    def test_classifies_execute_h3_sub_sections(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n"
            "## aclnnFoo\n\nintro\n\n"
            "-   **参数说明**\n\nexec param content\n\n"
            "-   **返回值**\n\nexec return content\n"
        )
        _, sections = split_sections(md)
        types = [s.section_type for s in sections]
        assert SectionType.PARAMS_EXECUTE in types
        assert SectionType.RETURN_CODES_EXECUTE in types

    def test_h3_sections_have_level_3(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n"
            "## aclnnFooGetWorkspaceSize\n\nintro\n\n"
            "-   **参数说明**\n\nparam content\n\n"
            "-   **返回值**\n\nreturn content\n"
        )
        _, sections = split_sections(md)
        h3_sections = [s for s in sections if s.level == 3]
        assert len(h3_sections) == 2
        for s in h3_sections:
            assert s.section_type in (SectionType.PARAMS_GET_WORKSPACE, SectionType.RETURN_CODES_GET_WORKSPACE)


class TestEdgeCases:
    def test_empty_document_raises(self):
        with pytest.raises(DocumentParsingError, match="Document is empty"):
            split_sections("")

    def test_single_section(self):
        md = "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n---\n## Only\n\njust content"
        _, sections = split_sections(md)
        assert len(sections) == 1
        assert sections[0].heading == "Only"

    def test_skips_empty_pre_content_section(self):
        md = (
            "# aclnnFoo-CANN社区版1.0.0-昇腾社区\n"
            "---\n"
            "\n"
            "## First\n\ncontent"
        )
        _, sections = split_sections(md)
        assert len(sections) == 1
        assert sections[0].heading == "First"

    def test_real_operator_document(self, sample_operator_path):
        content = sample_operator_path.read_text(encoding="utf-8")
        header, sections = split_sections(content)
        assert header.operator_name == "aclnnBatchNormElemt"
        assert header.cann_version == "9.0.0"
        # H2 sections + H3 sub-sections (params/return_codes for both APIs)
        assert len(sections) == 12

        types = [s.section_type for s in sections]
        assert SectionType.PRODUCT_SUPPORT in types
        assert SectionType.FUNCTION_DESCRIPTION in types
        assert SectionType.FUNCTION_PROTOTYPE in types
        assert SectionType.PARAMS_GET_WORKSPACE in types
        assert SectionType.RETURN_CODES_GET_WORKSPACE in types
        assert SectionType.PARAMS_EXECUTE in types
        assert SectionType.RETURN_CODES_EXECUTE in types
        assert SectionType.CONSTRAINTS in types
        assert SectionType.USAGE_EXAMPLE in types

    def test_minimal_operator_document(self, minimal_operator_path):
        content = minimal_operator_path.read_text(encoding="utf-8")
        header, sections = split_sections(content)
        assert header.operator_name == "aclnnTestOp"
        assert header.cann_version == "8.0.0"
        # H2 sections + H3 sub-sections
        assert len(sections) == 11

        types = [s.section_type for s in sections]
        assert SectionType.PRODUCT_SUPPORT in types
        assert SectionType.FUNCTION_DESCRIPTION in types
        assert SectionType.FUNCTION_PROTOTYPE in types
        assert SectionType.PARAMS_GET_WORKSPACE in types
        assert SectionType.RETURN_CODES_GET_WORKSPACE in types
        assert SectionType.PARAMS_EXECUTE in types
        assert SectionType.CONSTRAINTS in types
        assert SectionType.USAGE_EXAMPLE in types
