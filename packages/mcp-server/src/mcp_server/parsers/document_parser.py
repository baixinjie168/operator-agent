"""Parse CANN operator Markdown documents into structured data.

Uses section_splitter to split raw Markdown, then classifies sections
and extracts structured data (tables, code blocks) from each section.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from mcp_server.parsers.section_splitter import split_sections
from shared.models.enums import SectionType
from shared.models.operator import (
    FunctionSignature,
    ParsedOperatorDocument,
    ParsedSection,
    ProductSupport,
)

logger = logging.getLogger(__name__)

# Section classification patterns
SECTION_PATTERNS: list[tuple[re.Pattern[str], SectionType]] = [
    (re.compile(r"产品支持"), SectionType.PRODUCT_SUPPORT),
    (re.compile(r"功能说明"), SectionType.FUNCTION_DESCRIPTION),
    (re.compile(r"函数原型"), SectionType.FUNCTION_PROTOTYPE),
    (re.compile(r"约束"), SectionType.CONSTRAINTS),
    (re.compile(r"调用示例"), SectionType.USAGE_EXAMPLE),
]

# C function signature pattern
FUNC_SIG_PATTERN = re.compile(
    r"^(?P<return_type>\w+)\s+(?P<name>\w+)\s*\((?P<params>[\s\S]*)\)",
    re.MULTILINE,
)

# Table row pattern
TABLE_ROW_PATTERN = re.compile(r"^\|\s*(.+?)\s*\|$")
TABLE_SEPARATOR_PATTERN = re.compile(r"^[\|\s\-:]+$")

# Product support symbols
SUPPORTED_SYMBOLS = {"√", "✓", "是"}
UNSUPPORTED_SYMBOLS = {"×", "✗", "否", "-"}


def classify_section(heading: str, operator_name: str) -> SectionType:
    """Classify a section heading into a SectionType."""
    heading_stripped = heading.strip()

    if heading_stripped.endswith("GetWorkspaceSize"):
        return SectionType.GET_WORKSPACE_SIZE

    for pattern, section_type in SECTION_PATTERNS:
        if pattern.search(heading_stripped):
            return section_type

    if heading_stripped == operator_name:
        return SectionType.EXECUTE_API

    return SectionType.UNKNOWN


def parse_product_support_table(text: str) -> list[ProductSupport]:
    """Parse the product support table from markdown text."""
    results: list[ProductSupport] = []
    for line in text.split("\n"):
        line = line.strip()
        m = TABLE_ROW_PATTERN.match(line)
        if not m:
            continue
        if TABLE_SEPARATOR_PATTERN.match(line):
            continue

        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 2:
            continue

        product = cells[0].strip()
        symbol = cells[1].strip()

        if not product or product == "产品":
            continue

        supported = symbol in SUPPORTED_SYMBOLS
        results.append(ProductSupport(product=product, supported=supported))

    return results


def parse_function_signatures(text: str) -> list[FunctionSignature]:
    """Extract C function signatures from code blocks in the text."""
    results: list[FunctionSignature] = []
    in_code_block = False
    code_buffer: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "```":
            if in_code_block:
                code = "\n".join(code_buffer)
                sig = _parse_single_signature(code)
                if sig:
                    results.append(sig)
                code_buffer = []
            in_code_block = not in_code_block
            continue

        if in_code_block:
            code_buffer.append(line)

    return results


def _parse_single_signature(code: str) -> FunctionSignature | None:
    """Parse a single C function signature from code text."""
    normalized = code.strip()
    normalized = re.sub(r"//.*$", "", normalized, flags=re.MULTILINE)

    match = FUNC_SIG_PATTERN.match(normalized)
    if not match:
        return None

    return_type = match.group("return_type").strip()
    func_name = match.group("name").strip()
    params_raw = match.group("params").strip()

    params = [p.strip() for p in params_raw.split(",") if p.strip()]

    return FunctionSignature(
        return_type=return_type,
        function_name=func_name,
        parameters=params,
        raw_code=code.strip(),
    )


def parse_table_rows(text: str) -> list[dict[str, str]]:
    """Parse a markdown table into a list of row dicts.

    Handles multi-line cells where a row may span multiple lines
    (continuation lines have fewer columns than the header).
    """
    lines = text.split("\n")
    headers: list[str] = []
    rows: list[dict[str, str]] = []
    last_row: dict[str, str] | None = None

    for line in lines:
        stripped = line.strip()
        m = TABLE_ROW_PATTERN.match(stripped)
        if not m:
            continue

        cells = [c.strip() for c in m.group(1).split("|")]

        # Skip separator row
        if all(re.match(r"^[\s\-:]+$", c) for c in cells):
            continue

        # First non-separator row with enough cells → header
        if not headers and len(cells) >= 2:
            if "参数" in cells[0] or "产品" in cells[0] or "返回码" in cells[0]:
                headers = cells
                continue

        if not headers:
            continue

        # Full row: same number of columns as header and first cell is non-empty
        first_cell_empty = not cells[0].strip()
        if len(cells) >= len(headers) and not first_cell_empty:
            row = {}
            for idx, header in enumerate(headers):
                row[header] = cells[idx] if idx < len(cells) else ""
            rows.append(row)
            last_row = row
        elif last_row is not None:
            continuation = " ".join(c for c in cells if c)
            last_key = headers[-1]
            last_row[last_key] = (last_row.get(last_key, "") + " " + continuation).strip()

    return rows


def parse_operator_document(
    markdown_content: str,
    file_path: str = "",
) -> ParsedOperatorDocument:
    """Parse a CANN operator Markdown document into structured data.

    Args:
        markdown_content: Raw Markdown text.
        file_path: Optional file path for reference in logging.

    Returns:
        A ParsedOperatorDocument with classified sections and extracted data.
    """
    header, raw_sections = split_sections(markdown_content)

    parsed_sections: list[ParsedSection] = []

    for raw in raw_sections:
        parsed_section = ParsedSection(
            section_type=raw.section_type,
            heading=raw.heading,
            content=raw.body_text,
            line_start=raw.line_start,
            line_end=raw.line_end,
        )
        parsed_sections.append(parsed_section)

    doc = ParsedOperatorDocument(
        operator_name=header.operator_name,
        cann_version=header.cann_version,
        source_url=header.source_url,
        saved_date=header.saved_date,
        sections=parsed_sections,
    )

    logger.info(
        "Parsed operator '%s': %d sections",
        header.operator_name,
        len(parsed_sections),
    )

    return doc


def parse_operator_file(file_path: str | Path) -> ParsedOperatorDocument:
    """Read a Markdown file and parse it into a ParsedOperatorDocument."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    return parse_operator_document(content, file_path=str(path))
