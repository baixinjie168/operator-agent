"""Split CANN operator Markdown documents into labeled sections.

Uses markdown-it-py for robust token-based parsing. Handles:
- H2/H3 heading hierarchy for section and sub-section splitting
- Detached headings (``##`` on one line, actual text on the next)
- Footer navigation noise
- Metadata extraction (URL, Saved date)
"""

from __future__ import annotations

import logging
import re

from markdown_it import MarkdownIt

from shared.exceptions import DocumentParsingError
from shared.models.enums import SectionType

logger = logging.getLogger(__name__)

# Patterns for metadata and footer
URL_PATTERN = re.compile(r"^\*\*URL:\*\*\s*(.+)$")
SAVED_PATTERN = re.compile(r"^\*\*Saved:\*\*\s*(.+)$")
CANN_TITLE_PATTERN = re.compile(r"^(.+?)-CANN社区版(\d+\.\d+\.\d+)-昇腾社区$")
PLAIN_TITLE_PATTERN = re.compile(r"^(aclnn?\w+)$")
EXTRACTED_BY_PATTERN = re.compile(r"^\*Markdown extracted by")
NAVIGATION_PATTERN = re.compile(r"(?:上一篇|下一篇)")

# Heading classification patterns (for H2 level)
H2_PATTERNS: list[tuple[re.Pattern[str], SectionType]] = [
    (re.compile(r"产品支持"), SectionType.PRODUCT_SUPPORT),
    (re.compile(r"功能说明"), SectionType.FUNCTION_DESCRIPTION),
    (re.compile(r"函数原型"), SectionType.FUNCTION_PROTOTYPE),
    (re.compile(r"约束"), SectionType.CONSTRAINTS),
    (re.compile(r"调用示例"), SectionType.USAGE_EXAMPLE),
]

# H3 sub-section classification (matched against bold list item text like "**参数说明**")
H3_PATTERNS: list[tuple[re.Pattern[str], SectionType]] = [
    (re.compile(r"参数说明"), SectionType.PARAMS_GET_WORKSPACE),
    (re.compile(r"返回值"), SectionType.RETURN_CODES_GET_WORKSPACE),
]


class RawSection:
    """A raw section extracted from the Markdown, with classification."""

    __slots__ = ("section_type", "heading", "body_lines", "line_start", "line_end", "level")

    def __init__(
        self,
        section_type: SectionType,
        heading: str,
        body_lines: list[str],
        line_start: int,
        line_end: int,
        level: int = 2,
    ) -> None:
        self.section_type = section_type
        self.heading = heading
        self.body_lines = body_lines
        self.line_start = line_start
        self.line_end = line_end
        self.level = level

    @property
    def body_text(self) -> str:
        return "\n".join(self.body_lines)


class DocumentHeader:
    """Extracted header metadata from a CANN operator document."""

    __slots__ = ("operator_name", "cann_version", "source_url", "saved_date")

    def __init__(
        self,
        operator_name: str,
        cann_version: str,
        source_url: str | None = None,
        saved_date: str | None = None,
    ) -> None:
        self.operator_name = operator_name
        self.cann_version = cann_version
        self.source_url = source_url
        self.saved_date = saved_date


def _parse_title(title: str) -> tuple[str, str]:
    """Extract operator name and CANN version from the H1 title line.

    Supports:
    - '{name}-CANN社区版{version}-昇腾社区' → (name, version)
    - Plain operator name like 'aclnnAddRmsNorm' → (name, "unknown")
    """
    m = CANN_TITLE_PATTERN.match(title.strip())
    if m:
        return m.group(1).strip(), m.group(2)

    # Fallback: plain operator name (e.g. aclnnAddRmsNorm)
    m = PLAIN_TITLE_PATTERN.match(title.strip())
    if m:
        return m.group(1).strip(), "unknown"

    raise DocumentParsingError(f"Cannot parse operator title: {title!r}")


def _strip_footer(lines: list[str]) -> list[str]:
    """Remove footer navigation and extraction attribution noise."""
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if EXTRACTED_BY_PATTERN.match(stripped):
            cut = i
            break
        if NAVIGATION_PATTERN.search(stripped):
            for j in range(i - 1, max(i - 5, 0), -1):
                if lines[j].strip() == "---":
                    cut = j
                    break
            else:
                cut = i
            break
    return lines[:cut]


def _resolve_detached_h2(lines: list[str]) -> list[str]:
    """Merge detached H2 headings: '##' followed by text on the next line → '## text'."""
    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "##":
            for j in range(i + 1, min(i + 4, len(lines))):
                next_text = lines[j].strip()
                if next_text and not next_text.startswith("#"):
                    result.append(f"## {next_text}")
                    i = j + 1
                    break
            else:
                result.append(lines[i])
                i += 1
        else:
            result.append(lines[i])
            i += 1
    return result


def _classify_h2(heading: str, operator_name: str) -> SectionType:
    """Classify an H2 heading into a SectionType."""
    heading_stripped = heading.strip()

    if heading_stripped.endswith("GetWorkspaceSize"):
        return SectionType.GET_WORKSPACE_SIZE

    if heading_stripped == operator_name:
        return SectionType.EXECUTE_API

    for pattern, section_type in H2_PATTERNS:
        if pattern.search(heading_stripped):
            return section_type

    return SectionType.UNKNOWN


def _classify_h3(heading: str, parent_type: SectionType) -> SectionType:
    """Classify an H3 sub-section based on heading text and parent type."""
    for pattern, section_type in H3_PATTERNS:
        if pattern.search(heading):
            if section_type == SectionType.PARAMS_GET_WORKSPACE:
                return (
                    SectionType.PARAMS_EXECUTE
                    if parent_type == SectionType.EXECUTE_API
                    else SectionType.PARAMS_GET_WORKSPACE
                )
            if section_type == SectionType.RETURN_CODES_GET_WORKSPACE:
                return (
                    SectionType.RETURN_CODES_EXECUTE
                    if parent_type == SectionType.EXECUTE_API
                    else SectionType.RETURN_CODES_GET_WORKSPACE
                )
            return section_type
    return SectionType.UNKNOWN


def _find_h3_splits(lines: list[str], parent_type: SectionType) -> list[dict]:
    """Find H3-level split points within an H2 section body.

    CANN docs use bold list items like '- **参数说明**' and '- **返回值**'
    as pseudo-H3 sub-headings within H2 sections.

    Some documents contain zero-width spaces (\\u200b) or other invisible
    Unicode characters around formatting markers.  These must be stripped
    before regex matching to avoid silent failures.
    """
    H3_MARKER = re.compile(r"^[-*]\s+\*\*(.+?)\*\*")
    # Strip zero-width and other invisible Unicode formatting characters
    _INVISIBLE_RE = re.compile(r"[​‌‍﻿]")
    splits: list[dict] = []
    for i, line in enumerate(lines):
        cleaned = _INVISIBLE_RE.sub("", line.strip())
        m = H3_MARKER.match(cleaned)
        if m:
            heading_text = _INVISIBLE_RE.sub("", m.group(1)).strip()
            section_type = _classify_h3(heading_text, parent_type)
            if section_type != SectionType.UNKNOWN:
                splits.append({"index": i, "heading": heading_text, "section_type": section_type})
    return splits


def split_sections(markdown_content: str) -> tuple[DocumentHeader, list[RawSection]]:
    """Split a CANN operator Markdown document into header + classified sections.

    Uses markdown-it-py for token-based heading detection, then applies
    CANN-specific logic for detached headings, footer stripping, and
    H3 sub-section extraction.

    Args:
        markdown_content: Raw Markdown text.

    Returns:
        A tuple of (DocumentHeader, list of RawSection).

    Raises:
        DocumentParsingError: If the document structure is invalid.
    """
    lines = markdown_content.split("\n")

    if not lines or not lines[0].strip():
        raise DocumentParsingError("Document is empty")

    # --- Find first heading (H1 or H2) to extract operator name ---
    h1_line_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## "):
            h1_line_idx = i
            break

    if h1_line_idx is None:
        raise DocumentParsingError("No H1 or H2 heading found in document")

    # --- Parse title from first heading ---
    h1_text = lines[h1_line_idx].strip()
    # Strip heading prefix (# or ##)
    title_text = re.sub(r"^#{1,2}\s+", "", h1_text).strip()
    operator_name, cann_version = _parse_title(title_text)

    # --- Parse metadata lines (URL, Saved) ---
    source_url: str | None = None
    saved_date: str | None = None
    metadata_end = h1_line_idx + 1

    for i in range(h1_line_idx + 1, min(len(lines), h1_line_idx + 10)):
        line = lines[i].strip()
        if url_m := URL_PATTERN.match(line):
            source_url = url_m.group(1).strip()
            metadata_end = i + 1
        elif saved_m := SAVED_PATTERN.match(line):
            saved_date = saved_m.group(1).strip()
            metadata_end = i + 1
        elif line == "---":
            metadata_end = i + 1
            break

    header = DocumentHeader(
        operator_name=operator_name,
        cann_version=cann_version,
        source_url=source_url,
        saved_date=saved_date,
    )

    # --- Prepare content: strip footer, resolve detached H2s ---
    content_lines = _resolve_detached_h2(_strip_footer(lines[metadata_end:]))

    # --- Use markdown-it-py to find heading positions ---
    md = MarkdownIt("commonmark").enable("table")
    rejoined = "\n".join(content_lines)
    tokens = md.parse(rejoined)

    # Extract heading positions from tokens
    heading_positions: list[tuple[int, int, int, str]] = []  # (start, end, level, text)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open" and tok.map is not None:
            level = int(tok.tag[1])
            start_line = tok.map[0]
            end_line = tok.map[1]
            text = tokens[i + 1].content if i + 1 < len(tokens) else ""
            heading_positions.append((start_line, end_line, level, text))
            i += 3  # skip heading_open, inline, heading_close
            continue
        i += 1

    if not heading_positions:
        raise DocumentParsingError("No headings found in document")

    # --- Build H2 sections ---
    h2_positions = [(s, e, lvl, txt) for s, e, lvl, txt in heading_positions if lvl == 2]

    if not h2_positions:
        raise DocumentParsingError("No H2 headings found in document")

    sections: list[RawSection] = []

    for idx, (start, end, _, heading_text) in enumerate(h2_positions):
        # Content lines for this H2 section: from after heading to before next H2
        content_start = end
        if idx + 1 < len(h2_positions):
            content_end = h2_positions[idx + 1][0]
        else:
            content_end = len(content_lines)

        section_body = content_lines[content_start:content_end]
        global_offset = metadata_end
        section_type = _classify_h2(heading_text, operator_name)

        # Check for H3 sub-sections within this H2 body
        h3_splits = _find_h3_splits(section_body, section_type)

        if h3_splits:
            # Emit content before first H3 as parent section
            pre_h3_body = section_body[: h3_splits[0]["index"]]
            if any(line.strip() for line in pre_h3_body) or section_type in (
                SectionType.GET_WORKSPACE_SIZE,
                SectionType.EXECUTE_API,
            ):
                sections.append(
                    RawSection(
                        section_type=section_type,
                        heading=heading_text,
                        body_lines=pre_h3_body,
                        line_start=global_offset + start,
                        line_end=global_offset + content_start + h3_splits[0]["index"] - 1,
                        level=2,
                    )
                )

            # Emit each H3 sub-section
            for si, split in enumerate(h3_splits):
                sub_start = split["index"]
                sub_end = h3_splits[si + 1]["index"] if si + 1 < len(h3_splits) else len(section_body)
                sub_body = section_body[sub_start:sub_end]

                sections.append(
                    RawSection(
                        section_type=split["section_type"],
                        heading=split["heading"],
                        body_lines=sub_body,
                        line_start=global_offset + content_start + sub_start,
                        line_end=global_offset + content_start + sub_end - 1,
                        level=3,
                    )
                )
        else:
            # No H3 sub-sections, emit as single section
            sections.append(
                RawSection(
                    section_type=section_type,
                    heading=heading_text,
                    body_lines=section_body,
                    line_start=global_offset + start,
                    line_end=global_offset + content_end - 1,
                    level=2,
                )
            )

    logger.info(
        "Split document '%s' into %d sections",
        operator_name,
        len(sections),
    )

    return header, sections
