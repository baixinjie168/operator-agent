"""Shared context utilities for parameter extraction nodes.

Provides zero-LLM-cost pre-filtering tools:
- extract_param_context: slice section text to relevant parameter context
- parse_param_table: parse Markdown/HTML tables into structured rows

Used by llm_description_extract, allowed_range_extract, and other per-param nodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.utils.param_validators import is_ws_function as _is_ws_function  # noqa: F401 re-export


# ---------------------------------------------------------------------------
# Parameter context slicer
# ---------------------------------------------------------------------------

def extract_param_context(sections_text: str, param_name: str) -> str:
    """Extract the context fragment relevant to a specific parameter.

    Strategy:
    1. Split section text into lines
    2. Find lines mentioning *param_name* (with word-boundary awareness)
    3. Keep ±2 adjacent lines for semantic coherence
    4. Always preserve Markdown table headers (first ``|`` row + separator)
    5. v2: Expand to full <tr>...</tr> for HTML table rows
    6. v3: Resolve cross-parameter references ("与xxx一致" / "与`xxx`一致")
       by including the referenced parameter's table row
    7. Join the selected lines; fall back to full text if result is too short
    """
    lines = sections_text.split("\n")
    relevant: set[int] = set()

    # Phase 1: find lines that mention param_name
    for i, line in enumerate(lines):
        if _param_matches(line, param_name):
            for j in range(max(0, i - 2), min(len(lines), i + 3)):
                relevant.add(j)

    # Phase 1.5 (v2): expand HTML <tr> rows to full row boundaries
    in_html_table = False
    html_table_rows: list[tuple[int, int]] = []
    row_start = -1
    for i, line in enumerate(lines):
        lower = line.lower()
        if "<tr" in lower:
            in_html_table = True
            row_start = i
        if in_html_table and "</tr" in lower:
            html_table_rows.append((row_start, i))
            in_html_table = False

    for tr_start, tr_end in html_table_rows:
        if any(tr_start <= idx <= tr_end for idx in relevant):
            for j in range(tr_start, tr_end + 1):
                relevant.add(j)

    # Phase 2: always keep table headers (first | row + separator row)
    table_started = False
    for i, line in enumerate(lines):
        is_table_row = line.startswith("|")
        if is_table_row and not table_started:
            relevant.add(i)           # header row
            if i + 1 < len(lines):
                relevant.add(i + 1)   # separator row
            table_started = True
        elif not is_table_row:
            table_started = False

    # Phase 2.1: preserve HTML <thead> rows so the LLM can see column
    # headers (e.g. "数据格式", "数据类型") and map them to <td> values.
    in_thead = False
    for i, line in enumerate(lines):
        lower = line.lower()
        if "<thead" in lower:
            in_thead = True
        if in_thead:
            relevant.add(i)
        if "</thead" in lower:
            in_thead = False

    # Phase 2.5 (v3): resolve cross-parameter references
    # When a parameter's row contains "与xxx一致" / "与`xxx`一致" /
    # "和xxx相同" etc., include the referenced parameter's row so the
    # LLM can resolve the actual value.
    _resolve_cross_references(lines, relevant, html_table_rows, param_name)

    # Phase 3: no match found → return full text
    if not relevant:
        return sections_text

    # Phase 4: filtered text too short → return full text
    selected = [lines[i] for i in sorted(relevant)]
    result = "\n".join(selected)
    if len(result) < 100:
        return sections_text
    return result


# Regex: matches "与xxx一致", "与`xxx`一致", "和xxx相同", "与"xxx"一致" etc.
# Note: non-raw string so “ / ” (Chinese curly quotes) are interpreted.
# Use [a-zA-Z0-9_]* instead of \w* to avoid matching Chinese characters.
_CROSS_REF_RE = re.compile(
    '[与和同跟]\\s*["“”"‘’\'`]?'
    '\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*'
    '["“”"‘’\'`]?'
    '\\s*(?:一致|相同|一样|同)',
    re.UNICODE,
)


def _resolve_cross_references(
    lines: list[str],
    relevant: set[int],
    html_table_rows: list[tuple[int, int]],
    param_name: str,
) -> None:
    """When *param_name*'s table row contains cross-parameter references
    (e.g. "与self一致"), add the referenced parameter's row to *relevant*.

    Modifies *relevant* in place.  Only resolves one level deep to avoid
    combinatorial blow-up.
    """
    # Collect text from already-selected lines
    selected_text = "\n".join(lines[i] for i in sorted(relevant))

    # Find referenced parameter names
    refs = set(_CROSS_REF_RE.findall(selected_text))
    refs.discard(param_name)  # don't recurse into self

    if not refs:
        return

    # For each referenced param, find its table row and add to relevant
    for ref_name in refs:
        # HTML table rows
        for tr_start, tr_end in html_table_rows:
            row_text = "\n".join(lines[tr_start:tr_end + 1])
            if _param_matches(row_text, ref_name):
                for j in range(tr_start, tr_end + 1):
                    relevant.add(j)

        # Markdown table rows (lines starting with |)
        for i, line in enumerate(lines):
            if line.startswith("|") and _param_matches(line, ref_name):
                relevant.add(i)


def _param_matches(line: str, param_name: str) -> bool:
    """Check whether *line* mentions *param_name* with word-boundary awareness."""
    pattern = r"(?<![a-zA-Z0-9_])" + re.escape(param_name) + r"(?![a-zA-Z0-9_])"
    return bool(re.search(pattern, line))


# ---------------------------------------------------------------------------
# Table structure pre-parser
# ---------------------------------------------------------------------------

@dataclass
class ParamRow:
    """One row from a Markdown parameter table."""

    param_name: str
    param_type: str
    description: str
    raw_line: str  # original line, for source_citation


def parse_param_table(section_text: str) -> list[ParamRow]:
    """Parse Markdown/HTML parameter tables into structured :class:`ParamRow` objects.

    Supports:
    - Markdown tables: ``| param | type | description |``
    - Skips header row and separator row
    """
    rows: list[ParamRow] = []
    lines = section_text.split("\n")

    # Detect Markdown table rows
    md_rows: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "---" not in stripped:
            md_rows.append(stripped)

    if len(md_rows) > 1:  # skip header
        for row_line in md_rows[1:]:
            cells = [c.strip() for c in row_line.split("|")[1:-1]]
            if len(cells) >= 2:
                rows.append(
                    ParamRow(
                        param_name=cells[0],
                        param_type=cells[1] if len(cells) > 1 else "",
                        description=cells[2] if len(cells) > 2 else "",
                        raw_line=row_line,
                    )
                )
    return rows


def format_param_context(row: ParamRow, extra_paragraphs: list[str]) -> str:
    """Format a single parameter row + related paragraphs into concise context."""
    parts = [
        f"参数: {row.param_name}",
        f"类型: {row.param_type}",
        f"描述: {row.description}",
    ]
    if extra_paragraphs:
        parts.append("")
        parts.append("相关约束：")
        parts.extend(extra_paragraphs)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Function-type helpers
# ---------------------------------------------------------------------------

# _is_ws_function is re-exported from agent.utils.param_validators above
