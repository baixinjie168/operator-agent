"""Chunked extraction: split section into logical blocks and extract relations per block.

Supports three document formats:
- HTML tables: <table>...</table> (with <tr> grouping for large tables)
- Markdown tables: consecutive | lines (with header-preserving grouping)
- Parameter lists: - param_name（type，direction）：description
"""

import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.nodes.param_relation_extract.extract_relations import _extract_relations
from agent.nodes.param_relation_extract.merge_relations import _dedup_relations

logger = logging.getLogger(__name__)

# Regex to detect parameter list items: "  - param_name（type，direction）：..."
_PARAM_LIST_RE = re.compile(
    r"^\s*-\s+(\w+)\s*[（(][^）)]*[）)]\s*[:：]", re.UNICODE
)


def _is_param_list_item(line: str) -> bool:
    """Check if line is a top-level parameter list item."""
    return bool(_PARAM_LIST_RE.match(line))


def _is_nested_item(line: str) -> bool:
    """Check if line is a nested sub-item (indented - line)."""
    stripped = line.strip()
    indent = len(line) - len(line.lstrip())
    return indent >= 4 and stripped.startswith("-")


def _find_tr_end(lines: list[str], tr_start: int) -> int:
    """Find the line index after </tr> starting from tr_start."""
    for j in range(tr_start, len(lines)):
        if re.search(r"</tr\s*>", lines[j], re.IGNORECASE):
            return j + 1
    return tr_start + 1


def _merge_small_blocks(blocks: list[list[str]], max_size: int = 1500) -> list[str]:
    """Merge adjacent small text blocks while preserving block integrity."""
    merged: list[str] = []
    buffer = ""
    for block_lines in blocks:
        text = "\n".join(block_lines)
        if not text.strip():
            continue
        if len(buffer) + len(text) < max_size:
            buffer = buffer + "\n\n" + text if buffer else text
        else:
            if buffer:
                merged.append(buffer)
            buffer = text
    if buffer:
        merged.append(buffer)
    return merged


def split_into_chunks(section_text: str) -> list[str]:
    """Split section text into logical chunks for independent extraction.

    Strategy:
    - HTML tables: <table>...</table> as one block; split by <tr> if >15 rows
    - Markdown tables: consecutive | lines; split by 10 rows if >15 data rows
    - Parameter lists: each - param_name(type, dir): ... item as one block
    - Prose: split by blank lines
    - Merge small blocks (<1500 chars)
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    mode = "prose"  # "prose" | "md_table" | "param_list"

    lines = section_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- HTML table detection ---
        if mode != "html_table" and re.search(r"<table", stripped, re.IGNORECASE):
            if current:
                blocks.append(current)
                current = []
            # Collect entire <table>...</table>
            table_lines = [line]
            i += 1
            while i < len(lines) and not re.search(r"</table", lines[i], re.IGNORECASE):
                table_lines.append(lines[i])
                i += 1
            if i < len(lines):
                table_lines.append(lines[i])  # </table> line
            # Check if we need to split by <tr>
            tr_indices = [
                j for j, l in enumerate(table_lines)
                if re.search(r"<tr[\s>]", l, re.IGNORECASE)
            ]
            if len(tr_indices) > 15:
                # Extract header (first <tr>...</tr> block)
                header_end = _find_tr_end(table_lines, tr_indices[0])
                header = table_lines[:header_end]
                # Split remaining <tr> groups by 10, each keeping header
                for group_start in range(1, len(tr_indices), 10):
                    group = list(header)
                    end_idx = (
                        tr_indices[group_start + 10]
                        if group_start + 10 < len(tr_indices)
                        else len(table_lines)
                    )
                    start_idx = tr_indices[group_start]
                    group.extend(table_lines[start_idx:end_idx])
                    blocks.append(group)
            else:
                blocks.append(table_lines)
            mode = "prose"
            i += 1
            continue

        # --- Markdown table detection ---
        if mode == "prose" and stripped.startswith("|"):
            if current:
                blocks.append(current)
                current = []
            mode = "md_table"

        if mode == "md_table" and stripped.startswith("|"):
            current.append(line)
            i += 1
            # Check if table is too long and needs grouping
            if len(current) > 17 and "---" not in stripped:
                header = current[:2]
                data_rows = current[2:]
                for group_start in range(0, len(data_rows), 10):
                    group = list(header)
                    group.extend(data_rows[group_start:group_start + 10])
                    blocks.append(group)
                current = []
                mode = "prose"
            continue

        if mode == "md_table":
            if current:
                blocks.append(current)
            current = []
            mode = "prose"
            # Fall through to process current line

        # --- Parameter list detection ---
        if mode == "prose" and _is_param_list_item(line):
            if current:
                blocks.append(current)
                current = []
            mode = "param_list"

        if mode == "param_list":
            # New top-level parameter list item → split
            if _is_param_list_item(line):
                if current:
                    blocks.append(current)
                current = [line]
                i += 1
                continue
            # Nested sub-item or continuation line → belongs to current param
            if _is_nested_item(line) or stripped == "" or line[0:1] in (" ", "\t"):
                current.append(line)
                i += 1
                continue
            # Non-indented line (e.g., "- **返回值：**") → exit param_list mode
            if current:
                blocks.append(current)
                current = []
            mode = "prose"

        # --- Prose ---
        if mode == "prose":
            current.append(line)
            if stripped == "":
                blocks.append(current)
                current = []

        i += 1

    if current:
        blocks.append(current)

    return _merge_small_blocks(blocks, max_size=1500)


async def extract_relations_chunked(
    section_content: str,
    llm: ChatOpenAI,
    implicit_params: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Round 1: extract relations from each chunk independently."""
    chunks = split_into_chunks(section_content)
    all_relations: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            relations = await _extract_relations(chunk, llm, implicit_params)
        except Exception:
            logger.warning("Chunk %d/%d extraction failed, skipping", i + 1, len(chunks))
            continue
        # Mark source chunk for debugging
        for r in relations:
            r["_source_chunk"] = i
        all_relations.extend(relations)

    return _dedup_relations(all_relations)
