"""Shared file/content utilities for operator name extraction."""

from __future__ import annotations

import re
from pathlib import Path


def extract_operator_name_from_file(file_path: Path) -> str:
    """Extract operator name from a Markdown file.

    Tries to read the file and extract from H1 heading.
    Falls back to the filename stem.
    """
    stem = file_path.stem
    try:
        content = file_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            m = re.match(r"^#{1,2}\s+(.+?)-CANN社区版", line)
            if m:
                return m.group(1).strip()
            m = re.match(r"^#{1,2}\s+(aclnn?\w+)", line)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return stem


def extract_operator_name_from_content(content: str) -> str | None:
    """Extract operator name from Markdown content text.

    Scans H1/H2 headings for the aclnn operator name pattern.
    """
    if not content:
        return None
    for line in content.splitlines():
        m = re.match(r"^#{1,2}\s+(.+?)-CANN社区版", line)
        if m:
            return m.group(1).strip()
        m = re.match(r"^#{1,2}\s+(aclnn?\w+)", line)
        if m:
            return m.group(1).strip()
    return None
