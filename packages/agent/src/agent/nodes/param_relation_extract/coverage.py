"""Coverage checks: find uncovered parameters and paragraphs.

Dual-dimension coverage:
1. Parameter coverage: params not appearing in any relation
2. Paragraph coverage: paragraphs mentioning 2+ params but not cited
"""

import re
from typing import Any


def find_uncovered_params(
    param_names: list[str],
    relations: list[dict[str, Any]],
) -> list[str]:
    """Find parameters not appearing in any extracted relation.

    An isolated parameter (no relations) likely indicates
    that its relations were missed during extraction.
    """
    covered = set()
    for r in relations:
        for p in r.get("params", []):
            covered.add(p)

    return [name for name in param_names if name not in covered]


def find_uncovered_context_mentions(
    section_content: str,
    param_names: list[str],
    relations: list[dict[str, Any]],
) -> list[str]:
    """Find paragraphs mentioning 2+ params but not covered by any source_citation.

    Uses 60-char fingerprint matching against source_citations
    and word-boundary-aware parameter name matching.
    """
    # Collect 60-char fingerprints from existing source_citations
    existing_fragments: set[str] = set()
    for r in relations:
        citation = r.get("source_citation", "").strip()
        if citation:
            if len(citation) <= 60:
                # Short citation: use the whole thing as fingerprint
                existing_fragments.add(citation)
            else:
                for start in range(0, len(citation) - 30, 30):
                    existing_fragments.add(citation[start:start + 60])

    uncovered_paragraphs: list[str] = []
    for para in section_content.split("\n\n"):
        para = para.strip()
        if len(para) < 20:
            continue

        # Word-boundary matching to avoid "x" matching "axis"
        mentioned = [
            name for name in param_names
            if re.search(
                r"(?<![a-zA-Z0-9_])" + re.escape(name) + r"(?![a-zA-Z0-9_])",
                para,
            )
        ]
        if len(mentioned) >= 2:
            # Check if this paragraph is already covered by a source_citation
            # Bidirectional: either fragment in para, or para in fragment
            is_covered = any(
                frag in para or para in frag
                for frag in existing_fragments
            )
            if not is_covered:
                uncovered_paragraphs.append(para)

    return uncovered_paragraphs
