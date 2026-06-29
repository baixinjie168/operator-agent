"""Section content resolution helpers shared across extraction nodes.

Centralises the "promote exe -> ws for single-function operators" fallback so
that ``fetch_sections`` (param_relation_extract subgraph) and
``constraint_extract`` Pass 3 keep an identical ordering invariant:
**promote first, then append constraints** to ``ws_content``.

Single-function operators (e.g. aclnnCalculateMatmulWeightSize) have no
``params_get_workspace`` section — their parameter table lives in
``params_execute``.  Promoting exe content into ws lets downstream per-round
consumers (constraint_extract Pass 5) see the parameter table in the ws round.

Safety: the differential readers of ws/exe state (extract_ws_node /
extract_exe_node) recombine their output via ``merge_relations`` with plain
concatenation (``ws_relations + exe_relations``), so moving content from the
exe bucket to the ws bucket does not lose coverage.  See plan Step 4 note.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# Prefix used when appending the constraints section to ws_content. Kept
# identical to the historical inline implementation so merged section text is
# byte-compatible for two-stage operators (promotion never triggers there).
_CONSTRAINTS_PREFIX = "\n\n---\n## 约束说明\n"


async def resolve_ws_exe_content(
    mcp_client: "MCPClient",
    doc_id: int,
) -> tuple[str, str, str]:
    """Resolve ``(ws_content, exe_content, constraints_text)`` for a document.

    Fetches ``params_get_workspace`` / ``params_execute`` / ``constraints``
    sections and applies the single-function promotion: when
    ``params_get_workspace`` is empty but ``params_execute`` has content, the
    exe content is promoted into ``ws_content`` (the "primary" content).

    Ordering invariant (must not change): promote FIRST, then append
    constraints to ``ws_content``.  Appending constraints to the
    already-cleared ``exe_content`` would silently drop them.

    Returns:
        ws_content: params_get_workspace content (or promoted exe content for
            single-function operators), with the constraints section appended
            when present.
        exe_content: params_execute content (empty string after promotion for
            single-function operators).
        constraints_text: raw constraints section content (without prefix),
            returned separately for callers such as constraint_extract Pass
            4/4b/6 that consume it on its own.

    For two-stage operators ``ws_content`` is non-empty so the promotion
    never triggers — behavior is unchanged.
    """
    ws_section = await mcp_client.get_section(doc_id, "params_get_workspace")
    exe_section = await mcp_client.get_section(doc_id, "params_execute")
    constraints_section = await mcp_client.get_section(doc_id, "constraints")

    ws_content = ws_section.get("content", "") if ws_section else ""
    exe_content = exe_section.get("content", "") if exe_section else ""
    constraints_text = (
        constraints_section.get("content", "") if constraints_section else ""
    )

    # Single-function fallback: promote exe content into ws (the "primary"
    # content) when ws is empty.  Safe per the module docstring's safety note.
    if not ws_content.strip() and exe_content.strip():
        ws_content = exe_content
        exe_content = ""

    if constraints_text:
        ws_content += _CONSTRAINTS_PREFIX + constraints_text

    return ws_content, exe_content, constraints_text
