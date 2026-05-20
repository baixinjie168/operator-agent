"""Upload route: accepts operator documents and triggers the pipeline graph."""

from __future__ import annotations

import hashlib
import logging
import re

from fastapi import APIRouter, UploadFile

from agent.graph import create_pipeline_graph
from agent.mcp_client import MCPClient
from agent.schemas.upload import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])

_mcp_client = MCPClient()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    """Upload a CANN operator Markdown document for processing.

    Flow:
    1. Read file content, compute hash, extract operator name
    2. Check version via MCP — return existing if unchanged
    3. Save new/updated document via MCP
    4. Run pipeline graph: InitDoc → ParseParams → PersistParams
    5. Return results
    """
    content = (await file.read()).decode("utf-8")
    filename = file.filename or "unknown"

    operator_name = _extract_operator_name(content)
    if not operator_name:
        return UploadResponse(success=False, error=f"Cannot parse operator name from {filename}")

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    client = _mcp_client

    try:
        # Deterministic pre-processing
        version_info = await client.check_version(operator_name, content_hash)
        status = version_info.get("status", "new")
        existing_version = version_info.get("version")

        if status == "unchanged":
            existing = await client.get_parsed(operator_name, existing_version)
            if existing:
                return UploadResponse(
                    success=True,
                    operator_name=operator_name,
                    cann_version=existing.get("cann_version"),
                    status="unchanged",
                    version=existing_version,
                    sections_count=len(existing.get("sections", [])),
                )

        save_result = await client.save_doc(operator_name, content)
        new_version = save_result["version"]

        # Pipeline graph processing
        graph = create_pipeline_graph()
        result = await graph.ainvoke(
            {
                "operator_name": operator_name,
                "version": new_version,
                "content": content,
            },
        )

        # Read back saved results for response
        saved = await client.get_parsed(operator_name, new_version)
        sections_count = len(saved.get("sections", [])) if saved else 0

        pipeline_error = result.get("error")
        if pipeline_error:
            logger.warning("Pipeline completed with error: %s", pipeline_error)

        return UploadResponse(
            success=True,
            operator_name=operator_name,
            cann_version=saved.get("cann_version") if saved else None,
            status=status,
            version=new_version,
            sections_count=sections_count,
        )

    except Exception as e:
        logger.exception("Upload processing failed for %s", filename)
        return UploadResponse(success=False, error=str(e))


def _extract_operator_name(content: str) -> str | None:
    """Extract operator name from the first H1 or H2 title line.

    Supports formats:
    - # {name}-CANN社区版{version}-昇腾社区
    - # {name}  (plain H1, e.g. "# aclnnAddRmsNorm")
    - ## {name}  (H2 as first heading, e.g. "## aclnnAddRmsNorm")
    """
    for line in content.split("\n"):
        # Pattern 1: original format with CANN version suffix (H1 or H2)
        m = re.match(r"^#{1,2}\s+(.+?)-CANN社区版", line)
        if m:
            return m.group(1).strip()
        # Pattern 2: plain heading with operator-like name (aclnn/aclnnXxx)
        m = re.match(r"^#{1,2}\s+(aclnn?\w+)", line)
        if m:
            return m.group(1).strip()
    return None
