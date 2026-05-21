"""Upload route: accepts operator documents and triggers the pipeline graph."""

from __future__ import annotations

import hashlib
import logging
import re

from fastapi import APIRouter, UploadFile

from agent.graph import create_pipeline_graph
from agent.schemas.upload import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    """Upload a CANN operator Markdown document for processing.

    The route is thin — it reads the file, extracts the operator name,
    and delegates all MCP interactions to the pipeline graph:
    InitDoc → ParseParams → PersistParams.
    """
    content = (await file.read()).decode("utf-8")
    filename = file.filename or "unknown"

    operator_name = _extract_operator_name(content)
    if not operator_name:
        return UploadResponse(success=False, error=f"Cannot parse operator name from {filename}")

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    try:
        graph = create_pipeline_graph()
        result = await graph.ainvoke(
            {
                "operator_name": operator_name,
                "content": content,
                "content_hash": content_hash,
            },
        )

        pipeline_error = result.get("error")
        if pipeline_error:
            logger.warning("Pipeline completed with error: %s", pipeline_error)
            return UploadResponse(success=False, error=pipeline_error)

        return UploadResponse(
            success=True,
            operator_name=operator_name,
            cann_version=result.get("cann_version"),
            status=result.get("status"),
            version=result.get("version"),
            sections_count=len(result.get("sections", [])),
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
