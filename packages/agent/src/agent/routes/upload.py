"""Upload route: async pipeline execution with RuntimeManager observability.

Business logic is clean: no SSE, no events, no manual emit.
All tracing is handled by @traced_node decorators + RuntimeManager + LLMTracer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re

from fastapi import APIRouter, Request, UploadFile

from agent.db import (
    complete_run as db_complete_run,
    create_run as db_create_run,
    save_events as db_save_events,
    update_run_doc_id as db_update_run_doc_id,
)
from agent.graph import create_pipeline_graph_after_init
from agent.nodes.init_doc import init_doc_node as _init_doc
from agent.runtime import EventType, LLMTracer, RuntimeManager, traced_node
from agent.schemas.upload import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])


def _get_manager(request: Request) -> RuntimeManager:
    return request.app.state.runtime_manager


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile, request: Request) -> UploadResponse:
    """Upload a CANN operator Markdown document — returns run_id immediately.

    Pipeline runs asynchronously.  Connect to GET /api/v1/runs/{run_id}/stream
    for real-time progress via SSE.
    """
    content = (await file.read()).decode("utf-8")
    filename = file.filename or "unknown"

    operator_name = _extract_operator_name(content)
    if not operator_name:
        return UploadResponse(success=False, error=f"Cannot parse operator name from {filename}")

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    state_input = {"operator_name": operator_name, "content": content, "content_hash": content_hash}

    manager = _get_manager(request)
    run = manager.create_run(operator_name)
    db_create_run(run.run_id, operator_name, content_hash)

    asyncio.create_task(_run_pipeline(run.run_id, state_input, manager))

    return UploadResponse(success=True, task_id=run.run_id, operator_name=operator_name)


async def _run_pipeline(run_id: str, state_input: dict, manager: RuntimeManager) -> None:
    """Run pipeline with RuntimeManager observability.  Nodes emit events via @traced_node.

    init_doc runs first (outside the graph) so doc_id can be persisted to
    pipeline_runs immediately.  The remaining nodes run in a sub-graph afterward.
    """
    ctx = manager.enter_context(run_id)
    run = manager.get_run(run_id)
    if not run:
        return

    await asyncio.sleep(0.5)

    manager.emit(EventType.WORKFLOW_START, run_id, run.spans[run_id], {
        "agent_id": "doc",
        "node_id": "init_doc",
        "message": "DocAgent 开始处理文档...",
        "step_index": 0, "progress_pct": 0, "progress_text": "开始",
    })

    llm_tracer = LLMTracer()

    try:
        traced_init_doc = traced_node("init_doc")(_init_doc)
        init_result = await traced_init_doc(state_input)

        doc_id = init_result.get("doc_id") if isinstance(init_result, dict) else None
        if doc_id:
            try:
                db_update_run_doc_id(run_id, doc_id)
            except Exception as e:
                logger.warning("Failed to update doc_id on pipeline_runs: %s", e)

        init_status = init_result.get("status", "") if isinstance(init_result, dict) else ""
        if init_status == "error":
            _persist_to_db(run_id, run, init_result, manager)
            manager.emit(EventType.WORKFLOW_ERROR, run_id, run.spans[run_id], {
                "agent_id": "doc",
                "error": init_result.get("error", "init_doc failed") if isinstance(init_result, dict) else "init_doc failed",
            })
            manager.complete_run(run_id, error=str(init_result.get("error", "")) if isinstance(init_result, dict) else "")
            return

        if isinstance(init_result, dict):
            state_input.update(init_result)

        graph = create_pipeline_graph_after_init()
        result = await graph.ainvoke(state_input, config={"callbacks": [llm_tracer]})

        _persist_to_db(run_id, run, result, manager)

        sc = len(result.get("sections", []))
        pc = len(result.get("parameters", []))
        prod = len(result.get("product_support", []))
        status = result.get("status", "completed")
        op_name = result.get("operator_name", state_input.get("operator_name", ""))
        doc_id = result.get("doc_id")
        version = result.get("version")
        manager.emit(EventType.WORKFLOW_END, run_id, run.spans[run_id], {
            "agent_id": "doc",
            "message": f"DocParserAgent 完成。状态={status}, v{version}",
            "summary": f"全流程完成。{sc} sections, {pc} 参数, {prod} 产品。",
            "progress_pct": 100, "progress_text": "完成",
            "result": {
                "status": status, "version": version,
                "sections_count": sc, "parameters_count": pc, "product_count": prod,
                "doc_id": doc_id, "operator_name": op_name, "run_id": run_id,
            },
        })
        manager.complete_run(run_id)

    except Exception as e:
        logger.exception("Pipeline execution failed for run %s", run_id)
        manager.emit(EventType.WORKFLOW_ERROR, run_id, run.spans[run_id], {
            "agent_id": "doc", "error": str(e),
        })
        manager.complete_run(run_id, error=str(e))


def _persist_to_db(run_id: str, run, result: dict, manager: RuntimeManager) -> None:
    """Persist all runtime events + spans to DB directly (no MCP)."""
    events_payload = []
    for evt in run.events:
        sse = evt.to_sse()
        events_payload.append({
            "seq": evt.seq,
            "event_type": evt.event_type.value,
            "data": sse["data"],
        })
    try:
        db_save_events(run_id, events_payload)
    except Exception as e:
        logger.warning("Failed to persist events to DB: %s", e)
    try:
        db_complete_run(run_id, result, doc_id=result.get("doc_id"))
    except Exception as e:
        logger.warning("Failed to complete run in DB: %s", e)


def _extract_operator_name(content: str) -> str | None:
    """Extract operator name from the first H1 or H2 title line."""
    for line in content.split("\n"):
        m = re.match(r"^#{1,2}\s+(.+?)-CANN社区版", line)
        if m:
            return m.group(1).strip()
        m = re.match(r"^#{1,2}\s+(aclnn?\w+)", line)
        if m:
            return m.group(1).strip()
    return None