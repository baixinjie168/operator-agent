"""Pipeline run query routes: list runs, SSE stream, replay via RuntimeManager."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.db import (
    query_events as db_query_events,
    query_params_by_doc_id as db_query_params_by_doc_id,
    query_run as db_query_run,
    query_runs as db_query_runs,
    update_param_src_content as db_update_param_src_content,
)
from agent.runtime import RuntimeManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["runs"])


def _get_manager(request: Request) -> RuntimeManager:
    return request.app.state.runtime_manager


@router.get("/runs")
async def list_runs(operator_id: int | None = Query(default=None), limit: int = Query(default=20)):
    return {"runs": db_query_runs(operator_id, limit)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request):
    manager = _get_manager(request)
    run = manager.get_run(run_id)

    # Try DB if not in memory (server restart)
    if not run:
        db_run = db_query_run(run_id)
        if not db_run:
            return {"success": False, "error": "Run not found"}
        events = db_query_events(run_id)
        return {"success": True, "run": db_run, "events": events, "source": "db"}

    return {
        "success": True,
        "run": run.to_dict(),
        "events": [evt.to_sse() for evt in run.events],
        "source": "live",
    }


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request, since_seq: int = Query(default=0)):
    """SSE endpoint: replay DB history, then subscribe to live RuntimeManager events."""
    manager = _get_manager(request)

    async def generator():
        q = manager.subscribe(run_id)
        try:
            # Phase 1: replay DB history for late connections
            if since_seq >= 0:
                historical = db_query_events(run_id, since_seq)
                for evt in historical:
                    if await request.is_disconnected():
                        return
                    yield f"event: {evt['event_type']}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"

            # Check if run is active
            run = manager.get_run(run_id)
            if not run:
                db_run = db_query_run(run_id)
                if not db_run or db_run.get("status") != "running":
                    yield f"event: done\ndata: {json.dumps({'status': db_run.get('status') if db_run else 'unknown'})}\n\n"
                    return
                # Run exists in DB as 'running' but not in memory (server restart)
                # Events will be replayed from DB only
                yield f"event: done\ndata: {json.dumps({'status': db_run.get('status', 'running')})}\n\n"
                return

            # Phase 2: subscribe to live events
            yield ":ok\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ":keepalive\n\n"
                    continue
                except asyncio.CancelledError:
                    return

                yield f"event: {data.get('event_type', '')}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

                if data.get("event_type") in ("task.completed", "task.failed", "workflow.end", "workflow.error", "done"):
                    return
        finally:
            manager.unsubscribe(run_id, q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/operators/{operator_name}/latest-run")
async def get_latest_run_for_operator(operator_name: str):
    runs = db_query_runs(None, 200)
    match = [r for r in runs if r.get("operator_name") == operator_name]
    if not match:
        return {"success": False, "error": "No runs for this operator"}
    run = match[0]
    events = db_query_events(run["run_id"])
    return {"success": True, "run": run, "events": events}


@router.get("/runs/{run_id}/parameters")
async def get_run_parameters(run_id: str):
    """Return all extracted parameters for a pipeline run.

    Looks up doc_id from pipeline_runs, then queries the parameters table
    for param_name, param_type, direction, src_content and other fields.
    """
    db_run = db_query_run(run_id)
    if not db_run:
        return {"success": False, "error": "Run not found"}

    doc_id = db_run.get("doc_id")
    if not doc_id:
        return {"success": False, "error": "Run has no doc_id yet — pipeline may still be running"}

    params = db_query_params_by_doc_id(doc_id)
    return {
        "success": True,
        "run_id": run_id,
        "doc_id": doc_id,
        "operator_name": db_run.get("operator_name"),
        "count": len(params),
        "parameters": params,
    }


class UpdateSrcContentBody(BaseModel):
    src_content: str


@router.put("/parameters/{param_id}/src-content")
async def update_param_src_content(param_id: int, body: UpdateSrcContentBody):
    """Update the src_content field of a single parameter."""
    updated = db_update_param_src_content(param_id, body.src_content)
    if not updated:
        return {"success": False, "error": "Parameter not found"}
    return {"success": True, "param_id": param_id}
