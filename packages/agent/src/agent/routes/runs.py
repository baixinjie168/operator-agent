"""Pipeline run query routes: list runs, SSE stream, replay via RuntimeManager."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.db import (
    delete_task as db_delete_task,
)
from agent.db import (
    get_task_chain as db_get_task_chain,
)
from agent.db import (
    query_events as db_query_events,
)
from agent.db import (
    query_exec_results as db_query_exec_results,
)
from agent.db import (
    query_json_constraints_by_doc_id as db_query_json_constraints,
)
from agent.db import (
    query_params_by_doc_id as db_query_params_by_doc_id,
)
from agent.db import (
    query_relations_by_doc_id as db_query_relations_by_doc_id,
)
from agent.db import (
    query_run as db_query_run,
)
from agent.db import (
    query_runs as db_query_runs,
)
from agent.db import (
    query_test_cases as db_query_test_cases,
)
from agent.db import (
    update_param_src_content as db_update_param_src_content,
)
from agent.runtime import RuntimeManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["runs"])


def _get_manager(request: Request) -> RuntimeManager:
    return request.app.state.runtime_manager


@router.get("/runs")
async def list_runs(
    operator_id: int | None = Query(default=None),
    operator_name: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    limit: int = Query(default=20),
):
    return {"runs": db_query_runs(
        operator_id=operator_id,
        operator_name=operator_name,
        task_type=task_type,
        limit=limit,
    )}


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
            db_max_seq = -1
            if since_seq >= 0:
                historical = db_query_events(run_id, since_seq)
                for evt in historical:
                    if await request.is_disconnected():
                        return
                    yield f"event: {evt['event_type']}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"
                    db_max_seq = max(db_max_seq, evt.get("seq", -1))

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

            # Phase 1.5: replay in-memory events not yet persisted to DB
            replayed_max_seq = db_max_seq
            for evt in run.events:
                if evt.seq > db_max_seq:
                    sse = evt.to_sse()
                    yield f"event: {sse.get('event_type', '')}\ndata: {json.dumps(sse, ensure_ascii=False)}\n\n"
                    replayed_max_seq = max(replayed_max_seq, evt.seq)

            # Phase 2: subscribe to live events (skip any already replayed)
            yield ":ok\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                except TimeoutError:
                    yield ":keepalive\n\n"
                    continue
                except asyncio.CancelledError:
                    return

                if data.get("seq", -1) <= replayed_max_seq:
                    continue

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
    """Return the most recent pipeline run for ``operator_name``.

    Prefers runs that carry a ``doc_id`` (i.e. completed DocProcessorAgent /
    ExtractorAgent runs that produced ``json_constraints``) over pure
    GeneratorAgent runs, which never set ``doc_id`` and would otherwise
    shadow the constraint-producing run when sorting by ``created_at DESC``.

    The fallback to the most recent run of *any* type is preserved so
    callers that genuinely want "the most recent activity for this
    operator" still get an answer; they just get the
    constraint-producing one first.
    """
    runs = db_query_runs(None, 200)
    match = [r for r in runs if r.get("operator_name") == operator_name]
    if not match:
        return {"success": False, "error": "No runs for this operator"}

    # A run with a doc_id came from the DocProcessorAgent / ExtractorAgent
    # path (init_doc → update_run_doc_id) and therefore has json_constraints
    # attached.  GeneratorAgent runs never set doc_id, so we filter them
    # out of the "primary" candidate list.  See the trace in the bug
    # report: without this, the most recent (failed) generator run would
    # shadow the still-valid extractor run.
    with_doc = [r for r in match if r.get("doc_id")]
    run = with_doc[0] if with_doc else match[0]

    events = db_query_events(run["run_id"])
    return {"success": True, "run": run, "events": events}


@router.get("/runs/{run_id}/parameters")
async def get_run_parameters(run_id: str):
    """Return all extracted parameters for a pipeline run.

    Looks up doc_id from pipeline_runs, then queries the parameters table
    for param_name, param_type, direction, src_content and other fields.
    If the current run has no doc_id, walk up the parent task chain.
    """
    db_run = db_query_run(run_id)
    if not db_run:
        return {"success": False, "error": "Run not found"}

    doc_id = db_run.get("doc_id")
    if not doc_id:
        chain = db_get_task_chain(run_id)
        for task in chain:
            if task.get("doc_id"):
                doc_id = task["doc_id"]
                break
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


@router.get("/runs/{run_id}/relations")
async def get_run_relations(run_id: str):
    """Return all param_relations for a pipeline run.

    Looks up doc_id from pipeline_runs, then queries param_relations table.
    If the current run has no doc_id, walk up the parent task chain.
    """
    db_run = db_query_run(run_id)
    if not db_run:
        return {"success": False, "error": "Run not found"}

    doc_id = db_run.get("doc_id")
    if not doc_id:
        chain = db_get_task_chain(run_id)
        for task in chain:
            if task.get("doc_id"):
                doc_id = task["doc_id"]
                break
    if not doc_id:
        return {"success": False, "error": "Run has no doc_id yet — pipeline may still be running"}

    relations = db_query_relations_by_doc_id(doc_id)
    return {
        "success": True,
        "run_id": run_id,
        "doc_id": doc_id,
        "operator_name": db_run.get("operator_name"),
        "count": len(relations),
        "relations": relations,
    }


@router.get("/runs/{run_id}/json-constraints")
async def get_run_json_constraints(run_id: str):
    """Return json_constraints from document_versions for a pipeline run.

    If the current run has no doc_id (e.g. case_generate / test_execute tasks),
    walk up the parent task chain to find one that does.
    """
    db_run = db_query_run(run_id)
    if not db_run:
        return {"success": False, "error": "Run not found"}

    doc_id = db_run.get("doc_id")

    # Walk up parent chain to find a doc_id
    if not doc_id:
        chain = db_get_task_chain(run_id)
        for task in chain:
            if task.get("doc_id"):
                doc_id = task["doc_id"]
                break

    if not doc_id:
        return {"success": False, "error": "Run has no doc_id yet — pipeline may still be running"}

    result = db_query_json_constraints(doc_id)
    if not result:
        return {"success": False, "error": "No json_constraints found for this run"}

    return {
        "success": True,
        "run_id": run_id,
        "doc_id": doc_id,
        "operator_name": db_run.get("operator_name"),
        "json_constraints": result.get("json_constraints", {}),
    }


@router.get("/runs/{run_id}/chain")
async def get_task_chain_endpoint(run_id: str):
    """Return the full dependency chain for a task: task → parent → grandparent → ..."""
    chain = db_get_task_chain(run_id)
    if not chain:
        return {"success": False, "error": "Task not found"}
    return {"success": True, "chain": chain, "depth": len(chain)}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a task and all its descendant tasks with cascading data."""
    db_run = db_query_run(run_id)
    if not db_run:
        return {"success": False, "error": "Run not found"}

    result = db_delete_task(run_id)
    return {
        "success": True,
        "deleted_tasks": result["deleted_tasks"],
        "task_ids": result["task_ids"],
    }


@router.get("/test-cases")
async def list_test_cases(
    task_id: str | None = Query(default=None),
    operator_name: str | None = Query(default=None),
    supported_product: str | None = Query(default=None),
    limit: int = Query(default=500),
):
    """Query test cases by task_id or operator_name, optionally filtered by product."""
    cases = db_query_test_cases(task_id=task_id, operator_name=operator_name, limit=limit)

    # Filter by supported_product if specified
    if supported_product:
        filtered = []
        for c in cases:
            case_product = c.get("supported_product", "")
            if not case_product and isinstance(c.get("case_data"), dict):
                case_product = c["case_data"].get("supported_product", "")
            if case_product == supported_product:
                filtered.append(c)
        cases = filtered

    return {
        "success": True,
        "count": len(cases),
        "cases": cases,
    }


@router.get("/test-cases/products")
async def list_test_case_products(
    task_id: str | None = Query(default=None),
    operator_name: str | None = Query(default=None),
):
    """Get list of unique supported_product values for test cases."""
    cases = db_query_test_cases(task_id=task_id, operator_name=operator_name, limit=5000)

    products = set()
    for c in cases:
        p = c.get("supported_product", "")
        if not p and isinstance(c.get("case_data"), dict):
            p = c["case_data"].get("supported_product", "")
        if p:
            products.add(p)

    return {
        "success": True,
        "products": sorted(list(products)),
    }


@router.get("/exec-results")
async def list_exec_results(
    task_id: str | None = Query(default=None),
    case_id: int | None = Query(default=None),
    operator_name: str | None = Query(default=None),
    limit: int = Query(default=500),
):
    """Query exec results by task_id, case_id, or operator_name."""
    results = db_query_exec_results(
        task_id=task_id, case_id=case_id, operator_name=operator_name, limit=limit,
    )
    return {
        "success": True,
        "count": len(results),
        "results": results,
    }
