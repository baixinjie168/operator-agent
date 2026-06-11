"""GeneratorAgent route: async 5-step case-generation pipeline with SSE.

Mirrors the DocProcessorAgent pattern (see ``routes/upload.py``): the
endpoint returns immediately with a ``run_id`` and the pipeline runs in a
background ``asyncio`` task.  The client subscribes to
``/api/v1/runs/{run_id}/stream`` for real-time progress via SSE.

The 5 sub-steps are the same nodes used by the main pipeline:
    case_match_model → case_load_defs → case_init_static
    → case_solve_constraints → case_generate
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from fastapi import APIRouter, Request

from agent.graph import PipelineStage, build_pipeline
from agent.nodes.state import PipelineState
from agent.runtime import EventType, LLMTracer, RuntimeManager
from agent.schemas.cases import GeneratorRunRequest, GeneratorRunResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["generator"])


def _get_manager(request: Request) -> RuntimeManager:
    return request.app.state.runtime_manager


def _synthetic_content_hash(operator_name: str, count: int, seed: int, run_id: str) -> str:
    """Generate a deterministic content_hash for a generator run.

    The DocProcessorAgent's content_hash fingerprints the uploaded
    document. The GeneratorAgent has no document of its own — its
    "input" is the operator_name + the parameters the user picked.  We
    fold those plus the run_id into a sha256 so each invocation gets a
    distinct, reproducible fingerprint that satisfies the NOT NULL
    constraint on ``pipeline_runs.content_hash``.
    """
    payload = f"generator:{operator_name}:{count}:{seed}:{run_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@router.post("/generator/run", response_model=GeneratorRunResponse)
async def run_generator(body: GeneratorRunRequest, request: Request) -> GeneratorRunResponse:
    """Trigger the GeneratorAgent 5-step case-generation pipeline asynchronously.

    Returns ``task_id`` (same as ``run_id``) immediately.  Subscribe to
    ``GET /api/v1/runs/{task_id}/stream`` for real-time SSE progress events.
    """
    operator_name = body.operator_name.strip()
    if not operator_name:
        return GeneratorRunResponse(
            success=False, task_id="", operator_name=body.operator_name, count=body.count,
            error="operator_name is required",
        )

    count = body.count
    seed = body.seed if body.seed is not None else 42
    manager = _get_manager(request)
    run = manager.create_run(operator_name)
    run_id = run.run_id

    # ── Persist the pipeline_runs row IMMEDIATELY so the FK on
    # pipeline_events.run_id is satisfied by the time _run_case_pipeline
    # calls db_save_events(...).  Without this the insert would fail with
    # "FOREIGN KEY constraint failed" (see git history).  Mirror the
    # pattern in routes/upload.py.
    from agent.db import create_run as db_create_run
    from agent.db import find_parent_task

    parent_id = find_parent_task(operator_name, "constraint_extract")

    try:
        db_create_run(
            run_id,
            operator_name,
            _synthetic_content_hash(operator_name, count, seed, run_id),
            task_type="case_generate",
            parent_task_id=parent_id,
        )
    except Exception as e:
        # Don't fail the request just because the bookkeeping row didn't
        # land — the in-memory run can still produce cases.  The error
        # path below will still try db_save_events, which will also fail
        # on the same FK, but it logs and continues.
        logger.warning("Failed to insert pipeline_runs row for generator %s: %s", run_id, e)

    logger.info(
        "POST /generator/run: op=%s count=%d seed=%d run_id=%s",
        operator_name, count, seed, run_id,
    )

    asyncio.create_task(_run_case_pipeline(run_id, operator_name, count, seed, parent_id, manager))

    return GeneratorRunResponse(
        success=True, task_id=run_id, operator_name=operator_name, count=count,
    )


async def _run_case_pipeline(
    run_id: str, operator_name: str, count: int, seed: int,
    parent_task_id: str | None, manager: RuntimeManager,
) -> None:
    """Run the 5-step case-generation sub-graph with RuntimeManager observability."""
    ctx = manager.enter_context(run_id)
    run = manager.get_run(run_id)
    if not run:
        return

    await asyncio.sleep(0.3)

    manager.emit(EventType.WORKFLOW_START, run_id, run.spans[run_id], {
        "agent_id": "case",
        "node_id": "case_match_model",
        "message": f"GeneratorAgent 开始为 {operator_name} 生成测试用例...",
        "step_index": 0, "progress_pct": 0, "progress_text": "开始",
    })

    llm_tracer = LLMTracer()
    state_input: PipelineState = {
        "operator_name": operator_name,
        "cases_count": count,
        "cases_seed": seed,
    }

    try:
        graph = build_pipeline([PipelineStage.GENERATE])
        result = await graph.ainvoke(state_input, config={"callbacks": [llm_tracer]})

        # Persist all runtime events to DB.  Use the alias name (e.g.
        # "node.started") emitted by to_sse() so the DB row matches what
        # the SSE stream sends — the frontend's _eventRouteMap only
        # recognises the alias names, so raw enum values would silently
        # break replays after a backend restart.
        events_payload = []
        for evt in run.events:
            sse = evt.to_sse()
            events_payload.append({
                "seq": evt.seq,
                "event_type": sse["event_type"],
                "data": sse["data"],
            })

        from agent.db import complete_run as db_complete_run
        from agent.db import save_events as db_save_events

        try:
            db_save_events(run_id, events_payload)
        except Exception as e:
            logger.warning("Failed to persist generator events to DB: %s", e)

        cases_count = result.get("cases_count")
        cases_path = result.get("cases_path", "")
        cases_list = result.get("cases", [])
        error = result.get("error")
        status = "completed" if not error else "failed"

        logger.info(
            "generator pipeline result: keys=%s, cases_count=%s, "
            "cases_list_type=%s, cases_list_len=%d, error=%s",
            list(result.keys()), cases_count,
            type(cases_list).__name__, len(cases_list), error,
        )

        # Save individual case records to test_cases table
        if not error and cases_list:
            logger.info("Attempting to save %d cases to test_cases table...", len(cases_list))
            try:
                from agent.db import query_run as db_query_run
                from agent.db import save_test_cases as db_save_test_cases

                constraint_doc_id = None
                if parent_task_id:
                    parent_run = db_query_run(parent_task_id)
                    if parent_run:
                        constraint_doc_id = parent_run.get("doc_id")

                logger.info(
                    "save_test_cases params: task_id=%s, operator=%s, "
                    "cases_len=%d, constraint_doc_id=%s",
                    run_id, operator_name, len(cases_list), constraint_doc_id,
                )
                save_result = db_save_test_cases(
                    task_id=run_id,
                    operator_name=operator_name,
                    cases=cases_list,
                    constraint_doc_id=constraint_doc_id,
                )
                logger.info("Saved test cases to DB: %s", save_result)
            except Exception as e:
                logger.exception("Failed to save test cases to DB: %s", e)
        else:
            logger.warning(
                "Skipping test_cases save: error=%s, cases_list_empty=%s",
                error, not cases_list,
            )

        # ── Mirror upload.py: write the final run row (status +
        # result_json) so the DB reflects what really happened.  Without
        # this, every generator run would stay in 'running' status
        # forever in the DB even though the in-memory state is correct.
        try:
            db_complete_run(
                run_id,
                {
                    "status": status,
                    "operator_name": operator_name,
                    "cases_count": cases_count,
                    "cases_path": cases_path,
                },
                error=error,
            )
        except Exception as e:
            logger.warning("Failed to complete generator run in DB: %s", e)

        manager.emit(EventType.WORKFLOW_END, run_id, run.spans[run_id], {
            "agent_id": "case",
            "message": (
                f"GeneratorAgent 完成。生成 {cases_count} 个用例"
                if not error else f"GeneratorAgent 失败: {error}"
            ),
            "summary": f"用例生成完成。{cases_count} 个用例 → {cases_path}" if not error else f"失败: {error}",
            "progress_pct": 100, "progress_text": "完成" if not error else "失败",
            "result": {
                "status": status,
                "operator_name": operator_name,
                "cases_count": cases_count,
                "cases_path": cases_path,
                "run_id": run_id,
            },
        })
        manager.complete_run(run_id, error=error)

    except Exception as e:
        logger.exception("Case pipeline execution failed for run %s", run_id)
        manager.emit(EventType.WORKFLOW_ERROR, run_id, run.spans[run_id], {
            "agent_id": "case", "error": str(e),
        })
        # Try to mark the DB row as failed too — otherwise the row
        # would sit in 'running' status forever, hiding the crash.
        try:
            from agent.db import complete_run as db_complete_run
            db_complete_run(run_id, {}, error=str(e))
        except Exception as inner_e:
            logger.warning("Failed to mark generator run as failed in DB: %s", inner_e)
        manager.complete_run(run_id, error=str(e))
