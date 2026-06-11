"""ExecuterAgent route: async 3-step execution pipeline with SSE.

Mirrors the GeneratorAgent pattern (see ``routes/generator.py``): the
endpoint returns immediately with a ``run_id`` and the pipeline runs in a
background ``asyncio`` task.  The client subscribes to
``/api/v1/runs/{run_id}/stream`` for real-time progress via SSE.

The 3 sub-steps are the same nodes used by the main pipeline:
    exec_generate_atk → exec_cpu_derivation → exec_run_atk
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request

from agent.graph import PipelineStage, build_pipeline
from agent.nodes.state import PipelineState
from agent.runtime import EventType, LLMTracer, RuntimeManager
from agent.schemas.cases import ExecuteRunRequest, ExecuteRunResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["execute"])


def _get_manager(request: Request) -> RuntimeManager:
    return request.app.state.runtime_manager


def _synthetic_content_hash(operator_name: str, run_id: str) -> str:
    """Generate a deterministic content_hash for an execute run."""
    payload = f"execute:{operator_name}:{run_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cases_dir() -> Path:
    """Resolve the on-disk cases directory."""
    return Path(__file__).resolve().parents[4] / "cases"


@router.post("/execute/run", response_model=ExecuteRunResponse)
async def run_execute(body: ExecuteRunRequest, request: Request) -> ExecuteRunResponse:
    """Trigger the ExecuterAgent 3-step execution pipeline asynchronously.

    Returns ``task_id`` (same as ``run_id``) immediately.  Subscribe to
    ``GET /api/v1/runs/{task_id}/stream`` for real-time SSE progress events.
    """
    operator_name = body.operator_name.strip()
    if not operator_name:
        return ExecuteRunResponse(
            success=False, task_id="", operator_name=body.operator_name,
            error="operator_name is required",
        )

    from agent.db import create_run as db_create_run
    from agent.db import find_parent_task
    from agent.db import query_test_cases as db_query_test_cases

    parent_id = find_parent_task(operator_name, "case_generate")

    # Read cases from DB (preferred) or fall back to request body
    db_cases = []
    case_ids = []
    if parent_id:
        db_cases = db_query_test_cases(task_id=parent_id)
        case_ids = [c["id"] for c in db_cases]

    # Look up server info if server_id is provided
    server_info = None
    if body.server_id:
        try:
            from agent.db import get_server as db_get_server
            server_info = db_get_server(body.server_id)
        except Exception as e:
            logger.warning("Failed to fetch server info: %s", e)

    if db_cases:
        cases_data = [c["case_data"] for c in db_cases]
        # Filter cases by server's supported_product if server is specified
        if server_info and server_info.get("supported_product"):
            server_product = server_info["supported_product"]
            filtered = []
            filtered_ids = []
            for i, c in enumerate(db_cases):
                case_product = c.get("supported_product", "") or c.get("case_data", {}).get("supported_product", "")
                if case_product == server_product or not case_product:
                    filtered.append(c["case_data"])
                    filtered_ids.append(c["id"])
            if not filtered:
                # Collect unique products from cases
                case_products = set()
                for c in db_cases:
                    p = c.get("supported_product", "") or c.get("case_data", {}).get("supported_product", "")
                    if p:
                        case_products.add(p)
                return ExecuteRunResponse(
                    success=False, task_id="", operator_name=operator_name,
                    error=f"服务器 {server_info['name']} 支持的产品为「{server_product}」，"
                          f"但用例对应的产品为「{'、'.join(case_products) or '未知'}」，产品不匹配无法执行。"
                          f"请切换服务器或重新生成对应用例。",
                )
            cases_data = filtered
            case_ids = filtered_ids
        cases_json_str = json.dumps(cases_data, ensure_ascii=False)
    else:
        try:
            cases_data = json.loads(body.cases_json)
            if not isinstance(cases_data, list):
                return ExecuteRunResponse(
                    success=False, task_id="", operator_name=operator_name,
                    error="cases_json must be a JSON array",
                )
            cases_json_str = body.cases_json
        except json.JSONDecodeError as e:
            return ExecuteRunResponse(
                success=False, task_id="", operator_name=operator_name,
                error=f"Invalid JSON: {e}",
            )

    manager = _get_manager(request)
    run = manager.create_run(operator_name)
    run_id = run.run_id

    try:
        db_create_run(
            run_id,
            operator_name,
            _synthetic_content_hash(operator_name, run_id),
            task_type="test_execute",
            parent_task_id=parent_id,
        )
    except Exception as e:
        logger.warning("Failed to insert pipeline_runs row for execute %s: %s", run_id, e)

    cases_dir = _cases_dir()
    cases_dir.mkdir(parents=True, exist_ok=True)
    cases_path = cases_dir / f"{operator_name}_cases.json"
    cases_path.write_text(cases_json_str, encoding="utf-8")

    logger.info(
        "POST /execute/run: op=%s cases=%d run_id=%s source=%s server_id=%s",
        operator_name, len(cases_data), run_id, "db" if db_cases else "request", body.server_id,
    )

    asyncio.create_task(
        _run_execute_pipeline(run_id, operator_name, str(cases_path), len(cases_data), case_ids, manager, server_info)
    )

    return ExecuteRunResponse(
        success=True, task_id=run_id, operator_name=operator_name,
    )


async def _run_execute_pipeline(
    run_id: str, operator_name: str, cases_path: str, cases_count: int,
    case_ids: list[int], manager: RuntimeManager, server_info: dict | None = None,
) -> None:
    """Run the 3-step execution sub-graph with RuntimeManager observability."""
    ctx = manager.enter_context(run_id)
    run = manager.get_run(run_id)
    if not run:
        return

    await asyncio.sleep(0.3)

    server_name = server_info["name"] if server_info else "本地执行"

    manager.emit(EventType.WORKFLOW_START, run_id, run.spans[run_id], {
        "agent_id": "execute",
        "node_id": "exec_generate_atk",
        "message": f"ExecuterAgent 开始为 {operator_name} 执行测试用例（{server_name}）...",
        "step_index": 0, "progress_pct": 0, "progress_text": "开始",
    })

    llm_tracer = LLMTracer()

    operator_doc = ""
    try:
        from agent.mcp_client import MCPClient
        _mcp = MCPClient()
        doc_result = await _mcp.get_document_content(operator_name)
        if doc_result and doc_result.get("content"):
            operator_doc = doc_result["content"]
    except Exception as e:
        logger.warning("Failed to fetch operator document for %s: %s", operator_name, e)

    state_input: PipelineState = {
        "operator_name": operator_name,
        "cases_path": cases_path,
        "cases_count": cases_count,
        "content": operator_doc,
        "server_info": server_info,
    }

    try:
        graph = build_pipeline([PipelineStage.EXECUTE])
        result = await graph.ainvoke(state_input, config={"callbacks": [llm_tracer]})

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
            logger.warning("Failed to persist execute events to DB: %s", e)

        exec_result = result.get("exec_result", {})
        error = result.get("error")
        status = "completed" if not error else "failed"

        # Save exec results to exec_results table
        if not error and case_ids and exec_result:
            try:
                from agent.db import save_exec_results as db_save_exec_results

                total = exec_result.get("total", 0)
                passed = exec_result.get("passed", 0)
                exec_records = []
                for i, cid in enumerate(case_ids):
                    exec_records.append({
                        "case_id": cid,
                        "passed": 1 if i < passed else 0,
                        "cpu_precision_passed": 1,
                    })
                db_save_exec_results(
                    task_id=run_id,
                    operator_name=operator_name,
                    results=exec_records,
                )
            except Exception as e:
                logger.warning("Failed to save exec results to DB: %s", e)

        try:
            db_complete_run(
                run_id,
                {
                    "status": status,
                    "operator_name": operator_name,
                    "exec_result": exec_result,
                },
                error=error,
            )
        except Exception as e:
            logger.warning("Failed to complete execute run in DB: %s", e)

        passed = exec_result.get("passed", 0)
        total = exec_result.get("total", 0)

        manager.emit(EventType.WORKFLOW_END, run_id, run.spans[run_id], {
            "agent_id": "execute",
            "message": (
                f"ExecuterAgent 完成。{passed}/{total} 用例通过"
                if not error else f"ExecuterAgent 失败: {error}"
            ),
            "summary": f"执行完成。{passed}/{total} 通过" if not error else f"失败: {error}",
            "progress_pct": 100, "progress_text": "完成" if not error else "失败",
            "result": {
                "status": status,
                "operator_name": operator_name,
                "exec_result": exec_result,
                "run_id": run_id,
            },
        })
        manager.complete_run(run_id, error=error)

    except Exception as e:
        logger.exception("Execute pipeline failed for run %s", run_id)
        manager.emit(EventType.WORKFLOW_ERROR, run_id, run.spans[run_id], {
            "agent_id": "execute", "error": str(e),
        })
        try:
            from agent.db import complete_run as db_complete_run
            db_complete_run(run_id, {}, error=str(e))
        except Exception as inner_e:
            logger.warning("Failed to mark execute run as failed in DB: %s", inner_e)
        manager.complete_run(run_id, error=str(e))
