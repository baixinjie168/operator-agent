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
from pydantic import BaseModel, Field

from agent.db import (
    complete_run as db_complete_run,
)
from agent.db import (
    create_run as db_create_run,
)
from agent.db import (
    save_events as db_save_events,
)
from agent.db import (
    update_run_doc_id as db_update_run_doc_id,
)
from agent.graph import PipelineStage, build_pipeline
from agent.nodes.init_doc import init_doc_node as _init_doc
from agent.runtime import EventType, LLMTracer, RuntimeManager, traced_node
from agent.schemas.upload import UploadResponse
from agent.utils.file_utils import extract_operator_name_from_content as _extract_operator_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])


class ReExtractRequest(BaseModel):
    operator_name: str = Field(..., min_length=1)


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

    manager = _get_manager(request)
    run = manager.create_run(operator_name)
    db_create_run(
        run.run_id,
        operator_name,
        content_hash,
        task_type="constraint_extract",
    )

    state_input = {"operator_name": operator_name, "content": content, "content_hash": content_hash, "run_id": run.run_id}

    asyncio.create_task(_run_pipeline(run.run_id, state_input, manager, stages=[PipelineStage.EXTRACT]))

    return UploadResponse(success=True, task_id=run.run_id, operator_name=operator_name)


@router.post("/re-extract-constraints", response_model=UploadResponse)
async def re_extract_constraints(body: ReExtractRequest, request: Request) -> UploadResponse:
    """Re-extract constraints for an operator using its existing document.

    Fetches the latest document content from the database and runs the
    constraint extraction pipeline without requiring a new file upload.
    """
    operator_name = body.operator_name.strip()
    if not operator_name:
        return UploadResponse(success=False, error="operator_name is required")

    # Fetch the latest document content from the database
    try:
        from agent.mcp_client import MCPClient
        mcp = MCPClient()
        doc_result = await mcp.get_document_content(operator_name)
        if not doc_result or not doc_result.get("content"):
            return UploadResponse(
                success=False,
                operator_name=operator_name,
                error=f"未找到算子 {operator_name} 的文档，请先上传文档",
            )
        content = doc_result["content"]
    except Exception as e:
        logger.exception("Failed to fetch document for %s", operator_name)
        return UploadResponse(
            success=False,
            operator_name=operator_name,
            error=f"获取文档失败: {e}",
        )

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    manager = _get_manager(request)
    run = manager.create_run(operator_name)
    db_create_run(
        run.run_id,
        operator_name,
        content_hash,
        task_type="constraint_extract",
    )

    state_input = {"operator_name": operator_name, "content": content, "content_hash": content_hash, "run_id": run.run_id}

    asyncio.create_task(_run_pipeline(run.run_id, state_input, manager, stages=[PipelineStage.EXTRACT]))

    return UploadResponse(success=True, task_id=run.run_id, operator_name=operator_name)


async def _run_pipeline(
    run_id: str,
    state_input: dict,
    manager: RuntimeManager,
    *,
    stages: list[PipelineStage] | None = None,
) -> None:
    """Run pipeline with RuntimeManager observability.  Nodes emit events via @traced_node.

    For full pipeline (extract + generate + execute), creates 3 separate tasks:
    - constraint_extract task (current run_id)
    - case_generate task (new run_id, parent = constraint_extract)
    - test_execute task (new run_id, parent = case_generate)

    For extraction-only runs, uses the current run_id.

    Args:
        stages: Which pipeline stages to run.  Defaults to the full pipeline
            (extract + generate + execute).  Pass ``[PipelineStage.EXTRACT]``
            for extraction-only runs (used by ``re-extract-constraints``).
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

        if stages is None:
            stages = [PipelineStage.EXTRACT, PipelineStage.GENERATE, PipelineStage.EXECUTE]

        # For full pipeline, run each stage as a separate task
        is_full_pipeline = len(stages) > 1

        if is_full_pipeline:
            # Stage 1: EXTRACT (use current run_id)
            extract_graph = build_pipeline([PipelineStage.EXTRACT])
            state_input["run_id"] = run_id
            extract_result = await extract_graph.ainvoke(state_input, config={"callbacks": [llm_tracer]})
            _persist_to_db(run_id, run, extract_result, manager)
            manager.complete_run(run_id)

            if extract_result.get("error"):
                logger.warning("Extract stage failed, stopping pipeline")
                return

            # Stage 2: GENERATE (create new task)
            from agent.db import create_run as db_create_run
            operator_name = state_input.get("operator_name", "")
            content_hash = state_input.get("content_hash", "")

            gen_run = manager.create_run(operator_name)
            gen_run_id = gen_run.run_id
            db_create_run(
                gen_run_id,
                operator_name,
                content_hash,
                task_type="case_generate",
                parent_task_id=run_id,
            )

            gen_ctx = manager.enter_context(gen_run_id)
            manager.emit(EventType.WORKFLOW_START, gen_run_id, gen_run.spans[gen_run_id], {
                "agent_id": "case",
                "node_id": "case_match_model",
                "message": f"GeneratorAgent 开始为 {operator_name} 生成测试用例...",
                "step_index": 0, "progress_pct": 0, "progress_text": "开始",
            })

            gen_state = {**state_input, "run_id": gen_run_id, "cases_count": 10, "cases_seed": 42}
            gen_graph = build_pipeline([PipelineStage.GENERATE])
            gen_result = await gen_graph.ainvoke(gen_state, config={"callbacks": [llm_tracer]})
            _persist_to_db(gen_run_id, gen_run, gen_result, manager)
            manager.complete_run(gen_run_id)

            if gen_result.get("error"):
                logger.warning("Generate stage failed, stopping pipeline")
                return

            # Stage 3: EXECUTE (create new task)
            exec_run = manager.create_run(operator_name)
            exec_run_id = exec_run.run_id
            db_create_run(
                exec_run_id,
                operator_name,
                content_hash,
                task_type="test_execute",
                parent_task_id=gen_run_id,
            )

            exec_ctx = manager.enter_context(exec_run_id)
            manager.emit(EventType.WORKFLOW_START, exec_run_id, exec_run.spans[exec_run_id], {
                "agent_id": "execute",
                "node_id": "exec_generate_atk",
                "message": f"ExecuterAgent 开始为 {operator_name} 执行测试用例...",
                "step_index": 0, "progress_pct": 0, "progress_text": "开始",
            })

            cases_list = gen_result.get("cases", [])
            cases_path = gen_result.get("cases_path", "")
            exec_state = {
                "run_id": exec_run_id,
                "operator_name": operator_name,
                "cases_path": cases_path,
                "cases_count": len(cases_list),
                "content": state_input.get("content", ""),
            }
            exec_graph = build_pipeline([PipelineStage.EXECUTE])
            exec_result = await exec_graph.ainvoke(exec_state, config={"callbacks": [llm_tracer]})
            _persist_to_db(exec_run_id, exec_run, exec_result, manager)

            # Save exec results
            exec_result_data = exec_result.get("exec_result", {})
            if exec_result_data and cases_list:
                try:
                    from agent.db import query_test_cases as db_query_test_cases
                    from agent.db import save_exec_results as db_save_exec_results
                    saved_cases = db_query_test_cases(task_id=gen_run_id)
                    if saved_cases:
                        case_ids = [c["id"] for c in saved_cases]
                        passed = exec_result_data.get("passed", 0)
                        exec_records = []
                        for i, cid in enumerate(case_ids):
                            exec_records.append({
                                "case_id": cid,
                                "passed": 1 if i < passed else 0,
                                "cpu_precision_passed": 1,
                            })
                        db_save_exec_results(
                            task_id=exec_run_id,
                            operator_name=operator_name,
                            results=exec_records,
                        )
                except Exception as e:
                    logger.warning("Failed to save exec results: %s", e)

            manager.complete_run(exec_run_id)
        else:
            # Single stage run (e.g., extract-only)
            state_input["run_id"] = run_id
            graph = build_pipeline(stages)
            result = await graph.ainvoke(state_input, config={"callbacks": [llm_tracer]})

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
                "summary": f"提取完成。{sc} sections, {pc} 参数, {prod} 产品。",
                "progress_pct": 100, "progress_text": "完成",
                "result": {
                    "status": status, "version": version,
                    "sections_count": sc, "parameters_count": pc, "product_count": prod,
                    "doc_id": doc_id, "operator_name": op_name, "run_id": run_id,
                },
            })

            # Persist to DB AFTER emitting WORKFLOW_END so task.completed event is included
            _persist_to_db(run_id, run, result, manager)
            manager.complete_run(run_id)

    except Exception as e:
        logger.exception("Pipeline execution failed for run %s", run_id)
        manager.emit(EventType.WORKFLOW_ERROR, run_id, run.spans[run_id], {
            "agent_id": "doc", "error": str(e),
        })
        manager.complete_run(run_id, error=str(e))


class ContinuePipelineRequest(BaseModel):
    operator_name: str = Field(..., min_length=1)
    extract_task_id: str = Field(..., min_length=1)


@router.post("/continue-pipeline", response_model=UploadResponse)
async def continue_pipeline(body: ContinuePipelineRequest, request: Request) -> UploadResponse:
    """Continue pipeline after constraint extraction: generate cases + execute tests.

    Creates case_generate and test_execute tasks with parent_task_id pointing to the extract task.
    """
    operator_name = body.operator_name.strip()
    extract_task_id = body.extract_task_id.strip()

    if not operator_name:
        return UploadResponse(success=False, error="operator_name is required")
    if not extract_task_id:
        return UploadResponse(success=False, error="extract_task_id is required")

    # Fetch the latest document content from the database
    try:
        from agent.mcp_client import MCPClient
        mcp = MCPClient()
        doc_result = await mcp.get_document_content(operator_name)
        if not doc_result or not doc_result.get("content"):
            return UploadResponse(
                success=False,
                operator_name=operator_name,
                error=f"未找到算子 {operator_name} 的文档",
            )
        content = doc_result["content"]
    except Exception as e:
        logger.exception("Failed to fetch document for %s", operator_name)
        return UploadResponse(
            success=False,
            operator_name=operator_name,
            error=f"获取文档失败: {e}",
        )

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    manager = _get_manager(request)

    # Create case_generate task
    gen_run = manager.create_run(operator_name)
    gen_run_id = gen_run.run_id
    db_create_run(
        gen_run_id,
        operator_name,
        content_hash,
        task_type="case_generate",
        parent_task_id=extract_task_id,
    )

    state_input = {
        "operator_name": operator_name,
        "content": content,
        "content_hash": content_hash,
        "run_id": gen_run_id,
        "parent_task_id": extract_task_id,
    }

    asyncio.create_task(_run_generate_and_execute(gen_run_id, state_input, manager))

    return UploadResponse(success=True, task_id=gen_run_id, operator_name=operator_name)


async def _run_generate_and_execute(
    gen_run_id: str,
    state_input: dict,
    manager: RuntimeManager,
) -> None:
    """Run case generation and then test execution as separate tasks."""
    from agent.runtime import EventType, LLMTracer

    gen_run = manager.get_run(gen_run_id)
    if not gen_run:
        return

    await asyncio.sleep(0.3)

    operator_name = state_input.get("operator_name", "")
    extract_task_id = state_input.get("parent_task_id", "")

    manager.emit(EventType.WORKFLOW_START, gen_run_id, gen_run.spans[gen_run_id], {
        "agent_id": "case",
        "node_id": "case_match_model",
        "message": f"GeneratorAgent 开始为 {operator_name} 生成测试用例...",
        "step_index": 0, "progress_pct": 0, "progress_text": "开始",
    })

    llm_tracer = LLMTracer()

    try:
        # Stage 1: GENERATE
        gen_state = {**state_input, "cases_count": 10, "cases_seed": 42}
        gen_graph = build_pipeline([PipelineStage.GENERATE])
        gen_result = await gen_graph.ainvoke(gen_state, config={"callbacks": [llm_tracer]})

        # Emit WORKFLOW_END for generate task before persisting
        gen_cases_count = gen_result.get("cases_count", 0)
        gen_status = "completed" if not gen_result.get("error") else "failed"
        manager.emit(EventType.WORKFLOW_END, gen_run_id, gen_run.spans[gen_run_id], {
            "agent_id": "case",
            "message": f"GeneratorAgent 完成。生成 {gen_cases_count} 个用例",
            "summary": f"用例生成完成。{gen_cases_count} 个用例",
            "progress_pct": 100, "progress_text": "完成",
            "result": {
                "status": gen_status,
                "cases_count": gen_cases_count,
                "operator_name": operator_name,
                "run_id": gen_run_id,
            },
        })

        _persist_to_db(gen_run_id, gen_run, gen_result, manager)
        manager.complete_run(gen_run_id)

        if gen_result.get("error"):
            logger.warning("Generate stage failed, stopping pipeline")
            return

        # Stage 2: EXECUTE (create new task)
        exec_run = manager.create_run(operator_name)
        exec_run_id = exec_run.run_id
        content_hash = state_input.get("content_hash", "")
        db_create_run(
            exec_run_id,
            operator_name,
            content_hash,
            task_type="test_execute",
            parent_task_id=gen_run_id,
        )

        exec_ctx = manager.enter_context(exec_run_id)
        manager.emit(EventType.WORKFLOW_START, exec_run_id, exec_run.spans[exec_run_id], {
            "agent_id": "execute",
            "node_id": "exec_generate_atk",
            "message": f"ExecuterAgent 开始为 {operator_name} 执行测试用例...",
            "step_index": 0, "progress_pct": 0, "progress_text": "开始",
        })

        cases_list = gen_result.get("cases", [])
        cases_path = gen_result.get("cases_path", "")
        exec_state = {
            "run_id": exec_run_id,
            "operator_name": operator_name,
            "cases_path": cases_path,
            "cases_count": len(cases_list),
            "content": state_input.get("content", ""),
        }
        exec_graph = build_pipeline([PipelineStage.EXECUTE])
        exec_result = await exec_graph.ainvoke(exec_state, config={"callbacks": [llm_tracer]})

        # Emit WORKFLOW_END for execute task before persisting
        exec_result_data = exec_result.get("exec_result", {})
        exec_passed = exec_result_data.get("passed", 0)
        exec_total = exec_result_data.get("total", 0)
        exec_status = "completed" if not exec_result.get("error") else "failed"
        manager.emit(EventType.WORKFLOW_END, exec_run_id, exec_run.spans[exec_run_id], {
            "agent_id": "execute",
            "message": f"ExecuterAgent 完成。{exec_passed}/{exec_total} 用例通过",
            "summary": f"执行完成。{exec_passed}/{exec_total} 通过",
            "progress_pct": 100, "progress_text": "完成",
            "result": {
                "status": exec_status,
                "exec_result": exec_result_data,
                "operator_name": operator_name,
                "run_id": exec_run_id,
            },
        })

        _persist_to_db(exec_run_id, exec_run, exec_result, manager)

        # Save exec results
        if exec_result_data and cases_list:
            try:
                from agent.db import query_test_cases as db_query_test_cases
                from agent.db import save_exec_results as db_save_exec_results
                saved_cases = db_query_test_cases(task_id=gen_run_id)
                if saved_cases:
                    case_ids = [c["id"] for c in saved_cases]
                    passed = exec_result_data.get("passed", 0)
                    exec_records = []
                    for i, cid in enumerate(case_ids):
                        exec_records.append({
                            "case_id": cid,
                            "passed": 1 if i < passed else 0,
                            "cpu_precision_passed": 1,
                        })
                    db_save_exec_results(
                        task_id=exec_run_id,
                        operator_name=operator_name,
                        results=exec_records,
                    )
            except Exception as e:
                logger.warning("Failed to save exec results: %s", e)

        manager.complete_run(exec_run_id)

    except Exception as e:
        logger.exception("Generate and execute pipeline failed for run %s", gen_run_id)
        manager.emit(EventType.WORKFLOW_ERROR, gen_run_id, gen_run.spans[gen_run_id], {
            "agent_id": "case", "error": str(e),
        })
        manager.complete_run(gen_run_id, error=str(e))


def _persist_to_db(run_id: str, run, result: dict, manager: RuntimeManager) -> None:
    """Persist all runtime events + spans to DB directly (no MCP)."""
    events_payload = []
    for evt in run.events:
        sse = evt.to_sse()
        # Use the alias name (e.g. "node.started") to match what the SSE
        # stream emits.  The frontend's _eventRouteMap dispatches on these
        # alias names — if we stored the raw enum value ("node.start")
        # instead, replays after a backend restart would silently fail to
        # dispatch any events.
        events_payload.append({
            "seq": evt.seq,
            "event_type": sse["event_type"],
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


    return None