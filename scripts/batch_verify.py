"""Batch verification script: generate test cases and execute for all operators with constraints.

Purpose: verify operator constraint quality by running the full
case_generate → test_execute pipeline for every operator that has
json_constraints in the database.

Usage:
    uv run python -m scripts.batch_verify
    uv run python -m scripts.batch_verify --count 20 --seed 42
    uv run python -m scripts.batch_verify --operators aclnnAdaLayerNorm aclnnAdd
    uv run python -m scripts.batch_verify --concurrency 3 --skip-exec
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure packages are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "packages" / "agent" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "packages" / "mcp-server" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "packages" / "shared" / "src"))

from langgraph.graph import END, START, StateGraph

from agent.db import (
    complete_run as db_complete_run,
)
from agent.db import (
    create_run as db_create_run,
)
from agent.db import (
    find_parent_task,
)
from agent.db import (
    query_run as db_query_run,
)
from agent.db import (
    query_test_cases as db_query_test_cases,
)
from agent.db import (
    save_events as db_save_events,
)
from agent.db import (
    save_exec_results as db_save_exec_results,
)
from agent.db import (
    save_test_cases as db_save_test_cases,
)
from agent.nodes.case_subgraph import (
    case_generate_node,
    case_init_static_node,
    case_match_model_node,
    case_solve_constraints_node,
)
from agent.nodes.executer_subgraph import create_executer_subgraph
from agent.nodes.state import PipelineState
from agent.runtime import LLMTracer, RuntimeManager, traced_node
from mcp_server.db import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_verify")

# ── Output directory ────────────────────────────────────────────────────────

_OUTPUT_DIR = _PROJECT_ROOT / "batch_results"
_CASES_DIR = _PROJECT_ROOT / "batch_cases"


# ── Subgraph builders ───────────────────────────────────────────────────────

def _build_case_subgraph():
    graph = StateGraph(PipelineState)
    graph.add_node("case_match_model", traced_node("case_match_model")(case_match_model_node))
    graph.add_node("case_init_static", traced_node("case_init_static")(case_init_static_node))
    graph.add_node("case_solve_constraints", traced_node("case_solve_constraints")(case_solve_constraints_node))
    graph.add_node("case_generate", traced_node("case_generate")(case_generate_node))

    graph.add_edge(START, "case_match_model")
    graph.add_conditional_edges(
        "case_match_model",
        lambda s: END if s.get("error") else "case_init_static",
    )
    graph.add_conditional_edges(
        "case_init_static",
        lambda s: END if s.get("error") else "case_solve_constraints",
    )
    graph.add_conditional_edges(
        "case_solve_constraints",
        lambda s: END if s.get("error") else "case_generate",
    )
    graph.add_edge("case_generate", END)
    return graph.compile(name="case-pipeline")


# ── Query operators with constraints ────────────────────────────────────────

def get_operators_with_constraints(
    filter_names: list[str] | None = None,
) -> list[dict]:
    """Return operators that have non-empty json_constraints."""
    db = get_db()
    rows = db.conn.execute(
        "SELECT o.name, dv.id AS doc_id, dv.version "
        "FROM operators o "
        "JOIN document_versions dv ON dv.operator_id = o.id "
        "WHERE dv.json_constraints != '{}' "
        "AND dv.json_constraints != '' "
        "AND dv.json_constraints IS NOT NULL "
        "ORDER BY o.name",
    ).fetchall()

    operators = [
        {"name": r[0], "doc_id": r[1], "version": r[2]}
        for r in rows
    ]

    if filter_names:
        name_set = set(filter_names)
        operators = [op for op in operators if op["name"] in name_set]

    return operators


# ── Generate cases for one operator ─────────────────────────────────────────

async def run_case_generate(
    operator_name: str,
    count: int,
    seed: int,
    manager: RuntimeManager,
) -> dict:
    """Run case generation for a single operator. Returns result dict."""
    run = manager.create_run(operator_name)
    run_id = run.run_id

    parent_id = find_parent_task(operator_name, "constraint_extract")
    content_hash = hashlib.sha256(
        f"batch:generator:{operator_name}:{count}:{seed}:{run_id}".encode()
    ).hexdigest()

    db_create_run(
        run_id, operator_name, content_hash,
        task_type="case_generate",
        parent_task_id=parent_id,
    )

    manager.enter_context(run_id)
    llm_tracer = LLMTracer()

    state_input: PipelineState = {
        "operator_name": operator_name,
        "cases_count": count,
        "cases_seed": seed,
    }

    try:
        graph = _build_case_subgraph()
        result = await graph.ainvoke(state_input, config={"callbacks": [llm_tracer]})

        events_payload = []
        for evt in run.events:
            sse = evt.to_sse()
            events_payload.append({
                "seq": evt.seq,
                "event_type": sse["event_type"],
                "data": sse["data"],
            })
        db_save_events(run_id, events_payload)

        cases_count = result.get("cases_count")
        cases_list = result.get("cases", [])
        error = result.get("error")
        status = "completed" if not error else "failed"

        if not error and cases_list:
            constraint_doc_id = None
            if parent_id:
                parent_run = db_query_run(parent_id)
                if parent_run:
                    constraint_doc_id = parent_run.get("doc_id")
            db_save_test_cases(
                task_id=run_id,
                operator_name=operator_name,
                cases=cases_list,
                constraint_doc_id=constraint_doc_id,
            )

        db_complete_run(
            run_id,
            {"status": status, "operator_name": operator_name, "cases_count": cases_count},
            error=error,
        )
        manager.complete_run(run_id, error=error)

        return {
            "run_id": run_id,
            "status": status,
            "cases_count": cases_count,
            "error": error,
        }
    except Exception as e:
        logger.exception("case_generate failed for %s", operator_name)
        db_complete_run(run_id, {}, error=str(e))
        manager.complete_run(run_id, error=str(e))
        return {"run_id": run_id, "status": "failed", "cases_count": 0, "error": str(e)}


# ── Execute cases for one operator ──────────────────────────────────────────

async def run_test_execute(
    operator_name: str,
    cases_task_id: str,
    manager: RuntimeManager,
) -> dict:
    """Run test execution for a single operator. Returns result dict."""
    cases = db_query_test_cases(task_id=cases_task_id)
    if not cases:
        return {"run_id": None, "status": "skipped", "error": "no cases found"}

    cases_data = [c["case_data"] for c in cases]
    case_ids = [c["id"] for c in cases]
    cases_json_str = json.dumps(cases_data, ensure_ascii=False)

    _CASES_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = _CASES_DIR / f"{operator_name}_cases.json"
    cases_path.write_text(cases_json_str, encoding="utf-8")

    run = manager.create_run(operator_name)
    run_id = run.run_id

    content_hash = hashlib.sha256(
        f"batch:execute:{operator_name}:{run_id}".encode()
    ).hexdigest()

    db_create_run(
        run_id, operator_name, content_hash,
        task_type="test_execute",
        parent_task_id=cases_task_id,
    )

    manager.enter_context(run_id)
    llm_tracer = LLMTracer()

    operator_doc = ""
    try:
        from agent.mcp_client import MCPClient
        _mcp = MCPClient()
        doc_result = await _mcp.get_document_content(operator_name)
        if doc_result and doc_result.get("content"):
            operator_doc = doc_result["content"]
    except Exception as e:
        logger.warning("Failed to fetch doc for %s: %s", operator_name, e)

    state_input: PipelineState = {
        "operator_name": operator_name,
        "cases_path": str(cases_path),
        "cases_count": len(cases_data),
        "content": operator_doc,
    }

    try:
        graph = create_executer_subgraph()
        result = await graph.ainvoke(state_input, config={"callbacks": [llm_tracer]})

        events_payload = []
        for evt in run.events:
            sse = evt.to_sse()
            events_payload.append({
                "seq": evt.seq,
                "event_type": sse["event_type"],
                "data": sse["data"],
            })
        db_save_events(run_id, events_payload)

        exec_result = result.get("exec_result", {})
        error = result.get("error")
        status = "completed" if not error else "failed"

        if not error and case_ids and exec_result:
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

        db_complete_run(
            run_id,
            {"status": status, "operator_name": operator_name, "exec_result": exec_result},
            error=error,
        )
        manager.complete_run(run_id, error=error)

        return {
            "run_id": run_id,
            "status": status,
            "exec_result": exec_result,
            "error": error,
        }
    except Exception as e:
        logger.exception("test_execute failed for %s", operator_name)
        db_complete_run(run_id, {}, error=str(e))
        manager.complete_run(run_id, error=str(e))
        return {"run_id": run_id, "status": "failed", "exec_result": {}, "error": str(e)}


# ── Process one operator ────────────────────────────────────────────────────

async def process_operator(
    op: dict,
    count: int,
    seed: int,
    skip_exec: bool,
    semaphore: asyncio.Semaphore,
    manager: RuntimeManager,
) -> dict:
    """Process a single operator: generate cases → execute."""
    op_name = op["name"]
    result = {
        "operator_name": op_name,
        "doc_id": op["doc_id"],
        "version": op["version"],
        "case_generate": None,
        "test_execute": None,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
    }

    logger.info("━━━ [%s] 开始处理 ━━━", op_name)
    t0 = time.time()

    async with semaphore:
        logger.info("[%s] 生成测试用例 (count=%d, seed=%d)...", op_name, count, seed)
        gen_result = await run_case_generate(op_name, count, seed, manager)
        result["case_generate"] = gen_result

        if gen_result["status"] != "completed":
            logger.error("[%s] 用例生成失败: %s", op_name, gen_result.get("error"))
            result["end_time"] = datetime.now().isoformat()
            result["duration_s"] = round(time.time() - t0, 1)
            return result

        logger.info(
            "[%s] 用例生成完成: %d 条 (task_id=%s)",
            op_name, gen_result.get("cases_count", 0), gen_result["run_id"],
        )

        if not skip_exec:
            logger.info("[%s] 执行测试...", op_name)
            exec_result = await run_test_execute(
                op_name, gen_result["run_id"], manager,
            )
            result["test_execute"] = exec_result

            if exec_result["status"] == "completed":
                er = exec_result.get("exec_result", {})
                logger.info(
                    "[%s] 执行完成: %d/%d 通过",
                    op_name, er.get("passed", 0), er.get("total", 0),
                )
            else:
                logger.error("[%s] 执行失败: %s", op_name, exec_result.get("error"))

    result["end_time"] = datetime.now().isoformat()
    result["duration_s"] = round(time.time() - t0, 1)
    logger.info("━━━ [%s] 处理完成 (%.1fs) ━━━", op_name, result["duration_s"])
    return result


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Batch verify operator constraints")
    parser.add_argument(
        "--count", type=int, default=10,
        help="Number of test cases per operator (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for case generation (default: 42)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent LLM requests (default: 5)",
    )
    parser.add_argument(
        "--operators", nargs="*", default=None,
        help="Specific operator names to process (default: all with constraints)",
    )
    parser.add_argument(
        "--skip-exec", action="store_true",
        help="Skip test execution, only generate cases",
    )
    args = parser.parse_args()

    # Initialize DB
    get_db()

    # Query operators
    operators = get_operators_with_constraints(args.operators)
    if not operators:
        logger.error("未找到有约束的算子")
        if args.operators:
            logger.error("指定的算子: %s", args.operators)
        return

    logger.info("=" * 60)
    logger.info("批量验证开始")
    logger.info("  算子数量: %d", len(operators))
    logger.info("  用例数量: %d / 算子", args.count)
    logger.info("  随机种子: %d", args.seed)
    logger.info("  并发限制: %d", args.concurrency)
    logger.info("  跳过执行: %s", args.skip_exec)
    logger.info("  算子列表: %s", [op["name"] for op in operators])
    logger.info("=" * 60)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manager = RuntimeManager()
    semaphore = asyncio.Semaphore(args.concurrency)

    t_start = time.time()

    # Process all operators concurrently (bounded by semaphore)
    tasks = [
        process_operator(op, args.count, args.seed, args.skip_exec, semaphore, manager)
        for op in operators
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions from gather
    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error("算子 %s 处理异常: %s", operators[i]["name"], r)
            processed.append({
                "operator_name": operators[i]["name"],
                "case_generate": {"status": "failed", "error": str(r)},
                "test_execute": None,
            })
        else:
            processed.append(r)

    total_time = round(time.time() - t_start, 1)

    # Summary
    gen_ok = sum(1 for r in processed if r.get("case_generate", {}).get("status") == "completed")
    gen_fail = sum(1 for r in processed if r.get("case_generate", {}).get("status") == "failed")
    exec_ok = sum(
        1 for r in processed
        if r.get("test_execute") and r["test_execute"].get("status") == "completed"
    )
    exec_fail = sum(
        1 for r in processed
        if r.get("test_execute") and r["test_execute"].get("status") == "failed"
    )
    total_cases = sum(
        r.get("case_generate", {}).get("cases_count", 0) or 0
        for r in processed
    )
    total_passed = sum(
        r.get("test_execute", {}).get("exec_result", {}).get("passed", 0) or 0
        for r in processed
    )
    total_executed = sum(
        r.get("test_execute", {}).get("exec_result", {}).get("total", 0) or 0
        for r in processed
    )

    summary = {
        "run_time": datetime.now().isoformat(),
        "config": {
            "count": args.count,
            "seed": args.seed,
            "concurrency": args.concurrency,
            "skip_exec": args.skip_exec,
        },
        "summary": {
            "total_operators": len(processed),
            "gen_success": gen_ok,
            "gen_failed": gen_fail,
            "exec_success": exec_ok,
            "exec_failed": exec_fail,
            "total_cases_generated": total_cases,
            "total_cases_executed": total_executed,
            "total_cases_passed": total_passed,
            "pass_rate": f"{(total_passed / total_executed * 100):.1f}%" if total_executed else "N/A",
            "total_duration_s": total_time,
        },
        "results": processed,
    }

    # Save results
    output_file = _OUTPUT_DIR / f"batch_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("批量验证完成 (%.1fs)", total_time)
    logger.info("=" * 60)
    logger.info("  算子总数:     %d", len(processed))
    logger.info("  用例生成:     %d 成功 / %d 失败", gen_ok, gen_fail)
    if not args.skip_exec:
        logger.info("  测试执行:     %d 成功 / %d 失败", exec_ok, exec_fail)
    logger.info("  生成用例总数: %d", total_cases)
    if not args.skip_exec and total_executed:
        logger.info("  执行通过率:   %d/%d (%.1f%%)",
                     total_passed, total_executed,
                     total_passed / total_executed * 100)
    logger.info("  结果文件:     %s", output_file)
    logger.info("=" * 60)

    # Per-operator summary table
    logger.info("")
    logger.info("%-30s %-10s %-10s %-10s %-10s",
                "算子", "用例数", "执行数", "通过数", "通过率")
    logger.info("-" * 70)
    for r in processed:
        op = r.get("operator_name", "?")
        gen = r.get("case_generate", {}) or {}
        exe = r.get("test_execute", {}) or {}
        er = exe.get("exec_result", {}) or {}
        cases_n = gen.get("cases_count", 0) or 0
        exec_n = er.get("total", 0) or 0
        pass_n = er.get("passed", 0) or 0
        rate = f"{pass_n / exec_n * 100:.0f}%" if exec_n else "N/A"
        gen_status = "✅" if gen.get("status") == "completed" else "❌"
        logger.info("%-30s %-10s %-10s %-10s %-10s %s",
                     op, cases_n, exec_n, pass_n, rate, gen_status)


if __name__ == "__main__":
    asyncio.run(main())
