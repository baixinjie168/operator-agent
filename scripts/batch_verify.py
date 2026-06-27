"""Batch verification script: generate test cases and execute for all operators with constraints.

Purpose: verify operator constraint quality by running the full
case_generate → test_execute pipeline for every operator that has
json_constraints in the database.

At execution time, cases are filtered by product:
  * If the operator supports the enabled server's product, only those
    cases are executed (``reason=matched``).
  * Otherwise, the operator's first supported product is used as a
    fallback (``reason=fallback_mismatch``).
  * When no server is configured, the operator's first supported
    product is used (``reason=fallback_no_server``).

Usage:
    uv run python -m scripts.batch_verify
    uv run python -m scripts.batch_verify --count 20 --seed 42
    uv run python -m scripts.batch_verify --operators aclnnAdaLayerNorm aclnnAdd
    uv run python -m scripts.batch_verify --concurrency 3 --skip-exec
    uv run python -m scripts.batch_verify --server-id 2
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
    query_servers as db_query_servers,
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


# ── Server & operator product lookups ──────────────────────────────────────

def get_enabled_server(server_id: int | None = None) -> dict | None:
    """Return the active server to use for execution.

    Resolution order:
      1. ``server_id`` if provided (lookup by id)
      2. The newest server row (ordered by id DESC, same as the API)
    """
    servers = db_query_servers()
    if not servers:
        return None
    if server_id is not None:
        for s in servers:
            if s.get("id") == server_id:
                return s
        logger.warning("未找到 id=%s 的服务器, 回退到最新服务器", server_id)
    return servers[0]


def get_operator_supported_products(doc_id: int) -> list[str]:
    """Return the list of product names that the operator's doc supports.

    Reads ``document_versions.product_support`` (JSON list of
    ``{product, support}``) and returns the names where ``support`` is true.
    Falls back to ``json_constraints.product_support`` (a plain list of
    platform names) when the first source is empty.
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT product_support, json_constraints FROM document_versions WHERE id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        return []

    product_support_raw, json_constraints_raw = row[0], row[1]

    # Primary source: product_support JSON list of {product, support}
    if product_support_raw:
        try:
            entries = json.loads(product_support_raw)
            if isinstance(entries, list):
                return [
                    e["product"] for e in entries
                    if isinstance(e, dict)
                    and e.get("product")
                    and str(e.get("support", "")).lower() in ("true", "1", "yes", "√", "✓", "是")
                ]
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # Fallback: json_constraints["product_support"] (list of platform names)
    if json_constraints_raw:
        try:
            constraints = json.loads(json_constraints_raw)
            if isinstance(constraints, dict):
                ps = constraints.get("product_support")
                if isinstance(ps, list):
                    return [p for p in ps if isinstance(p, str) and p]
        except (json.JSONDecodeError, TypeError):
            pass

    return []


def pick_target_product(
    server_product: str | None,
    operator_products: list[str],
) -> tuple[str | None, str]:
    """Decide which product to filter cases by.

    Returns ``(target_product, reason)``:
      * server_product matches an operator product → server_product ("matched")
      * server_product not configured but operator has products → first
        operator product ("fallback_no_server")
      * server_product is set but operator doesn't list it → first operator
        product ("fallback_mismatch")
      * no signal at all → (None, "no_filter")
    """
    if not server_product and not operator_products:
        return None, "no_filter"

    if server_product and server_product in operator_products:
        return server_product, "matched"

    if server_product and operator_products:
        return operator_products[0], "fallback_mismatch"

    if not server_product and operator_products:
        return operator_products[0], "fallback_no_server"

    return server_product, "server_only"


# ── Generate cases for one operator ─────────────────────────────────────────

async def run_case_generate(
    operator_name: str,
    count: int,
    seed: int,
    manager: RuntimeManager,
    target_product: str | None = None,
) -> dict:
    """Run case generation for a single operator. Returns result dict.

    Args:
        target_product: If set, the generator only produces cases for this
            product (others skipped). The final ``cases_count`` then equals
            ``count`` rather than ``count * num_products``. When ``None``,
            the generator iterates over all supported products.
    """
    run = manager.create_run(operator_name)
    run_id = run.run_id

    parent_id = find_parent_task(operator_name, "constraint_extract")
    content_hash = hashlib.sha256(
        f"batch:generator:{operator_name}:{count}:{seed}:{target_product or 'ALL'}:{run_id}".encode()
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
    if target_product:
        state_input["target_product"] = target_product

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
    exec_count: int | None = None,
    target_product: str | None = None,
    product_reason: str = "no_filter",
    server_info: dict | None = None,
) -> dict:
    """Run test execution for a single operator. Returns result dict.

    Args:
        exec_count: If specified and > 0, only execute the first N cases
            (truncated from the generated list). If None, execute all cases.
        target_product: If set, only execute cases whose ``supported_product``
            matches this value. Cases with an empty column fall back to
            ``case_data["supported_product"]`` for the comparison.
        product_reason: Diagnostic string describing how ``target_product``
            was chosen (e.g. ``matched``, ``fallback_mismatch``).
        server_info: Full server row (ip/username/password/...) forwarded
            into the ExecuterAgent state.  Without this the
            ``exec_run_atk`` node short-circuits with
            ``server_info is incomplete (need ip/username/password)``.
    """
    cases = db_query_test_cases(task_id=cases_task_id)
    if not cases:
        return {"run_id": None, "status": "skipped", "error": "no cases found"}

    cases_data = [c["case_data"] for c in cases]
    case_ids = [c["id"] for c in cases]

    # Apply product filter: keep only cases whose supported_product matches
    # the target. Read from the column first, then case_data JSON.
    if target_product:
        filtered: list[dict] = []
        filtered_ids: list[int] = []
        for c in cases:
            case_product = c.get("supported_product", "") or ""
            if not case_product and isinstance(c.get("case_data"), dict):
                case_product = c["case_data"].get("supported_product", "") or ""
            if case_product == target_product:
                filtered.append(c["case_data"])
                filtered_ids.append(c["id"])
        logger.info(
            "[%s] 产品过滤: target=%s reason=%s total=%d matched=%d",
            operator_name, target_product, product_reason,
            len(cases), len(filtered),
        )
        if not filtered:
            return {
                "run_id": None,
                "status": "skipped",
                "error": (
                    f"产品「{target_product}」下没有可用用例 "
                    f"(reason={product_reason})"
                ),
                "target_product": target_product,
                "product_reason": product_reason,
            }
        cases_data = filtered
        case_ids = filtered_ids

    # Apply execution count limit: truncate to the first N cases.
    if exec_count is not None and exec_count > 0 and len(cases_data) > exec_count:
        logger.info(
            "[%s] 限制执行数量: %d → %d", operator_name, len(cases_data), exec_count,
        )
        cases_data = cases_data[:exec_count]
        case_ids = case_ids[:exec_count]

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
        "server_info": server_info,
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

class ProgressReporter:
    """Async-safe progress reporter for concurrent batch execution.

    Tracks the number of started / completed operators and logs a single
    line per start / completion with the current index, cumulative
    elapsed time, average duration and ETA. All updates are serialised
    through an :class:`asyncio.Lock` so log lines never interleave with
    each other.
    """

    def __init__(self, total: int, concurrency: int = 1) -> None:
        self.total = total
        self.concurrency = max(1, concurrency)
        self._started = 0
        self.completed = 0
        self._start_time = time.time()
        self._durations: list[float] = []
        self._lock = asyncio.Lock()

    @property
    def pct(self) -> float:
        return (self.completed / self.total * 100) if self.total else 0.0

    async def op_started(self, name: str) -> None:
        async with self._lock:
            self._started += 1
            started_idx = self._started
            elapsed = time.time() - self._start_time
            in_progress = started_idx - self.completed
            logger.info(
                "[%d/%d ▶ 开始] %s | 累计耗时: %.1fs | 已完成: %d/%d (%.1f%%) | 进行中: %d",
                started_idx, self.total, name, elapsed,
                self.completed, self.total, self.pct, in_progress,
            )

    async def op_completed(self, name: str, duration: float) -> None:
        async with self._lock:
            self.completed += 1
            self._durations.append(duration)
            n = self.completed
            avg = sum(self._durations) / len(self._durations)
            remaining = self.total - n
            eta = remaining * avg / self.concurrency
            total_elapsed = time.time() - self._start_time
            logger.info(
                "[%d/%d ✓ 完成] %s | 本次: %.1fs | 平均: %.1fs | 累计: %.1fs | 预计剩余: %.1fs",
                n, self.total, name, duration, avg, total_elapsed, eta,
            )

    def summary_line(self) -> str:
        elapsed = time.time() - self._start_time
        avg = (
            sum(self._durations) / len(self._durations)
            if self._durations else 0.0
        )
        return (
            f"进度: {self.completed}/{self.total} ({self.pct:.1f}%) | "
            f"累计: {elapsed:.1f}s | 平均: {avg:.1f}s/算子"
        )


async def process_operator(
    op: dict,
    count: int,
    seed: int,
    skip_exec: bool,
    semaphore: asyncio.Semaphore,
    manager: RuntimeManager,
    server_info: dict | None = None,
    server_product: str | None = None,
    progress: ProgressReporter | None = None,
) -> dict:
    """Process a single operator: generate cases → execute.

    Args:
        server_info: Full server row used for SSH/SFTP during execution.
            Forwarded to ``run_test_execute`` so the ExecuterAgent can
            connect to the remote host. ``None`` will cause
            ``exec_run_atk`` to short-circuit with
            ``server_info is incomplete``.
        server_product: Product supported by the currently enabled server.
            If the operator's supported products include it, those cases
            are executed. Otherwise, the operator's first supported product
            is used as a fallback. ``None`` disables the filter.
        progress: Optional shared reporter. When set, ``op_started`` is
            called once at the start and ``op_completed`` once at the end
            (always, even on exception).
    """
    op_name = op["name"]
    result = {
        "operator_name": op_name,
        "doc_id": op["doc_id"],
        "version": op["version"],
        "case_generate": None,
        "test_execute": None,
        "target_product": None,
        "product_reason": None,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
    }

    logger.info("━━━ [%s] 开始处理 ━━━", op_name)
    t0 = time.time()
    if progress is not None:
        await progress.op_started(op_name)

    # Resolve target product once so we can narrow case generation to the
    # same product we will later execute. This way ``--count`` reflects the
    # target product's count, not ``count * num_products``.
    operator_products = get_operator_supported_products(op["doc_id"])
    target_product, reason = pick_target_product(server_product, operator_products)
    logger.info(
        "[%s] 产品选择: server=%s operator=%s → target=%s (%s)",
        op_name, server_product or "未配置",
        operator_products or "无", target_product or "全部", reason,
    )

    try:
        async with semaphore:
            logger.info(
                "[%s] 生成测试用例 (count=%d, seed=%d, target_product=%s)...",
                op_name, count, seed, target_product or "ALL",
            )
            gen_result = await run_case_generate(
                op_name, count, seed, manager,
                target_product=target_product,
            )
            result["case_generate"] = gen_result
            result["target_product"] = target_product
            result["product_reason"] = reason

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
                logger.info(
                    "[%s] 执行测试 (limit=%d, product=%s)...",
                    op_name, count, target_product or "全部",
                )
                exec_result = await run_test_execute(
                    op_name, gen_result["run_id"], manager,
                    exec_count=count,
                    target_product=target_product,
                    product_reason=reason,
                    server_info=server_info,
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
    finally:
        if progress is not None:
            await progress.op_completed(op_name, time.time() - t0)


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Batch verify operator constraints")
    parser.add_argument(
        "--count", type=int, default=10,
        help="Number of test cases to generate and execute per operator (default: 10)",
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
    parser.add_argument(
        "--server-id", type=int, default=None,
        help="Server id to use as the enabled server (default: newest server)",
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

    # Resolve the currently enabled server (used for product filter).
    server_info = get_enabled_server(args.server_id)
    server_product = (server_info or {}).get("supported_product") or None
    if server_info:
        logger.info(
            "当前启用服务器: id=%s name=%s supported_product=%r",
            server_info.get("id"), server_info.get("name"), server_product,
        )
    else:
        logger.warning("未配置服务器, 执行阶段将不按产品过滤")

    logger.info("=" * 60)
    logger.info("批量验证开始")
    logger.info("  算子数量: %d", len(operators))
    logger.info("  生成/执行用例数: %d / 算子", args.count)
    logger.info("  随机种子: %d", args.seed)
    logger.info("  并发限制: %d", args.concurrency)
    logger.info("  跳过执行: %s", args.skip_exec)
    logger.info("  服务器产品: %s", server_product or "未配置")
    logger.info("  算子列表: %s", [op["name"] for op in operators])
    logger.info("=" * 60)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manager = RuntimeManager()
    semaphore = asyncio.Semaphore(args.concurrency)
    progress = ProgressReporter(total=len(operators), concurrency=args.concurrency)

    t_start = time.time()

    # Process all operators concurrently (bounded by semaphore)
    tasks = [
        process_operator(
            op, args.count, args.seed, args.skip_exec, semaphore, manager,
            server_info=server_info, server_product=server_product, progress=progress,
        )
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
            "server": (
                {
                    "id": server_info.get("id"),
                    "name": server_info.get("name"),
                    "supported_product": server_product,
                }
                if server_info else None
            ),
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
    logger.info("  进度:         %s", progress.summary_line())
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
    logger.info("%-30s %-14s %-18s %-10s %-10s %-10s %-8s",
                "算子", "目标产品", "产品原因", "用例数", "执行数", "通过数", "通过率")
    logger.info("-" * 110)
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
        # Truncate long product names for table alignment
        target = (r.get("target_product") or "全部")
        if len(target) > 12:
            target = target[:10] + ".."
        reason = r.get("product_reason") or "-"
        logger.info(
            "%-30s %-14s %-18s %-10s %-10s %-10s %-8s %s",
            op, target, reason, cases_n, exec_n, pass_n, rate, gen_status,
        )


if __name__ == "__main__":
    asyncio.run(main())
