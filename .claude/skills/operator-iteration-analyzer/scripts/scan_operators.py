#!/usr/bin/env python3
"""扫描 operator-agent 项目运行数据，汇总每个算子在各阶段的状态。

数据源：
- SQLite: data/operator_agent.db 的 document_versions.json_constraints
- 产物文件：batch_cases/*.json, cases/*.json, execution_results/*/result.json
- 日志文件：logs/generate_case_*.log
- pipeline_runs 表（用作日志缺失时的 fallback）

Usage:
    python scan_operators.py --project-root <path> [--operator <name>] --output <json>

输出：JSON 格式的状态汇总
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

# 确保 Windows 控制台能输出中文
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #


@dataclass
class CaseFile:
    """生成用例的产物文件。"""

    path: str
    product: str | None = None  # 来自文件名后缀


@dataclass
class ReportRecord:
    """单条执行报告。"""

    id: int | str
    run_result: str
    failure_reason: str | None = None


@dataclass
class OperatorStatus:
    """单个算子的端到端运行状态。"""

    operator_name: str
    doc_id: int | None = None
    constraint_status: str = "missing"  # success / failed / missing
    constraint_keys_count: int = 0
    case_generation_status: str = "missing"  # success / failed / missing
    case_files: list[CaseFile] = field(default_factory=list)
    execution_status: str = "missing"  # success / failed / partial / missing
    execution_dir: str | None = None
    result_json_path: str | None = None
    report_records: list[ReportRecord] = field(default_factory=list)
    log_path: str | None = None
    pipeline_runs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ScanSummary:
    """全量扫描的汇总。"""

    total: int = 0
    constraint_success: int = 0
    case_generation_success: int = 0
    case_generation_failed: int = 0
    case_generation_missing: int = 0
    execution_success: int = 0
    execution_failed: int = 0
    execution_partial: int = 0
    execution_missing: int = 0


# --------------------------------------------------------------------------- #
# 文件发现
# --------------------------------------------------------------------------- #


_PRODUCT_SAFE_RE = re.compile(r"[\\/:*?\"<>|]")


def _sanitize_product_name(product: str) -> str:
    """与 case_subgraph/generate.py 中的 _sanitize_product_name 保持一致。"""
    safe = _PRODUCT_SAFE_RE.sub("_", product)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe or "default"


def _find_case_files(project_root: Path, operator_name: str) -> list[CaseFile]:
    """查找算子的所有用例产物文件。

    兼容三种产物路径：
    1. batch_cases/{operator}_cases.json
    2. cases/{operator}_cases.json（generate_cases 节点老产物）
    3. cases/{operator}_{product_safe}_cases.json（case_subgraph 新产物）
    """
    found: list[CaseFile] = []

    # 路径 1：batch_cases
    batch_dir = project_root / "batch_cases"
    for path in batch_dir.glob(f"{operator_name}*_cases.json"):
        found.append(CaseFile(path=str(path.relative_to(project_root))))

    # 路径 2/3：cases（含按产品切分）
    cases_dir = project_root / "cases"
    if cases_dir.is_dir():
        for path in cases_dir.glob(f"{operator_name}*_cases.json"):
            # 尝试从文件名推断 product
            stem = path.stem  # 去掉 _cases.json 后缀
            product: str | None = None
            if stem.startswith(operator_name + "_"):
                tail = stem[len(operator_name) + 1 :]
                if tail:
                    product = tail
            found.append(
                CaseFile(
                    path=str(path.relative_to(project_root)),
                    product=product,
                )
            )

    return found


def _find_execution_result(project_root: Path, operator_name: str) -> tuple[Path | None, dict | None]:
    """查找 execution_results/{operator}/result.json 并解析。"""
    exec_dir = project_root / "execution_results" / operator_name
    result_path = exec_dir / "result.json"
    if not result_path.is_file():
        return None, None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("failed to parse %s: %s", result_path, e)
        return result_path, None
    return result_path, data


def _extract_report_records(result_data: dict) -> list[ReportRecord]:
    """从 result.json 中提取 report_records。

    兼容两种结构：
    1. {"task_report_data": {"report_records": [...]}}
    2. {"report_records": [...]}（顶层）
    """
    records_data: list[dict] = []
    trd = result_data.get("task_report_data")
    if isinstance(trd, dict) and isinstance(trd.get("report_records"), list):
        records_data = trd["report_records"]
    elif isinstance(result_data.get("report_records"), list):
        records_data = result_data["report_records"]

    records: list[ReportRecord] = []
    for r in records_data:
        if not isinstance(r, dict):
            continue
        records.append(
            ReportRecord(
                id=r.get("id", ""),
                run_result=str(r.get("run_result", "UNKNOWN")).upper(),
                failure_reason=r.get("failure_reason"),
            )
        )
    return records


def _classify_execution_status(records: list[ReportRecord]) -> str:
    """根据 report_records 判断执行状态。"""
    if not records:
        return "missing"
    success = sum(1 for r in records if r.run_result == "SUCCESS")
    failed = sum(1 for r in records if r.run_result == "FAILED")
    if failed == 0:
        return "success"
    if success == 0:
        return "failed"
    return "partial"


def _find_log_file(project_root: Path, operator_name: str) -> Path | None:
    """查找 logs/generate_case_{operator}.log。"""
    logs_dir = project_root / "logs"
    if not logs_dir.is_dir():
        return None
    for path in logs_dir.glob(f"generate_case_{operator_name}*.log"):
        return path
    return None


# --------------------------------------------------------------------------- #
# 数据库查询
# --------------------------------------------------------------------------- #


def _query_document_versions(
    db_path: Path,
    operator_name: str | None = None,
) -> list[dict[str, Any]]:
    """查询 document_versions 表，按 operator 聚合。

    返回结构：
    [
        {
            "operator_name": "aclnnAbs",
            "doc_id": 1,
            "json_constraints_raw": "...",
            "json_constraints": {...},  # 解析后的 dict
            "content": "...",  # 文档原始内容（用于校对）
        }
    ]
    """
    if not db_path.is_file():
        logger.warning("DB file not found: %s", db_path)
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if operator_name:
            rows = conn.execute(
                """
                SELECT
                    dv.id AS doc_id,
                    dv.json_constraints AS json_constraints_raw,
                    dv.content AS content,
                    o.name AS operator_name
                FROM document_versions dv
                JOIN operators o ON o.id = dv.operator_id
                WHERE o.name = ?
                ORDER BY dv.id DESC
                """,
                (operator_name,),
            ).fetchall()
        else:
            # 找出每个 operator 最新版本的 document_version
            rows = conn.execute(
                """
                SELECT
                    dv.id AS doc_id,
                    dv.json_constraints AS json_constraints_raw,
                    dv.content AS content,
                    o.name AS operator_name
                FROM document_versions dv
                JOIN operators o ON o.id = dv.operator_id
                WHERE dv.id IN (
                    SELECT MAX(id) FROM document_versions GROUP BY operator_id
                )
                ORDER BY o.name
                """,
            ).fetchall()
    finally:
        conn.close()

    results: list[dict[str, Any]] = []
    for row in rows:
        raw = row["json_constraints_raw"] or "{}"
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        results.append(
            {
                "operator_name": row["operator_name"],
                "doc_id": row["doc_id"],
                "json_constraints_raw": raw,
                "json_constraints": parsed,
                "content": row["content"] or "",
            }
        )
    return results


def _query_pipeline_runs(db_path: Path, operator_name: str) -> list[dict[str, Any]]:
    """查询 pipeline_runs 表，获取最近的 task 信息。"""
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT run_id, task_type, task_name, status, error,
                   result_json, created_at, completed_at
            FROM pipeline_runs
            WHERE operator_name = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (operator_name,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# 核心扫描逻辑
# --------------------------------------------------------------------------- #


def _has_real_constraints(jc: dict) -> bool:
    """判断 json_constraints 是否包含真实约束（非空 {}）。"""
    if not jc:
        return False
    # 至少要包含 inputs 或 outputs 或 constraints_in_parameters 之一
    for key in ("inputs", "outputs", "constraints_in_parameters"):
        v = jc.get(key)
        if isinstance(v, dict) and v:
            return True
        if isinstance(v, list) and v:
            return True
    return False


def _scan_single_operator(
    project_root: Path,
    db_path: Path,
    operator_name: str,
) -> OperatorStatus:
    """扫描单个算子。"""
    status = OperatorStatus(operator_name=operator_name)

    # 1. DB 查询约束
    docs = _query_document_versions(db_path, operator_name)
    if not docs:
        status.constraint_status = "missing"
        status.errors.append("document_versions 表无该算子记录")
    else:
        # 取最新版本
        latest = docs[0]
        status.doc_id = latest["doc_id"]
        jc = latest["json_constraints"]
        if _has_real_constraints(jc):
            status.constraint_status = "success"
            status.constraint_keys_count = len(jc.get("inputs", {}))
        else:
            status.constraint_status = "failed"
            status.errors.append("json_constraints 为空或无 inputs/outputs/constraints")

        # 2. pipeline_runs
        status.pipeline_runs = _query_pipeline_runs(db_path, operator_name)

    # 3. 查找用例产物
    status.case_files = _find_case_files(project_root, operator_name)
    if status.case_files:
        status.case_generation_status = "success"
    else:
        # 检查日志判断是否生成失败
        log_path = _find_log_file(project_root, operator_name)
        if log_path:
            status.log_path = str(log_path.relative_to(project_root))
            # 简单判定：日志中是否包含 traceback
            try:
                content = log_path.read_text(encoding="utf-8", errors="replace")
                if "Traceback" in content or "Error:" in content:
                    status.case_generation_status = "failed"
                else:
                    # 日志存在但无产物 → 可能未执行
                    status.case_generation_status = "missing"
            except OSError as e:
                status.errors.append(f"无法读取日志 {log_path}: {e}")
        # 如果 pipeline_runs 中有 failed 的 case_generate → 标记为 failed
        for run in status.pipeline_runs:
            if run.get("task_type") == "case_generate" and run.get("status") == "failed":
                status.case_generation_status = "failed"
                if run.get("error"):
                    status.errors.append(f"case_generate 失败：{run['error']}")

    # 4. 查找执行结果
    result_path, result_data = _find_execution_result(project_root, operator_name)
    if result_data is not None:
        status.result_json_path = str(result_path.relative_to(project_root))
        status.execution_dir = str(result_path.parent.relative_to(project_root))
        status.report_records = _extract_report_records(result_data)
        status.execution_status = _classify_execution_status(status.report_records)

    return status


def _compute_summary(statuses: list[OperatorStatus]) -> ScanSummary:
    s = ScanSummary()
    s.total = len(statuses)
    for st in statuses:
        if st.constraint_status == "success":
            s.constraint_success += 1
        if st.case_generation_status == "success":
            s.case_generation_success += 1
        elif st.case_generation_status == "failed":
            s.case_generation_failed += 1
        else:
            s.case_generation_missing += 1
        if st.execution_status == "success":
            s.execution_success += 1
        elif st.execution_status == "failed":
            s.execution_failed += 1
        elif st.execution_status == "partial":
            s.execution_partial += 1
        else:
            s.execution_missing += 1
    return s


def _status_to_dict(st: OperatorStatus) -> dict:
    d = asdict(st)
    # CaseFile 转 dict
    d["case_files"] = [asdict(c) for c in st.case_files]
    d["report_records"] = [asdict(r) for r in st.report_records]
    return d


def scan_all_operators(
    project_root: Path,
    db_path: Path,
    operator_name: str | None = None,
) -> dict:
    """主入口。

    Returns:
        {
            "project_root": str,
            "db_path": str,
            "operators": [OperatorStatus dict, ...],
            "summary": ScanSummary dict,
        }
    """
    statuses: list[OperatorStatus] = []
    if operator_name:
        statuses.append(_scan_single_operator(project_root, db_path, operator_name))
    else:
        # 查询所有 operator
        all_docs = _query_document_versions(db_path, operator_name=None)
        # 去重 operator
        seen = set()
        for d in all_docs:
            name = d["operator_name"]
            if name in seen:
                continue
            seen.add(name)
            st = _scan_single_operator(project_root, db_path, name)
            statuses.append(st)

    summary = _compute_summary(statuses)
    return {
        "project_root": str(project_root),
        "db_path": str(db_path),
        "operators": [_status_to_dict(st) for st in statuses],
        "summary": asdict(summary),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _resolve_project_root(path: str | None) -> Path:
    if path:
        return Path(path).resolve()
    # 默认：相对当前工作目录向上找到 .claude
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".claude").is_dir():
            return parent
    return cwd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描 operator-agent 算子状态")
    parser.add_argument(
        "--project-root",
        help="operator-agent 项目根目录（默认自动检测）",
    )
    parser.add_argument("--operator", help="单个算子名称（省略则扫描全部）")
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用 debug 日志",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    project_root = _resolve_project_root(args.project_root)
    db_path = project_root / "data" / "operator_agent.db"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = scan_all_operators(project_root, db_path, operator_name=args.operator)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("scan complete: %s", output_path)

    # 控制台摘要
    s = result["summary"]
    print(json.dumps({
        "scanned_operators": s["total"],
        "constraint_success": s["constraint_success"],
        "case_generation": {
            "success": s["case_generation_success"],
            "failed": s["case_generation_failed"],
            "missing": s["case_generation_missing"],
        },
        "execution": {
            "success": s["execution_success"],
            "failed": s["execution_failed"],
            "partial": s["execution_partial"],
            "missing": s["execution_missing"],
        },
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())