#!/usr/bin/env python3
"""根因分类脚本。

输入：scan_operators.py 输出的 JSON，针对每个失败算子判断属于哪一类问题：
- CONSTRAINT_WRONG: 约束提取错误
- CONSTRAINT_MISSING: 约束缺失
- GENERATOR_CODE_BUG: 用例生成代码 bug
- LLM_PROMPT_GAP: LLM 提示词遗漏
- EXECUTION_ENV_ERROR: 执行环境错误
- CONSTRAINT_GENERATOR_BOTH: 双端都可能有问题

Usage:
    python classify_failure.py --scan-result scan_results.json --operator aclnnAbs
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# 确保 Windows 控制台能输出中文
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

logger = logging.getLogger(__name__)


# 分类代码
CONSTRAINT_WRONG = "CONSTRAINT_WRONG"
CONSTRAINT_MISSING = "CONSTRAINT_MISSING"
GENERATOR_CODE_BUG = "GENERATOR_CODE_BUG"
LLM_PROMPT_GAP = "LLM_PROMPT_GAP"
EXECUTION_ENV_ERROR = "EXECUTION_ENV_ERROR"
CONSTRAINT_GENERATOR_BOTH = "CONSTRAINT_GENERATOR_BOTH"
UNKNOWN = "UNKNOWN"

# 分类显示名（中文）
CATEGORY_LABELS: dict[str, str] = {
    CONSTRAINT_WRONG: "约束提取错误",
    CONSTRAINT_MISSING: "约束缺失",
    GENERATOR_CODE_BUG: "生成代码 Bug",
    LLM_PROMPT_GAP: "LLM 提示词遗漏",
    EXECUTION_ENV_ERROR: "执行环境错误",
    CONSTRAINT_GENERATOR_BOTH: "约束+生成双端问题",
    UNKNOWN: "未知",
}

# 分类优先级（用于 CONSTRAINT_GENERATOR_BOTH 的判定）
CATEGORY_SEVERITY: dict[str, int] = {
    CONSTRAINT_WRONG: 3,
    CONSTRAINT_MISSING: 3,
    GENERATOR_CODE_BUG: 2,
    LLM_PROMPT_GAP: 3,
    EXECUTION_ENV_ERROR: 1,
    CONSTRAINT_GENERATOR_BOTH: 4,
    UNKNOWN: 0,
}


@dataclass
class FailureClassification:
    """单个算子的根因分类结果。"""

    operator_name: str
    category: str
    confidence: float  # 0.0 ~ 1.0
    evidence: list[str] = field(default_factory=list)
    secondary_categories: list[str] = field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------- #
# 信号检测
# --------------------------------------------------------------------------- #


_GENERATOR_CODE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Python 异常
    (re.compile(r"AttributeError.*'(NoneType|int|str|list|dict|tuple)'.*has no attribute"), "Python AttributeError（多为 None 兜底缺失）"),
    (re.compile(r"KeyError: ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"), "KeyError（多为字段缺失）"),
    (re.compile(r"TypeError:.*unsupported operand type"), "TypeError（类型运算不兼容）"),
    (re.compile(r"IndexError: list index out of range"), "IndexError（列表越界）"),
    (re.compile(r"ValueError:.*empty"), "ValueError 空集合"),
    (re.compile(r"ZeroDivisionError"), "ZeroDivisionError"),
    # generators 模块特定路径
    (re.compile(r"File \".*generators/(case_builder|shape_sampler|value_sampler|dtype_picker|shape_groups)\.py\""), "异常发生在 generators/ 模块"),
]

_CONSTRAINT_PATTERN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"json_constraints not found"), "约束未生成"),
    (re.compile(r"constraint_doc_id.*null|null.*constraint_doc_id"), "约束记录缺失"),
]

_EXEC_ENV_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ssh: connect to host.*port.*Connection refused", re.I), "SSH 连接失败"),
    (re.compile(r"No such file or directory:.*atk(_run)?(\.py|\.sh)?"), "ATK 文件不存在"),
    (re.compile(r"ASCEND_RT(_ERROR|_VISIBLE_DEVICES).*not set"), "环境变量未设置"),
    (re.compile(r"RuntimeError:.*device|driver", re.I), "设备/驱动错误"),
    (re.compile(r"CudaError|cuda runtime error", re.I), "CUDA 错误"),
    (re.compile(r"timeout.*expired", re.I), "执行超时"),
]


def _detect_generator_code_signals(text: str) -> list[str]:
    """检测生成代码 bug 的信号。"""
    hits: list[str] = []
    for pat, desc in _GENERATOR_CODE_PATTERNS:
        if pat.search(text):
            hits.append(desc)
    return hits


def _detect_constraint_signals(text: str) -> list[str]:
    """检测约束错误/缺失的信号。"""
    hits: list[str] = []
    for pat, desc in _CONSTRAINT_PATTERN:
        if pat.search(text):
            hits.append(desc)
    return hits


def _detect_exec_env_signals(text: str) -> list[str]:
    """检测执行环境错误的信号。"""
    hits: list[str] = []
    for pat, desc in _EXEC_ENV_PATTERNS:
        if pat.search(text):
            hits.append(desc)
    return hits


# --------------------------------------------------------------------------- #
# 约束对比
# --------------------------------------------------------------------------- #


def _check_param_completeness(jc: dict, src_doc: str) -> list[str]:
    """简单检查 json_constraints 是否遗漏了源文档中的参数。

    这是一个粗略的启发式检查，提取源文档参数表中的参数名（加粗的列），
    与 json_constraints inputs/outputs 的 key 对比。
    """
    if not src_doc or not jc:
        return []
    # 粗略：提取 src_doc 中加粗的参数名 pattern `**xx**`
    doc_params = set(re.findall(r"\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*", src_doc))
    json_inputs = set((jc.get("inputs") or {}).keys())
    json_outputs = set((jc.get("outputs") or {}).keys())
    json_params = json_inputs | json_outputs
    missing = doc_params - json_params
    # 过滤掉明显不是参数的（aclnn开头的函数名等）
    missing = {
        m for m in missing
        if not m.startswith("aclnn") and m not in {"ND", "N/A", "NULL", "TRUE", "FALSE"}
    }
    return sorted(missing)


def _check_shape_consistency(jc: dict) -> list[str]:
    """检查 json_constraints 中 shape 字段是否异常。"""
    issues: list[str] = []
    for category in ("inputs", "outputs"):
        params = jc.get(category) or {}
        for pname, pv in params.items():
            if not isinstance(pv, dict):
                continue
            # dimensions.value 可能是 list 或 [[min,max],...]
            dims = pv.get("dimensions", {})
            if not isinstance(dims, dict):
                continue
            value = dims.get("value")
            if value is None:
                continue
            # 检查格式
            if not isinstance(value, list):
                issues.append(f"{category}.{pname}.dimensions.value 不是列表: {value!r}")
                continue
            # 如果是 [[min,max],...] 格式
            if value and all(isinstance(x, list) and len(x) == 2 for x in value):
                for i, (lo, hi) in enumerate(value):
                    if not (isinstance(lo, int) and isinstance(hi, int)):
                        issues.append(f"{category}.{pname}.dimensions[{i}] 不是整数对: [{lo},{hi}]")
                    elif lo < 0:
                        issues.append(f"{category}.{pname}.dimensions[{i}] 出现负值: [{lo},{hi}]")
    return issues


# --------------------------------------------------------------------------- #
# 主分类逻辑
# --------------------------------------------------------------------------- #


def _read_failure_texts(operator_status: dict, project_root: Path) -> dict[str, str]:
    """汇总算子的所有失败文本：日志 + pipeline_runs error + report_records failure_reason。"""
    texts: dict[str, str] = {}

    log_path_rel = operator_status.get("log_path")
    if log_path_rel:
        log_path = project_root / log_path_rel
        try:
            texts["log"] = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("cannot read log %s: %s", log_path, e)

    for run in operator_status.get("pipeline_runs", []):
        if run.get("error"):
            texts.setdefault("pipeline_errors", "")
            texts["pipeline_errors"] += f"\n[{run.get('task_type', '?')}] {run['error']}"

    failure_reasons: list[str] = []
    for rec in operator_status.get("report_records", []):
        if rec.get("failure_reason"):
            failure_reasons.append(f"id={rec.get('id')}: {rec['failure_reason']}")
    if failure_reasons:
        texts["execution_failures"] = "\n".join(failure_reasons)

    return texts


def classify_operator(
    operator_status: dict,
    *,
    project_root: Path,
    json_constraints: dict | None = None,
    src_doc_path: Path | None = None,
) -> FailureClassification:
    """对单个算子进行根因分类。

    Args:
        operator_status: scan_operators.py 输出的 operator dict
        project_root: 项目根目录
        json_constraints: 已加载的约束 dict（可选，避免重复读 DB）
        src_doc_path: 算子源文档 .md 路径（可选）
    """
    name = operator_status["operator_name"]
    fc = FailureClassification(operator_name=name, category=UNKNOWN, confidence=0.0)

    texts = _read_failure_texts(operator_status, project_root)
    all_text = "\n".join(texts.values())

    # 信号收集
    gen_signals = _detect_generator_code_signals(all_text)
    con_signals = _detect_constraint_signals(all_text)
    env_signals = _detect_exec_env_signals(all_text)

    fc.evidence.extend(gen_signals)
    fc.evidence.extend(con_signals)
    fc.evidence.extend(env_signals)

    # 仅约束缺失分支
    if operator_status.get("constraint_status") != "success":
        fc.category = CONSTRAINT_MISSING if con_signals else CONSTRAINT_WRONG
        fc.confidence = 0.9
        fc.notes = "DB 中 json_constraints 为空或无 inputs/outputs/constraints_in_parameters"
        return fc

    # 用例生成失败分支
    if operator_status.get("case_generation_status") == "failed":
        if gen_signals:
            fc.category = GENERATOR_CODE_BUG
            fc.confidence = 0.85
            fc.notes = "用例生成日志出现 Python 异常，且堆栈指向 generators/ 模块"
            return fc
        if "约束未生成" in " ".join(con_signals) or "约束记录缺失" in " ".join(con_signals):
            fc.category = CONSTRAINT_WRONG
            fc.confidence = 0.9
            return fc
        # 默认视为生成代码问题（待人工进一步看日志）
        fc.category = GENERATOR_CODE_BUG
        fc.confidence = 0.6
        fc.notes = "用例生成标记为失败，未检测到具体异常类型，建议查看完整日志"
        return fc

    # 执行失败（case_generation 可能 success/missing/partial，不影响执行失败判定）
    if operator_status.get("execution_status") in {"failed", "partial"}:

        # 先看是否环境错误
        if env_signals:
            fc.category = EXECUTION_ENV_ERROR
            fc.confidence = 0.9
            fc.notes = "执行报错指向环境/驱动/ATK 安装问题"
            return fc

        # 看是否 generator code bug（典型：生成的 shape/dtype 越界）
        failure_text = texts.get("execution_failures", "") + "\n" + all_text
        if re.search(r"shape.*(invalid|out of range|negative|too large|dimension)", failure_text, re.I):
            # 进一步对照约束
            if json_constraints:
                shape_issues = _check_shape_consistency(json_constraints)
                if shape_issues:
                    fc.category = GENERATOR_CODE_BUG
                    fc.confidence = 0.7
                    fc.notes = f"约束中 shape 字段存在可疑：{shape_issues[:3]}"
                    fc.secondary_categories = [CONSTRAINT_WRONG]
                    return fc
            fc.category = GENERATOR_CODE_BUG
            fc.confidence = 0.75
            fc.notes = "执行失败信息指向 shape 不合法，但 json_constraints 中 shape 字段看起来合规，问题可能出在 case_builder 的 shape 采样逻辑"
            return fc

        if re.search(r"dtype.*(mismatch|incompatible|unsupported)", failure_text, re.I):
            if json_constraints:
                # 简单检查：是否有 input/output dtype 冲突
                inputs = json_constraints.get("inputs") or {}
                outputs = json_constraints.get("outputs") or {}
                input_dtypes = {
                    p.get("dtype", {}).get("value") if isinstance(p, dict) else None
                    for p in inputs.values()
                }
                output_dtypes = {
                    p.get("dtype", {}).get("value") if isinstance(p, dict) else None
                    for p in outputs.values()
                }
                # 如果 inputs 都用 FLOAT，但 outputs dtype 为空，可能是 LLM 漏了
                if input_dtypes and not any(output_dtypes):
                    fc.category = CONSTRAINT_MISSING
                    fc.confidence = 0.75
                    fc.notes = "outputs 中部分参数 dtype 为空，可能 LLM 漏提"
                    fc.secondary_categories = [LLM_PROMPT_GAP]
                    return fc
            fc.category = CONSTRAINT_GENERATOR_BOTH
            fc.confidence = 0.6
            fc.notes = "dtype 不匹配，可能是约束 dtype 列表不全，或生成器选 dtype 时出错"
            return fc

        # 默认：可能 prompt gap
        if re.search(r"(unknown parameter|unexpected argument|missing argument)", failure_text, re.I):
            # 参数名/数量对不上 → 约束问题（参数提取错误或缺失）
            if json_constraints and src_doc_path and src_doc_path.is_file():
                try:
                    src_doc = src_doc_path.read_text(encoding="utf-8")
                except OSError:
                    src_doc = ""
                missing = _check_param_completeness(json_constraints, src_doc)
                if missing:
                    fc.category = CONSTRAINT_MISSING
                    fc.confidence = 0.8
                    fc.notes = f"源文档提到但 json_constraints 缺失：{missing}"
                    fc.secondary_categories = [LLM_PROMPT_GAP]
                    return fc
            fc.category = CONSTRAINT_WRONG
            fc.confidence = 0.7
            fc.notes = "执行报错提及 unknown parameter，可能是约束中参数名/数量与算子实际签名不一致"
            return fc

        # 兜底：未识别的执行失败
        fc.category = UNKNOWN
        fc.confidence = 0.3
        fc.notes = "未识别出明确的失败模式，建议人工查看 execution failure_reason 和源码"
        return fc

    # 全部成功
    fc.category = "SUCCESS"
    fc.confidence = 1.0
    fc.notes = "约束+用例+执行三阶段全部通过"
    return fc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="算子根因分类")
    parser.add_argument("--scan-result", required=True, help="scan_operators.py 的输出 JSON")
    parser.add_argument("--operator", help="指定单个算子（省略则分类所有失败算子）")
    parser.add_argument(
        "--project-root",
        help="operator-agent 项目根目录",
    )
    parser.add_argument("--output", "-o", required=True, help="分类结果 JSON 输出路径")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scan_data = json.loads(Path(args.scan_result).read_text(encoding="utf-8"))
    project_root = Path(args.project_root or scan_data.get("project_root", ".")).resolve()

    # 加载 DB（仅在需要读 json_constraints 时）
    db_path = project_root / "data" / "operator_agent.db"
    conn_jc_lookup: dict[str, dict] = {}
    try:
        if db_path.is_file():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            for row in conn.execute("""
                SELECT o.name AS op, dv.json_constraints AS jc
                FROM document_versions dv
                JOIN operators o ON o.id = dv.operator_id
                WHERE dv.id IN (SELECT MAX(id) FROM document_versions GROUP BY operator_id)
            """):
                try:
                    conn_jc_lookup[row["op"]] = json.loads(row["jc"] or "{}")
                except json.JSONDecodeError:
                    conn_jc_lookup[row["op"]] = {}
            conn.close()
    except Exception as e:
        logger.warning("DB load failed: %s", e)

    classifications: list[dict] = []
    for op in scan_data["operators"]:
        if args.operator and op["operator_name"] != args.operator:
            continue
        # 只分类有失败/缺数据的算子
        is_failed = (
            op.get("constraint_status") != "success"
            or op.get("case_generation_status") == "failed"
            or op.get("execution_status") in {"failed", "partial"}
        )
        if not is_failed and not args.operator:
            continue

        jc = conn_jc_lookup.get(op["operator_name"])
        src_path = project_root / "operators" / f"{op['operator_name']}.md"
        if not src_path.is_file():
            src_path = None

        fc = classify_operator(
            op,
            project_root=project_root,
            json_constraints=jc,
            src_doc_path=src_path,
        )
        d = asdict(fc)
        d["category_label"] = CATEGORY_LABELS.get(fc.category, fc.category)
        classifications.append(d)

    # 排序：按置信度降序、按严重度降序
    classifications.sort(
        key=lambda x: (
            -CATEGORY_SEVERITY.get(x["category"], 0),
            -x["confidence"],
        )
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "classifications": classifications,
            "summary": _summarize(classifications),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("classification complete: %s", out_path)

    print(json.dumps({
        "classified": len(classifications),
        "summary": _summarize(classifications),
    }, ensure_ascii=False, indent=2))
    return 0


def _summarize(classifications: list[dict]) -> dict[str, int]:
    s: dict[str, int] = {}
    for c in classifications:
        cat = c["category"]
        s[cat] = s.get(cat, 0) + 1
    return s


if __name__ == "__main__":
    sys.exit(main())