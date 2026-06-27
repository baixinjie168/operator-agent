#!/usr/bin/env python3
"""约束深度分析：对 CONSTRAINT_WRONG / CONSTRAINT_MISSING 类别的算子，
逐字段对照源文档，找出具体错误字段和修复建议。

Usage:
    python analyze_constraint.py --scan-result scan_results.json \
        --classification classification.json --operator aclnnAbs --output constraint_analysis.json
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
from typing import Any

# 确保 Windows 控制台能输出中文
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

logger = logging.getLogger(__name__)


@dataclass
class ConstraintIssue:
    """单个约束错误。"""

    category: str  # param_missing / param_attr_wrong / relation_missing / relation_wrong / shape_wrong / dtype_wrong
    severity: str  # high / medium / low
    param_name: str
    field_name: str | None  # 出错的字段（如 dimensions.value、dtype.value）
    src_doc_evidence: str
    json_constraints_state: str
    likely_root_cause: str
    affected_prompt: str | None = None
    affected_code: str | None = None
    fix_suggestion: str = ""


@dataclass
class ConstraintAnalysis:
    """单个算子的约束分析结果。"""

    operator_name: str
    constraint_issues: list[ConstraintIssue] = field(default_factory=list)
    summary: str = ""
    src_doc_path: str | None = None


# --------------------------------------------------------------------------- #
# 源文档解析
# --------------------------------------------------------------------------- #


_PARAM_NAME_IN_DOC_RE = re.compile(r"\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*")
_PARAM_TABLE_HEADER_RE = re.compile(
    r"参数名\|输入/输出\|描述\|使用说明\|数据类型\|数据格式\|维度\(shape\)\|非连续Tensor"
)


def _parse_src_doc(src_doc: str) -> dict[str, Any]:
    """从 Markdown 源文档中粗略提取参数信息。

    Returns:
        {
            "param_names": [...],   # 参数名（加粗）
            "table_rows": [...],    # 参数表行
            "explanation": str,
        }
    """
    result: dict[str, Any] = {
        "param_names": sorted(set(_PARAM_NAME_IN_DOC_RE.findall(src_doc))),
        "table_rows": [],
        "explanation": "",
    }

    # 尝试解析参数表（简化版：按行分割，找含 `|` 的行）
    lines = src_doc.splitlines()
    in_table = False
    for line in lines:
        if _PARAM_TABLE_HEADER_RE.search(line):
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("---") or line.strip().startswith("|-"):
                continue
            if "|" not in line:
                in_table = False
                continue
            cells = [c.strip() for c in line.split("|")]
            if cells and cells[0]:
                result["table_rows"].append(cells)

    # 提取功能说明（`## 功能说明` 章节）
    m = re.search(r"##\s*功能说明\s*\n+(.*?)(?=\n##\s|\Z)", src_doc, re.S)
    if m:
        result["explanation"] = m.group(1).strip()[:500]

    return result


# --------------------------------------------------------------------------- #
# 约束字段读取
# --------------------------------------------------------------------------- #


def _load_json_constraints(db_path: Path, operator_name: str) -> dict | None:
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT dv.json_constraints
            FROM document_versions dv
            JOIN operators o ON o.id = dv.operator_id
            WHERE o.name = ?
            ORDER BY dv.id DESC LIMIT 1
            """,
            (operator_name,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    raw = row[0] or "{}"
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# 对比分析
# --------------------------------------------------------------------------- #


def _analyze_param_completeness(
    parsed_doc: dict, jc: dict, op_name: str,
) -> list[ConstraintIssue]:
    """对比源文档参数与 json_constraints 参数。"""
    issues: list[ConstraintIssue] = []
    doc_params = set(parsed_doc["param_names"])
    json_inputs = set((jc.get("inputs") or {}).keys())
    json_outputs = set((jc.get("outputs") or {}).keys())
    json_params = json_inputs | json_outputs

    # 过滤掉明显不是参数的标识符
    filtered_doc_params = {
        p for p in doc_params
        if not p.startswith("aclnn") and p not in {"ND", "N/A", "NULL", "TRUE", "FALSE"}
    }

    missing = filtered_doc_params - json_params
    for m in sorted(missing):
        issues.append(ConstraintIssue(
            category="param_missing",
            severity="high",
            param_name=m,
            field_name=None,
            src_doc_evidence=f"源文档参数表或加粗文本中提到 `{m}`",
            json_constraints_state=f"json_constraints 的 inputs/outputs 中不存在 `{m}`",
            likely_root_cause=(
                "可能是 implicit_param_extract 或 llm_description_extract 子图漏掉了"
                "不在函数签名中的参数，或 assemble_result 未合并隐式参数"
            ),
            affected_prompt="packages/agent/src/agent/prompts/system.py（隐式参数相关 prompt）",
            affected_code="packages/agent/src/agent/nodes/implicit_param_extract.py 或 llm_description_extract 子图",
            fix_suggestion=(
                f"在 implicit_param_extract 的 prompt 中增加示例，明确要求提取形如 `{m}` "
                "的非函数签名参数；同时检查 assemble_result 是否将其归并到 inputs"
            ),
        ))

    # 额外参数（json 有但源文档无）
    extra = json_params - filtered_doc_params
    for e in sorted(extra):
        issues.append(ConstraintIssue(
            category="param_extra",
            severity="medium",
            param_name=e,
            field_name=None,
            src_doc_evidence="源文档未显式提及",
            json_constraints_state=f"json_constraints 中存在 `{e}`（属于衍生参数或幻觉）",
            likely_root_cause="若 is_operator_param=false 则为幻觉；若 true 则为衍生参数，正常",
            fix_suggestion="检查该参数的 is_operator_param 字段；若 false 则需要在 prompt 中约束 LLM 不要凭空添加参数",
        ))

    return issues


def _analyze_param_attrs(parsed_doc: dict, jc: dict) -> list[ConstraintIssue]:
    """对比参数属性的常见错误。"""
    issues: list[ConstraintIssue] = []

    # 检查 dimensions 异常
    for category in ("inputs", "outputs"):
        params = jc.get(category) or {}
        for pname, pv in params.items():
            if not isinstance(pv, dict):
                continue
            dims = pv.get("dimensions", {})
            if isinstance(dims, dict):
                value = dims.get("value")
                if isinstance(value, list) and value:
                    # 检查是否有 None
                    if any(v is None for v in value):
                        issues.append(ConstraintIssue(
                            category="shape_wrong",
                            severity="high",
                            param_name=pname,
                            field_name="dimensions.value",
                            src_doc_evidence="—",
                            json_constraints_state=f"{pname}.dimensions.value 包含 None: {value!r}",
                            likely_root_cause="LLM 输出了不规范的 dimensions，应使用 [] 或 [[min,max]]",
                            affected_prompt="packages/agent/src/agent/prompts/system.py（SHAPE_EXTRACT_PROMPT）",
                            affected_code="packages/agent/src/agent/nodes/shape_extract.py",
                            fix_suggestion="在 SHAPE_EXTRACT_PROMPT 中强调：dimensions.value 只能是 [] (标量)、[min,max] (维度范围) 或 [[min,max], ...] (逐维范围)；不允许 null",
                        ))

            # 检查 dtype
            dtype = pv.get("dtype", {})
            if isinstance(dtype, dict):
                dv = dtype.get("value")
                if isinstance(dv, list) and not dv:
                    issues.append(ConstraintIssue(
                        category="dtype_wrong",
                        severity="medium",
                        param_name=pname,
                        field_name="dtype.value",
                        src_doc_evidence="源文档数据类型列",
                        json_constraints_state=f"{pname}.dtype.value 为空列表 []",
                        likely_root_cause="LLM 未能从源文档识别 dtype 列表",
                        fix_suggestion="在 DTYPE_EXTRACT_PROMPT 中增加示例：当源文档写'支持 FLOAT16'时，输出 ['FLOAT16'] 而非 []",
                    ))

            # 检查 format
            fmt = pv.get("format", {})
            if isinstance(fmt, dict):
                fv = fmt.get("value")
                if isinstance(fv, str) and fv == "":
                    issues.append(ConstraintIssue(
                        category="format_wrong",
                        severity="low",
                        param_name=pname,
                        field_name="format.value",
                        src_doc_evidence="—",
                        json_constraints_state=f"{pname}.format.value 为空字符串",
                        likely_root_cause="Tensor 参数的 format 应为 ['ND']，非 Tensor 应为 N/A",
                        fix_suggestion="在 DFORMAT_EXTRACT_PROMPT 中明确：Tensor 参数默认 ['ND']，标量参数为 'N/A'",
                    ))

    return issues


def _analyze_constraints_in_params(parsed_doc: dict, jc: dict) -> list[ConstraintIssue]:
    """检查 constraints_in_parameters 是否遗漏常见模式。"""
    issues: list[ConstraintIssue] = []
    cip = jc.get("constraints_in_parameters") or {}
    if not cip:
        # constraints_in_parameters 完全没有 → 严重
        if jc.get("inputs") or jc.get("outputs"):
            issues.append(ConstraintIssue(
                category="relation_missing",
                severity="high",
                param_name="—",
                field_name="constraints_in_parameters",
                src_doc_evidence="源文档参数使用说明列",
                json_constraints_state="constraints_in_parameters 为空 {}",
                likely_root_cause=(
                    "build_param_relations 节点或 build_param_constraint 节点未产出，"
                    "或参数关系提取 Agent 子图失败"
                ),
                affected_code="packages/agent/src/agent/nodes/build_param_relations.py, build_param_constraint.py",
                fix_suggestion="查看 pipeline_events 中 param_relation 节点的 error 字段；常见原因：LLM 输出 expr 语法错误被 AST 校验拒绝",
            ))
        return issues

    # 检查 src_text 是否与源文档对应
    for product, constraints in cip.items():
        if not isinstance(constraints, list):
            continue
        for c in constraints:
            if not isinstance(c, dict):
                continue
            expr = c.get("expr", "")
            relation_params = c.get("relation_params", [])
            src_text = c.get("src_text", "")
            expr_type = c.get("expr_type", "")

            # 检查 expr 是否包含 relation_params 中未列出的参数
            if isinstance(expr, str):
                # 简单提取参数名：匹配 x1, alpha, hidden_size 这种
                used_names = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr))
                keywords = {"and", "or", "not", "in", "is", "None", "True", "False",
                            "all", "any", "len", "range", "for", "if", "else",
                            "dtype", "shape", "format", "range_value", "value"}
                used_names -= keywords
                rel_set = set(relation_params)
                missing_in_rel = used_names - rel_set
                if missing_in_rel:
                    issues.append(ConstraintIssue(
                        category="relation_wrong",
                        severity="medium",
                        param_name=",".join(sorted(missing_in_rel)),
                        field_name="relation_params",
                        src_doc_evidence=src_text or "—",
                        json_constraints_state=f"expr `{expr}` 使用了 {missing_in_rel} 但 relation_params 未列出",
                        likely_root_cause="build_param_relations 节点在构建 relation_params 时遗漏了 expr 中实际引用的变量",
                        affected_code="packages/agent/src/agent/nodes/build_param_relations.py",
                        fix_suggestion="在 build_param_relations 中增加 AST 校验：从 expr 中提取所有标识符，与 relation_params 求差集，缺失则补全",
                    ))

            # 检查 expr 是否为空字符串
            if isinstance(expr, str) and not expr.strip() and expr_type != "presence_dependency":
                issues.append(ConstraintIssue(
                    category="relation_wrong",
                    severity="low",
                    param_name=",".join(relation_params),
                    field_name="expr",
                    src_doc_evidence=src_text or "—",
                    json_constraints_state="expr 为空字符串",
                    likely_root_cause="LLM 没有输出可执行的表达式（可能是描述性约束）",
                    fix_suggestion="在 build_param_relations 的 prompt 中要求：所有约束必须给出 Python 风格的可执行 expr，描述性约束应被识别并跳过",
                ))

    return issues


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def analyze_constraint(
    operator_name: str,
    project_root: Path,
    db_path: Path,
) -> ConstraintAnalysis:
    """对单个算子做约束分析。"""
    analysis = ConstraintAnalysis(operator_name=operator_name)

    # 1. 读取源文档
    src_path = project_root / "operators" / f"{operator_name}.md"
    if not src_path.is_file():
        # 尝试在 operators/ 子目录中
        candidates = list((project_root / "operators").rglob(f"{operator_name}.md"))
        if candidates:
            src_path = candidates[0]
    if not src_path.is_file():
        analysis.summary = f"未找到算子源文档 {src_path}"
        return analysis
    analysis.src_doc_path = str(src_path.relative_to(project_root))
    src_doc = src_path.read_text(encoding="utf-8", errors="replace")
    parsed_doc = _parse_src_doc(src_doc)

    # 2. 读取约束
    jc = _load_json_constraints(db_path, operator_name)
    if not jc:
        analysis.summary = "json_constraints 为空或不存在"
        analysis.constraint_issues.append(ConstraintIssue(
            category="constraint_missing",
            severity="high",
            param_name="—",
            src_doc_evidence="—",
            json_constraints_state="document_versions.json_constraints 为 {} 或 NULL",
            likely_root_cause="约束提取 Pipeline 未完成，或 assemble_result 节点失败",
            affected_code="packages/agent/src/agent/nodes/assemble_result.py",
            fix_suggestion="查看 pipeline_runs 中 constraint_extract 任务的 error；常见原因：上游节点（如 param_relation_extract）失败导致 assemble_result 无数据可组装",
        ))
        return analysis

    # 3. 对比
    issues: list[ConstraintIssue] = []
    issues.extend(_analyze_param_completeness(parsed_doc, jc, operator_name))
    issues.extend(_analyze_param_attrs(parsed_doc, jc))
    issues.extend(_analyze_constraints_in_params(parsed_doc, jc))

    analysis.constraint_issues = issues
    if not issues:
        analysis.summary = "未发现明显约束错误（可能是 LLM 遗漏语义级约束，需人工 review）"
    else:
        sev_count: dict[str, int] = {}
        for i in issues:
            sev_count[i.severity] = sev_count.get(i.severity, 0) + 1
        analysis.summary = (
            f"发现 {len(issues)} 个约束问题 "
            f"(high: {sev_count.get('high', 0)}, "
            f"medium: {sev_count.get('medium', 0)}, "
            f"low: {sev_count.get('low', 0)})"
        )
    return analysis


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="约束深度分析")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--operator", action="append", required=True, help="可多次指定")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    project_root = Path(args.project_root).resolve()
    db_path = project_root / "data" / "operator_agent.db"

    all_analyses = []
    for op in args.operator:
        analysis = analyze_constraint(op, project_root, db_path)
        d = asdict(analysis)
        d["constraint_issues"] = [asdict(i) for i in analysis.constraint_issues]
        all_analyses.append(d)
        logger.info("%s: %s", op, analysis.summary)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"analyses": all_analyses}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sys.stdout.buffer.write(json.dumps({
        "analyzed": len(all_analyses),
        "operators": [
            {"operator": a["operator_name"], "issue_count": len(a["constraint_issues"]), "summary": a["summary"]}
            for a in all_analyses
        ],
    }, ensure_ascii=False, indent=2).encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())