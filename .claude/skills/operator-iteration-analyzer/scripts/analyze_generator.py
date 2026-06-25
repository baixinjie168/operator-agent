#!/usr/bin/env python3
"""生成代码错误定位：对 GENERATOR_CODE_BUG 类别的算子，
从失败日志中提取堆栈，定位到 generators/ 模块的具体文件、函数、行号，
给出修复建议。

Usage:
    python analyze_generator.py --scan-result scan_results.json \
        --classification classification.json --operator aclnnAbs --output generator_analysis.json
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


# Python traceback 行 pattern
_TRACEBACK_FILE_RE = re.compile(
    r'File "(.+?\.py)", line (\d+), in (\S+)\s*\n\s*(.+?)(?=\n\s*\n|\Z)',
    re.S,
)


@dataclass
class GeneratorIssue:
    """单个生成代码错误。"""

    file: str
    function: str
    line: int
    code_snippet: str
    issue: str
    exception: str | None = None
    fix_suggestion: str = ""


@dataclass
class GeneratorAnalysis:
    """单个算子的生成代码分析结果。"""

    operator_name: str
    generator_issues: list[GeneratorIssue] = field(default_factory=list)
    summary: str = ""


def _extract_traceback(log_text: str) -> list[dict[str, Any]]:
    """从日志文本中提取 traceback 帧。"""
    frames = []
    for m in _TRACEBACK_FILE_RE.finditer(log_text):
        file_path = m.group(1)
        line_no = int(m.group(2))
        func = m.group(3)
        snippet = m.group(4).strip()
        frames.append({
            "file": file_path,
            "line": line_no,
            "function": func,
            "snippet": snippet,
        })
    return frames


def _read_code_window(file_path: Path, line_no: int, window: int = 8) -> str:
    """读取文件中 line_no 附近的代码片段。"""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(0, line_no - window)
        end = min(len(lines), line_no + window)
        snippet_lines = lines[start:end]
        annotated = []
        for i, line in enumerate(snippet_lines, start=start + 1):
            marker = "→ " if i == line_no else "  "
            annotated.append(f"{marker}{i:4d}: {line}")
        return "\n".join(annotated)
    except OSError as e:
        return f"<无法读取文件: {e}>"


def _guess_fix(file_path: Path, line_no: int, snippet: str, exception: str | None) -> str:
    """根据代码片段和异常推测修复方向。"""
    s = snippet.lower()
    exc = (exception or "").lower()

    if "nonetype" in exc and ".lower()" in exc:
        return (
            "类型为 None 时调用 .lower()：检查变量来源，添加 None 兜底：\n"
            "  value = value or 'default_value'"
        )
    if "nonetype" in exc and "has no attribute" in exc:
        attr = re.search(r"'[^']+'\s+object has no attribute '([^']+)'", exception or "")
        attr_name = attr.group(1) if attr else "<attr>"
        return f"对 None 调用 .{attr_name}：在该变量来源处加 None 检查或默认值"
    if "keyerror" in exc:
        return "访问不存在的 key：在字典访问前用 .get(key, default)，或先用 `if key in d` 判断"
    if "indexerror" in exc:
        return "列表越界：访问前先检查 len(list) > index，或使用更安全的取值方式"
    if "typeerror" in exc and "unsupported operand" in exc:
        return "运算类型不兼容：在运算前显式转换类型（如 int()）或加入类型守卫"
    if "valueerror" in exc and "empty" in exc:
        return "空集合运算：在解包或访问前判空 `if not items: return default`"
    if "zerodivisionerror" in exc:
        return "除零错误：分母加 if x == 0: return default 守卫"

    # 通用建议
    return (
        "1. 在该行前添加类型/None 守卫\n"
        "2. 检查上游采样函数是否在边界条件下返回 None/空\n"
        "3. 单元测试覆盖该边界"
    )


def _analyze_log_for_generator_bugs(
    log_text: str,
    project_root: Path,
) -> list[GeneratorIssue]:
    """从日志文本中提取 generators/ 相关的 traceback 帧。"""
    frames = _extract_traceback(log_text)
    issues: list[GeneratorIssue] = []
    seen: set[tuple[str, int]] = set()

    # 找出异常类型
    exception_match = re.search(
        r"^(\w+Error):\s*(.+?)$",
        log_text,
        re.M,
    )
    exception_str = (
        f"{exception_match.group(1)}: {exception_match.group(2)}"
        if exception_match else None
    )

    for frame in frames:
        file_path_str = frame["file"]
        line_no = frame["line"]
        func = frame["function"]
        # 标准化为相对路径
        try:
            file_path = Path(file_path_str)
            rel = file_path.relative_to(project_root) if file_path.is_absolute() else file_path
        except ValueError:
            rel = Path(file_path_str)
        rel_str = str(rel).replace("\\", "/")

        # 只关心 generators/ 模块
        if "generators/" not in rel_str and "agent/generators" not in rel_str:
            continue
        key = (rel_str, line_no)
        if key in seen:
            continue
        seen.add(key)

        code_window = _read_code_window(file_path, line_no)
        suggestion = _guess_fix(file_path, line_no, frame["snippet"], exception_str)

        issues.append(GeneratorIssue(
            file=rel_str,
            function=func,
            line=line_no,
            code_snippet=code_window,
            issue=(
                f"{exception_str or '未知异常'}\n"
                f"  定位：{rel_str}:{line_no} in {func}()\n"
                f"  帧片段：{frame['snippet']}"
            ),
            exception=exception_str,
            fix_suggestion=suggestion,
        ))

    return issues


def analyze_generator_for_operator(
    operator_status: dict,
    project_root: Path,
) -> GeneratorAnalysis:
    """对单个算子做生成代码错误定位。"""
    name = operator_status["operator_name"]
    analysis = GeneratorAnalysis(operator_name=name)

    log_path_rel = operator_status.get("log_path")
    log_text = ""
    if log_path_rel:
        log_path = project_root / log_path_rel
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            analysis.summary = f"无法读取日志 {log_path}: {e}"
            return analysis

    # 也尝试从 pipeline_runs 收集 error
    for run in operator_status.get("pipeline_runs", []):
        if run.get("task_type") == "case_generate" and run.get("error"):
            log_text += "\n\n[pipeline_runs.error]\n" + run["error"]

    if not log_text:
        analysis.summary = "无日志可分析"
        return analysis

    issues = _analyze_log_for_generator_bugs(log_text, project_root)

    if not issues:
        # 即使没有 traceback，也可以给一些通用建议
        analysis.summary = "日志中未发现 generators/ 模块的 traceback；请人工检查完整日志"
        # 提取日志末尾的关键错误
        tail_lines = log_text.strip().splitlines()[-30:]
        if tail_lines:
            analysis.summary += f"\n\n日志末尾：\n```\n" + "\n".join(tail_lines) + "\n```"
        return analysis

    analysis.generator_issues = issues
    analysis.summary = f"定位到 {len(issues)} 个 generators/ 模块异常位置"
    return analysis


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成代码错误定位")
    parser.add_argument("--scan-result", required=True, help="scan_operators.py 输出")
    parser.add_argument("--project-root", help="项目根（覆盖 scan_result 中的）")
    parser.add_argument("--operator", action="append", help="可多次指定；省略则分析所有 GENERATOR_CODE_BUG 算子")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scan_data = json.loads(Path(args.scan_result).read_text(encoding="utf-8"))
    project_root = Path(args.project_root or scan_data.get("project_root", ".")).resolve()

    targets: list[dict] = []
    if args.operator:
        name_set = set(args.operator)
        targets = [op for op in scan_data["operators"] if op["operator_name"] in name_set]
    else:
        # 默认：所有 case_generation_status == failed 或 execution_status in {failed, partial} 的算子
        targets = [
            op for op in scan_data["operators"]
            if op.get("case_generation_status") == "failed"
            or op.get("execution_status") in {"failed", "partial"}
        ]

    analyses = []
    for op in targets:
        analysis = analyze_generator_for_operator(op, project_root)
        d = asdict(analysis)
        d["generator_issues"] = [asdict(i) for i in analysis.generator_issues]
        analyses.append(d)
        logger.info("%s: %s", analysis.operator_name, analysis.summary)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"analyses": analyses}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "analyzed": len(analyses),
        "operators": [
            {"operator": a["operator_name"], "issue_count": len(a["generator_issues"]), "summary": a["summary"]}
            for a in analyses
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())