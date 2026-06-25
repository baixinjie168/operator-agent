#!/usr/bin/env python3
"""统一编排脚本：一键跑完算子迭代分析全流程。

执行顺序：
1. scan_operators         → 扫描所有算子状态
2. classify_failure       → 根因分类
3. analyze_constraint     → 约束深度分析（仅失败算子）
4. analyze_generator      → 生成代码定位（仅失败算子）
5. analyze_prompt         → 提示词优化（仅失败算子）
6. generate_report        → 生成最终报告

Usage:
    python run_analysis.py \
        --project-root /path/to/operator-agent \
        [--operator aclnnAbs] \
        --output-dir reports/ \
        --format both
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 确保 Windows 控制台能输出中文
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


def _run_script(args: list[str], cwd: Path | None = None) -> int:
    """运行同目录下的脚本。"""
    cmd = [sys.executable] + args
    logger.debug("running: %s", " ".join(cmd))
    # 设置 UTF-8 环境避免子进程编码问题
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if result.stdout:
        logger.debug("stdout:\n%s", result.stdout)
    if result.stderr:
        logger.debug("stderr:\n%s", result.stderr)
    return result.returncode


def _step_scan(project_root: Path, operator: str | None, output_dir: Path) -> Path:
    out = output_dir / "scan_result.json"
    args = [
        str(SCRIPT_DIR / "scan_operators.py"),
        "--project-root", str(project_root),
        "--output", str(out),
    ]
    if operator:
        args += ["--operator", operator]
    rc = _run_script(args)
    if rc != 0:
        raise RuntimeError("scan_operators.py failed")
    return out


def _step_classify(scan_path: Path, output_dir: Path, project_root: Path, operator: str | None) -> Path:
    out = output_dir / "classification.json"
    args = [
        str(SCRIPT_DIR / "classify_failure.py"),
        "--scan-result", str(scan_path),
        "--project-root", str(project_root),
        "--output", str(out),
    ]
    if operator:
        args += ["--operator", operator]
    rc = _run_script(args)
    if rc != 0:
        raise RuntimeError("classify_failure.py failed")
    return out


def _step_constraint(
    project_root: Path,
    failed_operators: list[str],
    output_dir: Path,
) -> Path | None:
    if not failed_operators:
        logger.info("no failed operators; skip constraint analysis")
        return None
    out = output_dir / "constraint_analysis.json"
    args = [
        str(SCRIPT_DIR / "analyze_constraint.py"),
        "--project-root", str(project_root),
        "--output", str(out),
    ]
    for op in failed_operators:
        args += ["--operator", op]
    rc = _run_script(args)
    if rc != 0:
        raise RuntimeError("analyze_constraint.py failed")
    return out


def _step_generator(
    scan_path: Path,
    project_root: Path,
    failed_operators: list[str],
    output_dir: Path,
) -> Path | None:
    if not failed_operators:
        return None
    out = output_dir / "generator_analysis.json"
    args = [
        str(SCRIPT_DIR / "analyze_generator.py"),
        "--scan-result", str(scan_path),
        "--project-root", str(project_root),
        "--output", str(out),
    ]
    for op in failed_operators:
        args += ["--operator", op]
    rc = _run_script(args)
    if rc != 0:
        raise RuntimeError("analyze_generator.py failed")
    return out


def _step_prompt(
    failed_operators: list[str],
    constraint_path: Path | None,
    output_dir: Path,
) -> Path | None:
    if not failed_operators:
        return None
    out = output_dir / "prompt_analysis.json"
    args = [
        str(SCRIPT_DIR / "analyze_prompt.py"),
        "--output", str(out),
    ]
    if constraint_path:
        args += ["--constraint-analysis", str(constraint_path)]
    for op in failed_operators:
        args += ["--operator", op]
    rc = _run_script(args)
    if rc != 0:
        raise RuntimeError("analyze_prompt.py failed")
    return out


def _step_report(
    scan_path: Path,
    classification_path: Path,
    constraint_path: Path | None,
    generator_path: Path | None,
    prompt_path: Path | None,
    output_dir: Path,
    mode: str,
    fmt: str,
    operator: str | None,
) -> Path:
    if operator:
        report_path = output_dir / f"{operator}_analysis"
    else:
        report_path = output_dir / "all_operators_analysis"
    args = [
        str(SCRIPT_DIR / "generate_report.py"),
        "--scan-result", str(scan_path),
        "--classification", str(classification_path),
        "--mode", mode,
        "--format", fmt,
        "--output", str(report_path),
    ]
    if constraint_path:
        args += ["--constraint-analysis", str(constraint_path)]
    if generator_path:
        args += ["--generator-analysis", str(generator_path)]
    if prompt_path:
        args += ["--prompt-analysis", str(prompt_path)]
    rc = _run_script(args)
    if rc != 0:
        raise RuntimeError("generate_report.py failed")
    if fmt in ("md", "both"):
        return report_path.with_suffix(".md")
    return report_path.with_suffix(".html")


def run_full_analysis(
    project_root: Path,
    operator: str | None,
    output_dir: Path,
    fmt: str = "md",
) -> dict:
    """执行完整的算子迭代分析流程。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "单算子" if operator else "全量"

    # Step 1
    logger.info("step 1/6: scan_operators")
    scan_path = _step_scan(project_root, operator, output_dir)
    scan_data = json.loads(scan_path.read_text(encoding="utf-8"))

    # Step 2
    logger.info("step 2/6: classify_failure")
    classification_path = _step_classify(scan_path, output_dir, project_root, operator)
    classification_data = json.loads(classification_path.read_text(encoding="utf-8"))

    # 收集失败算子
    failed_operators = [
        c["operator_name"] for c in classification_data.get("classifications", [])
        if c["category"] not in ("SUCCESS",)
    ]

    # Step 3
    logger.info("step 3/6: analyze_constraint (operators=%d)", len(failed_operators))
    constraint_path = _step_constraint(project_root, failed_operators, output_dir)

    # Step 4
    logger.info("step 4/6: analyze_generator")
    generator_path = _step_generator(scan_path, project_root, failed_operators, output_dir)

    # Step 5
    logger.info("step 5/6: analyze_prompt")
    prompt_path = _step_prompt(failed_operators, constraint_path, output_dir)

    # Step 6
    logger.info("step 6/6: generate_report")
    report_path = _step_report(
        scan_path,
        classification_path,
        constraint_path,
        generator_path,
        prompt_path,
        output_dir,
        mode,
        fmt,
        operator,
    )

    return {
        "scan_result": str(scan_path),
        "classification": str(classification_path),
        "constraint_analysis": str(constraint_path) if constraint_path else None,
        "generator_analysis": str(generator_path) if generator_path else None,
        "prompt_analysis": str(prompt_path) if prompt_path else None,
        "report": str(report_path),
        "failed_operators": failed_operators,
        "summary": scan_data["summary"],
        "classification_summary": classification_data.get("summary", {}),
        "generated_at": datetime.now().isoformat(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="算子迭代分析一键运行")
    parser.add_argument("--project-root", required=True, help="operator-agent 项目根")
    parser.add_argument("--operator", help="单个算子名称（省略则全量）")
    parser.add_argument("--output-dir", default="./reports/iteration_analysis",
                        help="中间产物和报告输出目录")
    parser.add_argument("--format", choices=["md", "html", "both"], default="md")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not (project_root / "data" / "operator_agent.db").is_file():
        logger.warning("DB not found at %s, will still attempt to run with empty data", project_root / "data" / "operator_agent.db")

    try:
        result = run_full_analysis(
            project_root=project_root,
            operator=args.operator,
            output_dir=output_dir,
            fmt=args.format,
        )
    except Exception as e:
        logger.exception("analysis failed: %s", e)
        return 1

    print("\n=== 算子迭代分析完成 ===")
    print(json.dumps({
        "report_path": result["report"],
        "failed_operators": result["failed_operators"],
        "summary": result["summary"],
        "classification_summary": result["classification_summary"],
        "intermediate": {
            "scan": result["scan_result"],
            "classification": result["classification"],
            "constraint": result["constraint_analysis"],
            "generator": result["generator_analysis"],
            "prompt": result["prompt_analysis"],
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())