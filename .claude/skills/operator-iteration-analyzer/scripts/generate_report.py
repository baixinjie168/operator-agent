#!/usr/bin/env python3
"""汇总所有分析结果生成最终报告。

输入：
- scan_result.json (scan_operators.py)
- classification.json (classify_failure.py)
- constraint_analysis.json (analyze_constraint.py)
- generator_analysis.json (analyze_generator.py)
- prompt_analysis.json (analyze_prompt.py)

输出：Markdown 报告（默认），可选 HTML（明亮风格）

Usage:
    python generate_report.py \
        --scan-result scan.json --classification classification.json \
        --constraint-analysis constraint.json \
        --generator-analysis generator.json \
        --prompt-analysis prompt.json \
        --output reports/all_operators_analysis.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
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


# --------------------------------------------------------------------------- #
# Markdown 生成
# --------------------------------------------------------------------------- #


def _md_escape(text: str | None) -> str:
    """转义 markdown 特殊字符（保留 code fence 等语法）。"""
    if not text:
        return "—"
    return text.replace("|", "\\|")


def _gen_summary_section(scan_data: dict) -> str:
    s = scan_data["summary"]
    lines = [
        "## 一、整体概览",
        "",
        f"- 扫描算子总数：**{s['total']}**",
        "",
        "| 阶段 | 成功 | 失败 | 缺失/部分 | 成功率 |",
        "|------|------|------|----------|--------|",
        f"| 约束提取 | {s['constraint_success']} | "
        f"{s['total'] - s['constraint_success']} | — | "
        f"{s['constraint_success'] / s['total'] * 100 if s['total'] else 0:.1f}% |",
        f"| 用例生成 | {s['case_generation_success']} | {s['case_generation_failed']} | "
        f"{s['case_generation_missing']} | "
        f"{s['case_generation_success'] / s['total'] * 100 if s['total'] else 0:.1f}% |",
        f"| 用例执行 | {s['execution_success'] + s['execution_partial']} | {s['execution_failed']} | "
        f"{s['execution_missing']} | "
        f"{s['execution_success'] / s['total'] * 100 if s['total'] else 0:.1f}% |",
        "",
    ]
    return "\n".join(lines)


def _gen_failure_section(
    classifications: list[dict],
    scan_data: dict,
) -> str:
    """失败算子明细。"""
    if not classifications:
        return "## 二、失败算子明细\n\n**所有算子三阶段均通过！**\n"

    lines = ["## 二、失败算子明细", ""]

    # 按 operator_name 索引 scan_data
    op_lookup = {op["operator_name"]: op for op in scan_data["operators"]}

    # 按分类分组
    by_category: dict[str, list[dict]] = {}
    for c in classifications:
        by_category.setdefault(c["category"], []).append(c)

    cat_order = [
        "CONSTRAINT_WRONG", "CONSTRAINT_MISSING",
        "GENERATOR_CODE_BUG", "LLM_PROMPT_GAP",
        "CONSTRAINT_GENERATOR_BOTH", "EXECUTION_ENV_ERROR", "UNKNOWN",
    ]
    cat_labels = {
        "CONSTRAINT_WRONG": "约束提取错误",
        "CONSTRAINT_MISSING": "约束缺失",
        "GENERATOR_CODE_BUG": "生成代码 Bug",
        "LLM_PROMPT_GAP": "LLM 提示词遗漏",
        "CONSTRAINT_GENERATOR_BOTH": "约束+生成双端问题",
        "EXECUTION_ENV_ERROR": "执行环境错误",
        "UNKNOWN": "未知",
    }

    counter = 0
    for cat in cat_order:
        items = by_category.get(cat)
        if not items:
            continue
        lines.append(f"### 2.{cat_order.index(cat) + 1} {cat_labels[cat]}（{len(items)} 个算子）")
        lines.append("")
        for c in items:
            counter += 1
            op = op_lookup.get(c["operator_name"], {})
            lines.append(f"#### {counter}. {c['operator_name']}")
            lines.append("")
            lines.append(f"- **分类**：{c['category_label']}（置信度 {c['confidence'] * 100:.0f}%）")
            lines.append(f"- **状态**：约束={op.get('constraint_status', '?')}, "
                         f"用例={op.get('case_generation_status', '?')}, "
                         f"执行={op.get('execution_status', '?')}")
            lines.append(f"- **说明**：{_md_escape(c.get('notes'))}")

            # 失败用例
            failed_records = [
                r for r in op.get("report_records", [])
                if r.get("run_result") == "FAILED"
            ]
            if failed_records:
                lines.append(f"- **失败用例数**：{len(failed_records)} / {len(op.get('report_records', []))}")
                sample = failed_records[:3]
                for r in sample:
                    lines.append(f"  - id={r.get('id')}: {r.get('failure_reason', '')[:200]}")

            # 证据
            if c.get("evidence"):
                lines.append("- **关键证据**：")
                for ev in c["evidence"][:5]:
                    lines.append(f"  - {ev}")

            # 二次分类
            if c.get("secondary_categories"):
                sec = ", ".join(c["secondary_categories"])
                lines.append(f"- **可能伴随问题**：{sec}")

            lines.append("")
    return "\n".join(lines)


def _gen_constraint_section(constraint_analyses: list[dict]) -> str:
    if not constraint_analyses:
        return ""
    lines = ["## 三、约束问题深度分析", ""]
    any_issue = False
    for ca in constraint_analyses:
        if not ca.get("constraint_issues"):
            continue
        any_issue = True
        lines.append(f"### {ca['operator_name']}")
        lines.append("")
        lines.append(f"- 源文档路径：`{ca.get('src_doc_path', '—')}`")
        lines.append(f"- 问题摘要：{ca.get('summary', '—')}")
        lines.append("")
        # 按 severity 排序
        issues = sorted(
            ca["constraint_issues"],
            key=lambda i: {"high": 0, "medium": 1, "low": 2}.get(i["severity"], 3),
        )
        for i, iss in enumerate(issues, 1):
            lines.append(f"#### 问题 {i}（{iss['severity'].upper()}）：{iss['category']}")
            lines.append("")
            lines.append(f"- **参数**：`{iss['param_name']}`")
            if iss.get("field_name"):
                lines.append(f"- **字段**：{iss['field_name']}")
            lines.append(f"- **源文档证据**：{_md_escape(iss.get('src_doc_evidence'))}")
            lines.append(f"- **约束当前状态**：{_md_escape(iss.get('json_constraints_state'))}")
            lines.append(f"- **根因推测**：{_md_escape(iss.get('likely_root_cause'))}")
            if iss.get("affected_prompt"):
                lines.append(f"- **相关提示词**：`{iss['affected_prompt']}`")
            if iss.get("affected_code"):
                lines.append(f"- **相关代码**：`{iss['affected_code']}`")
            if iss.get("fix_suggestion"):
                lines.append(f"- **修复建议**：{_md_escape(iss['fix_suggestion'])}")
            lines.append("")

    if not any_issue:
        return ""
    return "\n".join(lines)


def _gen_generator_section(generator_analyses: list[dict]) -> str:
    if not generator_analyses:
        return ""
    lines = ["## 四、生成代码 Bug 定位", ""]
    any_issue = False
    for ga in generator_analyses:
        if not ga.get("generator_issues"):
            continue
        any_issue = True
        lines.append(f"### {ga['operator_name']}")
        lines.append("")
        lines.append(f"- 摘要：{ga.get('summary', '—')}")
        lines.append("")
        for i, iss in enumerate(ga["generator_issues"], 1):
            lines.append(f"#### Bug {i}：`{iss['file']}:{iss['line']}` in `{iss['function']}()`")
            lines.append("")
            if iss.get("exception"):
                lines.append(f"- **异常**：`{iss['exception']}`")
            lines.append(f"- **代码片段**：")
            lines.append("")
            lines.append("```python")
            lines.append(iss.get("code_snippet", "—"))
            lines.append("```")
            lines.append("")
            lines.append(f"- **修复建议**：")
            lines.append("")
            lines.append("```python")
            lines.append(iss.get("fix_suggestion", "—"))
            lines.append("```")
            lines.append("")
    if not any_issue:
        return ""
    return "\n".join(lines)


def _gen_prompt_section(prompt_analyses: list[dict]) -> str:
    if not prompt_analyses:
        return ""
    lines = ["## 五、提示词优化建议", ""]
    any_issue = False
    for pa in prompt_analyses:
        if not pa.get("prompt_improvements"):
            continue
        any_issue = True
        lines.append(f"### {pa['operator_name']}")
        lines.append("")
        lines.append(f"- 摘要：{pa.get('summary', '—')}")
        lines.append("")
        for i, p in enumerate(pa["prompt_improvements"], 1):
            lines.append(f"#### 建议 {i}：`{p['prompt_file']}` 中的 `{p['prompt_name']}`")
            lines.append("")
            lines.append(f"- **问题**：{p['issue']}")
            lines.append(f"- **改进方向**：{p['improvement']}")
            lines.append(f"- **理由**：{p['rationale']}")
            if p.get("added_example"):
                lines.append("")
                lines.append("**新增示例内容**：")
                lines.append("")
                lines.append("```markdown")
                lines.append(p["added_example"])
                lines.append("```")
            lines.append("")
    if not any_issue:
        return ""
    return "\n".join(lines)


def _gen_action_section(
    scan_data: dict,
    classifications: list[dict],
) -> str:
    """TopN 优先级 + 可执行清单。"""
    lines = ["## 六、迭代优化优先级清单", ""]

    # 按失败算子数排序
    op_failure_count: dict[str, int] = {}
    for op in scan_data["operators"]:
        count = sum(
            1 for r in op.get("report_records", []) if r.get("run_result") == "FAILED"
        )
        if op.get("case_generation_status") == "failed":
            count += 1
        if count > 0:
            op_failure_count[op["operator_name"]] = count

    if op_failure_count:
        sorted_ops = sorted(op_failure_count.items(), key=lambda x: -x[1])
        lines.append("### 6.1 TopN 失败算子")
        lines.append("")
        cat_lookup = {c["operator_name"]: c for c in classifications}
        for i, (name, cnt) in enumerate(sorted_ops[:10], 1):
            cat = cat_lookup.get(name, {}).get("category_label", "未知")
            lines.append(f"{i}. **{name}**（{cnt} 个失败）— {cat}")
        lines.append("")

    # 可执行 checklist
    lines.append("### 6.2 可执行修复清单")
    lines.append("")
    todo_count = 0
    for c in classifications:
        if c["category"] == "GENERATOR_CODE_BUG":
            todo_count += 1
            lines.append(f"- [ ] 修复算子 **{c['operator_name']}** 的生成代码 Bug")
        elif c["category"] in ("CONSTRAINT_WRONG", "CONSTRAINT_MISSING"):
            todo_count += 1
            lines.append(f"- [ ] 重新提取算子 **{c['operator_name']}** 的约束（修复 prompt 或上游节点）")
        elif c["category"] == "LLM_PROMPT_GAP":
            todo_count += 1
            lines.append(f"- [ ] 优化提示词以覆盖算子 **{c['operator_name']}** 的遗漏约束")
        elif c["category"] == "EXECUTION_ENV_ERROR":
            todo_count += 1
            lines.append(f"- [ ] 排查算子 **{c['operator_name']}** 的执行环境")
    lines.append("")
    lines.append(f"**总任务数**：{todo_count}")
    lines.append("")

    return "\n".join(lines)


def generate_markdown_report(
    *,
    scan_data: dict,
    classification_data: dict,
    constraint_data: dict | None,
    generator_data: dict | None,
    prompt_data: dict | None,
    title: str = "operator-agent 算子迭代分析报告",
    mode: str = "全量",
) -> str:
    """生成完整的 Markdown 报告。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = scan_data["summary"]
    classifications = classification_data.get("classifications", [])
    constraint_analyses = (constraint_data or {}).get("analyses", [])
    generator_analyses = (generator_data or {}).get("analyses", [])
    prompt_analyses = (prompt_data or {}).get("analyses", [])

    parts = [
        f"# {title}",
        "",
        f"> 生成时间：{timestamp}",
        f"> 分析模式：{mode}",
        f"> 扫描算子数：{summary['total']}",
        "",
        _gen_summary_section(scan_data),
        _gen_failure_section(classifications, scan_data),
        _gen_constraint_section(constraint_analyses),
        _gen_generator_section(generator_analyses),
        _gen_prompt_section(prompt_analyses),
        _gen_action_section(scan_data, classifications),
        "---",
        "",
        "_本报告由 `operator-iteration-analyzer` skill 自动生成_",
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML 生成（明亮风格）
# --------------------------------------------------------------------------- #


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{
    --bg:#fff;--text:#1a202c;--text-secondary:#4a5568;--border:#e2e8f0;
    --primary:#3182ce;--primary-light:#ebf8ff;--success:#38a169;--warning:#d69e2e;
    --error:#e53e3e;--code-bg:#f7fafc;--shadow:0 4px 12px rgba(0,0,0,.08);
  }}
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;
    line-height:1.7;color:var(--text);background:var(--bg);padding:2rem}}
  h1{{font-size:2rem;border-bottom:3px solid var(--primary);padding-bottom:.6rem;margin-bottom:1rem}}
  h2{{font-size:1.5rem;margin-top:2rem;margin-bottom:1rem;color:var(--primary);
    border-bottom:2px solid var(--border);padding-bottom:.4rem}}
  h3{{font-size:1.2rem;margin-top:1.5rem;margin-bottom:.6rem;color:var(--text)}}
  h4{{font-size:1.05rem;margin-top:1.2rem;margin-bottom:.4rem;color:var(--text-secondary)}}
  table{{width:100%;border-collapse:collapse;margin:1rem 0}}
  th,td{{border:1px solid var(--border);padding:.6rem .8rem;text-align:left}}
  th{{background:var(--code-bg);font-weight:600}}
  tr:nth-child(even) td{{background:#f8fafc}}
  .tag{{display:inline-block;padding:.15em .6em;border-radius:10px;font-size:.85em;font-weight:600}}
  .tag-high{{background:#fff5f5;color:var(--error);border:1px solid #feb2b2}}
  .tag-medium{{background:#fffaf0;color:var(--warning);border:1px solid #fbd38d}}
  .tag-low{{background:var(--primary-light);color:var(--primary);border:1px solid #90cdf4}}
  pre{{background:#1e293b;color:#e2e8f0;padding:1rem 1.2rem;border-radius:8px;
    overflow-x:auto;margin:.6rem 0;font-size:.85rem}}
  code{{background:var(--code-bg);padding:.15em .4em;border-radius:3px;font-size:.9em}}
  pre code{{background:none;color:inherit;padding:0}}
  blockquote{{background:var(--primary-light);border-left:4px solid var(--primary);
    padding:.6rem 1rem;margin:.6rem 0;border-radius:0 6px 6px 0}}
  ul,ol{{margin:.6rem 0 .6rem 2rem}}
  .toc{{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;
    padding:1rem 1.2rem;margin:1rem 0}}
  .toc a{{color:var(--primary);text-decoration:none}}
  .toc a:hover{{text-decoration:underline}}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _md_to_html_basic(md: str) -> str:
    """非常简单的 markdown→html 转换（仅满足报告展示需要）。"""
    import re

    # 转义
    lines = md.split("\n")
    out: list[str] = []
    in_code = False
    code_buf: list[str] = []
    in_list = False

    def flush_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in lines:
        # code fence
        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + "\n".join(code_buf) + "</code></pre>")
                code_buf = []
                in_code = False
            else:
                flush_list()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            flush_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{m.group(2)}</h{level}>")
            continue

        # 引用
        if line.startswith(">"):
            flush_list()
            out.append(f"<blockquote>{line[1:].strip()}</blockquote>")
            continue

        # 表格（简单处理）
        if line.startswith("|") and "|" in line[1:]:
            flush_list()
            cells = [c.strip() for c in line.strip("|").split("|")]
            # 跳过分隔行
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            tag = "th" if out and out[-1].endswith("</tr>") else "td"
            if tag == "td":
                # 改成 td
                out.append(f"<td>{'</td><td>'.join(cells)}</td></tr>")
            else:
                out.append(f"<tr><th>{'</th><th>'.join(cells)}</th></tr>")
            continue

        # 列表
        m = re.match(r"^\s*-\s+(.+)$", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            content = m.group(1)
            # 处理粗体
            content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", content)
            # 处理 inline code
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            out.append(f"<li>{content}</li>")
            continue

        # 数字列表
        m = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if m:
            if not in_list:
                out.append("<ol>")
                in_list = True
            content = m.group(1)
            content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            out.append(f"<li>{content}</li>")
            continue

        # 空行
        if not line.strip():
            flush_list()
            continue

        # 普通段落
        flush_list()
        content = line
        content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", content)
        content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
        out.append(f"<p>{content}</p>")

    flush_list()
    return "\n".join(out)


def generate_html_report(
    *,
    scan_data: dict,
    classification_data: dict,
    constraint_data: dict | None,
    generator_data: dict | None,
    prompt_data: dict | None,
    title: str = "operator-agent 算子迭代分析报告",
    mode: str = "全量",
) -> str:
    md = generate_markdown_report(
        scan_data=scan_data,
        classification_data=classification_data,
        constraint_data=constraint_data,
        generator_data=generator_data,
        prompt_data=prompt_data,
        title=title,
        mode=mode,
    )
    body = _md_to_html_basic(md)
    return _HTML_TEMPLATE.format(title=title, body=body)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_json_or_none(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成算子迭代分析报告")
    parser.add_argument("--scan-result", required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--constraint-analysis")
    parser.add_argument("--generator-analysis")
    parser.add_argument("--prompt-analysis")
    parser.add_argument("--title", default="operator-agent 算子迭代分析报告")
    parser.add_argument("--mode", choices=["单算子", "全量"], default="全量")
    parser.add_argument(
        "--format", choices=["md", "html", "both"], default="md",
    )
    parser.add_argument("--output", "-o", required=True, help="报告输出路径")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scan_data = json.loads(Path(args.scan_result).read_text(encoding="utf-8"))
    classification_data = json.loads(Path(args.classification).read_text(encoding="utf-8"))
    constraint_data = _load_json_or_none(args.constraint_analysis)
    generator_data = _load_json_or_none(args.generator_analysis)
    prompt_data = _load_json_or_none(args.prompt_analysis)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format in ("md", "both"):
        md = generate_markdown_report(
            scan_data=scan_data,
            classification_data=classification_data,
            constraint_data=constraint_data,
            generator_data=generator_data,
            prompt_data=prompt_data,
            title=args.title,
            mode=args.mode,
        )
        md_path = out_path if args.format == "md" else out_path.with_suffix(".md")
        md_path.write_text(md, encoding="utf-8")
        logger.info("markdown report: %s", md_path)

    if args.format in ("html", "both"):
        html = generate_html_report(
            scan_data=scan_data,
            classification_data=classification_data,
            constraint_data=constraint_data,
            generator_data=generator_data,
            prompt_data=prompt_data,
            title=args.title,
            mode=args.mode,
        )
        html_path = out_path if args.format == "html" else out_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        logger.info("html report: %s", html_path)

    print(json.dumps({
        "report_generated": True,
        "format": args.format,
        "output": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())