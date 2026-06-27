#!/usr/bin/env python3
"""提示词优化建议：对 LLM_PROMPT_GAP / CONSTRAINT_MISSING 类别的算子，
根据失败模式给出对应的 prompt 改进建议。

Usage:
    python analyze_prompt.py --scan-result scan_results.json \
        --constraint-analysis constraint_analysis.json \
        --operator aclnnAbs --output prompt_suggestions.json
"""

from __future__ import annotations

import argparse
import json
import logging
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
class PromptImprovement:
    """单个提示词优化建议。"""

    prompt_file: str
    prompt_name: str  # 提示词变量名
    issue: str  # 当前 prompt 漏掉了什么
    improvement: str  # 建议增加什么内容
    rationale: str  # 为什么这个改进能解决问题
    added_example: str | None = None  # 示例 prompt 内容


@dataclass
class PromptAnalysis:
    operator_name: str
    prompt_improvements: list[PromptImprovement] = field(default_factory=list)
    summary: str = ""


# --------------------------------------------------------------------------- #
# 失败模式 → 提示词映射
# --------------------------------------------------------------------------- #


# 失败模式关键词 → 提示词建议
_FAILURE_TO_PROMPTS: list[dict[str, Any]] = [
    {
        "match_keywords": ["dimensions.value", "None", "shape_wrong"],
        "prompts": [{
            "prompt_file": "packages/agent/src/agent/prompts/system.py",
            "prompt_name": "SHAPE_EXTRACT_PROMPT",
            "issue": "LLM 输出 dimensions.value 时偶尔混入 None 或非法格式",
            "improvement": "增加显式的输出格式约定：仅允许 [] / [min,max] / [[min,max],...] 三种格式",
            "rationale": (
                "本次失败中 json_constraints 的 dimensions.value 出现了 None 或非标准格式，"
                "导致 case_builder 的 shape 采样逻辑崩溃或产生非法 shape"
            ),
            "added_example": (
                "### 输出格式约定\n"
                "- 标量参数：dimensions.value = []\n"
                "- 固定 N 维：dimensions.value = [N, N]\n"
                "- 范围维度（如 0-8 维）：dimensions.value = [0, 8]\n"
                "- 逐维范围（如 [[1,1], [3,3]]）：dimensions.value = [[1,1], [3,3]]\n"
                "**禁止**输出 null、字符串或嵌套结构不规范的形式"
            ),
        }],
    },
    {
        "match_keywords": ["dtype.value", "dtype为空", "dtype_mismatch"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "DTYPE_EXTRACT_PROMPT",
                "issue": "LLM 未识别或未正确输出部分参数的 dtype 列表",
                "improvement": "增加 dtype 互斥规则和示例：当输入 dtype 确定时，输出 dtype 通常与之一致",
                "rationale": (
                    "本次失败中部分参数 dtype.value 为空，导致生成时无法选择合法 dtype，"
                    "ATK 报 dtype mismatch"
                ),
                "added_example": (
                    "### dtype 提取规则\n"
                    "1. 源文档明确列出的数据类型（如'支持 FLOAT16、FLOAT32'）必须全部提取\n"
                    "2. 当源文档说'与 x 同类型'时，从 x 的 dtype.value 复制\n"
                    "3. 当源文档未提及但参数是 Tensor 时，默认 ['FLOAT16', 'FLOAT32']\n"
                    "4. 当参数是 aclScalar 时，dtype 通常与对应 Tensor 一致"
                ),
            },
        ],
    },
    {
        "match_keywords": ["param_missing", "隐式参数", "implicit_param"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "IMPLICIT_PARAM_EXTRACT_PROMPT",
                "issue": "LLM 漏掉了不在函数签名中但出现在参数表或使用说明中的参数",
                "improvement": "强调：必须从参数表的所有行和使用说明中提取参数名（包括 alpha、axis 等）",
                "rationale": (
                    "本次失败中源文档明确列出参数 X，但 json_constraints.inputs 中缺失，"
                    "原因是 implicit_param_extract 节点认为它不在函数签名中就跳过了"
                ),
                "added_example": (
                    "### 隐式参数识别清单\n"
                    "以下类型的参数即便不在函数签名中也必须提取：\n"
                    "- alpha、beta、gamma（缩放/移位参数）\n"
                    "- axis、dim、axes（轴向参数）\n"
                    "- keepdim、transpose、broadcast（布尔开关）\n"
                    "- padding、stride、dilation、kernel_size（卷积相关）\n"
                    "- numLayers、bidirectional、batch_first（RNN 相关）\n\n"
                    "识别方法：扫描参数表每一行的'参数名'列，对照函数签名补齐"
                ),
            },
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "LLM_DESCRIPTION_EXTRACT_PROMPT",
                "issue": "llm_description_extract 子图未能识别表格外的描述信息",
                "improvement": "增强子图 prompt：要求 LLM 完整读取'参数说明'章节的全部文字",
                "rationale": (
                    "源文档中部分参数通过'xxx 表示 yyy'的形式出现在文字段落中，"
                    "LLM 提取时只关注了参数表，导致这些参数被遗漏"
                ),
                "added_example": (
                    "### 描述完整性扫描\n"
                    "1. 读取'参数说明'章节的所有段落\n"
                    "2. 对每个粗体参数名（如 **alpha**），在 inputs 中添加对应条目\n"
                    "3. 对每个'Tensor X 表示 Y'的句式，提取 X 为参数名"
                ),
            },
        ],
    },
    {
        "match_keywords": ["relation_missing", "constraints_in_parameters 为空"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "PARAM_RELATION_EXTRACT_PROMPT",
                "issue": "LLM 提取参数关系时漏掉了部分关系",
                "improvement": "增加关系提取的覆盖率要求：每个使用说明列都至少对应一条 relation",
                "rationale": (
                    "本次失败中 constraints_in_parameters 字段为空，原因是上游"
                    "param_relation_extract 节点的 LLM 输出 relation 数量 < 参数数量"
                ),
                "added_example": (
                    "### 关系提取覆盖率要求\n"
                    "对参数表中每个'使用说明'列含关系性描述的参数，至少输出 1 条 relation：\n"
                    "- 'X 与 Y 同类型' → type_equality 关系\n"
                    "- 'X 与 Y 维度一致' → shape_dependency 关系\n"
                    "- 'X 为标量' → self_* 关系\n"
                    "- '当 X 为 True 时 Y 存在' → presence_dependency 关系"
                ),
            },
        ],
    },
    {
        "match_keywords": ["expr", "Python 语法", "AST"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "BUILD_PARAM_RELATIONS_PROMPT",
                "issue": "LLM 输出的 expr 不是合法 Python 表达式",
                "improvement": "在 prompt 中强调 expr 必须是可被 Python eval() 执行的合法表达式",
                "rationale": (
                    "build_param_relations 的 AST 校验会拒绝非法 expr，导致 relation 被丢弃。"
                    "需要让 LLM 直接产出符合 Python 语法的表达式"
                ),
                "added_example": (
                    "### expr 输出规则\n"
                    "1. 必须是合法 Python 表达式（可用 eval() 执行）\n"
                    "2. 属性访问使用 `.dtype`、`.shape`、`.format`、`.range_value`\n"
                    "3. 比较使用 `==`、`!=`、`<`、`>`；逻辑使用 `and`、`or`、`not`\n"
                    "4. 范围判断示例：`x.shape[0] == 4 * hidden_size`\n"
                    "5. 条件表达式示例：`out.dtype == x.dtype if x.dtype else True`\n"
                    "6. **禁止**输出自然语言或伪代码"
                ),
            },
        ],
    },
    {
        "match_keywords": ["format", "is_support_discontinuous"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "DFORMAT_EXTRACT_PROMPT",
                "issue": "LLM 未正确填充 format 字段",
                "improvement": "明确：Tensor 参数 format.value 默认为 ['ND']，标量参数为 'N/A'",
                "rationale": (
                    "部分 Tensor 参数的 format.value 被错误设为空字符串或 'N/A'，"
                    "ATK 不接受空 format"
                ),
            },
        ],
    },
    {
        "match_keywords": ["array_length", "aclIntArray", "aclTensorList"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "ARRAY_LENGTH_EXTRACT_PROMPT",
                "issue": "LLM 漏掉了 aclTensorList / aclIntArray 等数组类型的 array_length",
                "improvement": "明确：aclTensorList / aclIntArray / aclFloatArray / aclBoolArray 必须有 array_length",
                "rationale": (
                    "ATK 调用 aclTensorList 类型参数时需要明确长度，"
                    "缺失 array_length 会导致生成阶段崩溃"
                ),
            },
        ],
    },
    {
        "match_keywords": ["optional", "Optional", "可选"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "OPTIONAL_EXTRACT_PROMPT",
                "issue": "LLM 未能识别含 'Optional' 后缀或描述中含'可省略'的参数",
                "improvement": "明确：参数名带 Optional 后缀，或描述中含'可省略/可选'字样时，is_optional=true",
                "rationale": (
                    "错误地标记可选参数为必选会导致生成阶段总是生成它，"
                    "如果该参数需要特定其他参数配合则会失败"
                ),
            },
        ],
    },
    {
        "match_keywords": ["allowed_range", "range_value", "取值范围"],
        "prompts": [
            {
                "prompt_file": "packages/agent/src/agent/prompts/system.py",
                "prompt_name": "ALLOWED_RANGE_EXTRACT_PROMPT",
                "issue": "LLM 未提取标量参数的取值范围",
                "improvement": "明确：源文档说'只支持 X'/'取值 Y 到 Z'时，allowed_range_value 必须输出对应值",
                "rationale": (
                    "allowed_range_value 缺失会导致生成器在范围内随机选值，可能产生非法 case"
                ),
            },
        ],
    },
]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def _match_failure_to_prompts(
    constraint_issues: list[dict],
) -> list[dict[str, Any]]:
    """根据约束问题列表匹配对应的 prompt 改进建议。"""
    matched: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for issue in constraint_issues:
        issue_text = json.dumps(issue, ensure_ascii=False)
        for entry in _FAILURE_TO_PROMPTS:
            for keyword in entry["match_keywords"]:
                if keyword.lower() in issue_text.lower():
                    for prompt in entry["prompts"]:
                        key = (prompt["prompt_file"], prompt["prompt_name"])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            matched.append(prompt)
                    break
    return matched


def analyze_prompt(
    operator_name: str,
    constraint_analysis: dict | None,
) -> PromptAnalysis:
    """对单个算子做 prompt 优化建议。"""
    analysis = PromptAnalysis(operator_name=operator_name)

    constraint_issues = []
    if constraint_analysis and constraint_analysis.get("operator_name") == operator_name:
        constraint_issues = constraint_analysis.get("constraint_issues", [])

    matched_prompts = _match_failure_to_prompts(constraint_issues)
    if not matched_prompts:
        analysis.summary = "未匹配到具体的提示词改进点（建议人工 review）"
        return analysis

    for p in matched_prompts:
        analysis.prompt_improvements.append(PromptImprovement(
            prompt_file=p["prompt_file"],
            prompt_name=p["prompt_name"],
            issue=p["issue"],
            improvement=p["improvement"],
            rationale=p["rationale"],
            added_example=p.get("added_example"),
        ))
    analysis.summary = f"生成 {len(matched_prompts)} 条提示词优化建议"
    return analysis


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 提示词优化建议")
    parser.add_argument("--constraint-analysis", help="analyze_constraint.py 的输出 JSON")
    parser.add_argument("--scan-result", help="scan_operators.py 的输出（用于读取失败原因）")
    parser.add_argument("--operator", action="append", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    constraint_lookup: dict[str, dict] = {}
    if args.constraint_analysis:
        data = json.loads(Path(args.constraint_analysis).read_text(encoding="utf-8"))
        for a in data.get("analyses", []):
            constraint_lookup[a["operator_name"]] = a

    analyses = []
    for op_name in args.operator:
        analysis = analyze_prompt(op_name, constraint_lookup.get(op_name))
        d = asdict(analysis)
        d["prompt_improvements"] = [asdict(i) for i in analysis.prompt_improvements]
        analyses.append(d)
        logger.info("%s: %s", op_name, analysis.summary)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"analyses": analyses}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "analyzed": len(analyses),
        "operators": [
            {"operator": a["operator_name"], "suggestion_count": len(a["prompt_improvements"]), "summary": a["summary"]}
            for a in analyses
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())