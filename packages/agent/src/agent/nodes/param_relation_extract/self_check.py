"""Self-check and targeted extraction prompts and functions.

Round 3: Focused extraction for uncovered parameters and paragraphs.
Round 4: LLM self-reflection to find missed relations.
"""

import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.nodes.param_relation_extract.extract_relations import _parse_relations_response
from agent.nodes.param_relation_extract.prompts import RELATION_TYPE_DEFINITIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts — all outputs include param_optional for schema consistency
# ---------------------------------------------------------------------------

FOCUSED_RELATION_PROMPT = """\
你是参数关系提取专家。
以下文档内容与参数 "{param_name}" 相关。

提取 "{param_name}" 与其他参数之间的所有耦合关系。

## 关系类型
{relation_types}

## 规则
1. 只提取涉及 2+ 参数的关系
2. 对每条关系，列出所有涉及的参数名
3. source_citation 必须是文档中的原始文本
4. param_optional 标注每个参数是否可选（从文档中的"可选参数"等描述判断）
5. 如果该参数没有找到关系，返回空数组 []

## 输出格式
严格按以下 JSON 数组返回：
[{{
  "relation_type": "shape",
  "platform": "",
  "description": "...",
  "params": ["{param_name}", "other_param"],
  "param_optional": {{"{param_name}": false, "other_param": false}},
  "source_citation": "原始文本..."
}}]

## 文档内容
{context}
"""

PARAGRAPH_RELATION_PROMPT = """\
你是参数关系提取专家。
以下文档段落同时提及了多个参数，可能包含它们之间的隐含耦合关系。

## 关系类型
{relation_types}

## 涉及的参数
{mentioned_params}

## 规则
1. 只提取涉及 2+ 参数的关系
2. source_citation 必须是段落中的原始文本
3. param_optional 标注每个参数是否可选
4. 如果段落中没有参数间关系，返回空数组 []

## 输出格式
严格按以下 JSON 数组返回：
[{{
  "relation_type": "value",
  "platform": "",
  "description": "...",
  "params": ["param_a", "param_b"],
  "param_optional": {{"param_a": false, "param_b": false}},
  "source_citation": "原始文本..."
}}]

## 段落内容
{paragraph}
"""

SELF_CHECK_PROMPT = """\
你是约束关系审查专家。
以下文档内容已处理完毕，提取了这些参数关系：

## 已提取关系（共 {count} 条）
{extracted_relations}

## 文档内容
{section_content}

## 你的任务
检查文档中是否有遗漏的参数耦合关系。
重点关注：
1. 散文段落中的隐含关系（不在表格中的）
2. 涉及 3+ 参数的关系，可能被简化了
3. 不同段落之间的交叉引用
4. 条件性关系（"当X时，Y需要满足..."）

如果发现遗漏关系，按相同 JSON 格式返回。
如果没有遗漏，返回空数组 []。
不要返回已在提取列表中的关系（对比 description 和 source_citation）。

## 输出格式
严格按以下 JSON 数组返回：
[{{
  "relation_type": "...",
  "platform": "",
  "description": "...",
  "params": ["..."],
  "param_optional": {{...}},
  "source_citation": "原始文本..."
}}]
"""


# ---------------------------------------------------------------------------
# Round 3: Targeted extraction functions
# ---------------------------------------------------------------------------

async def extract_relations_for_param(
    llm: ChatOpenAI,
    section_content: str,
    param_name: str,
    param_names: list[str],
) -> list[dict[str, Any]]:
    """Round 3a: focused extraction for an uncovered parameter."""
    from agent.nodes.context_utils import extract_param_context

    context = extract_param_context(section_content, param_name)
    prompt = FOCUSED_RELATION_PROMPT.format(
        param_name=param_name,
        relation_types=RELATION_TYPE_DEFINITIONS,
        context=context,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    relations = _parse_relations_response(text)

    for r in relations:
        r["_source"] = f"targeted_param:{param_name}"

    return relations


async def extract_relations_for_paragraph(
    llm: ChatOpenAI,
    paragraph: str,
    mentioned_params: list[str],
) -> list[dict[str, Any]]:
    """Round 3b: focused extraction for an uncovered paragraph."""
    prompt = PARAGRAPH_RELATION_PROMPT.format(
        relation_types=RELATION_TYPE_DEFINITIONS,
        mentioned_params=", ".join(mentioned_params),
        paragraph=paragraph,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    relations = _parse_relations_response(text)

    for r in relations:
        r["_source"] = "targeted_paragraph"

    return relations


# ---------------------------------------------------------------------------
# Round 4: Self-reflection
# ---------------------------------------------------------------------------

async def self_check_relations(
    llm: ChatOpenAI,
    extracted: list[dict[str, Any]],
    section_content: str,
) -> list[dict[str, Any]]:
    """Round 4: LLM self-reflection to find missed relations.

    Summary includes description(50c) + source_citation(40c)
    so the LLM can judge whether existing relations cover a passage.
    """
    summary_lines = []
    for i, r in enumerate(extracted, 1):
        params = ", ".join(r.get("params", []))
        desc = r.get("description", "")[:50]
        src = r.get("source_citation", "")[:40]
        summary_lines.append(
            f"{i}. [{r.get('relation_type', '')}] {params}\n"
            f"   desc: {desc}...\n"
            f"   src: {src}..."
        )

    prompt = SELF_CHECK_PROMPT.format(
        count=len(extracted),
        extracted_relations="\n".join(summary_lines),
        section_content=section_content,
    )
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    additional = _parse_relations_response(text)

    for r in additional:
        r["_source"] = "self_check"

    return additional
