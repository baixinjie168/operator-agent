"""Prompt and response parser for Layer-2 LLM single-parameter constraint extraction.

Used by ``_extract_long_tail`` in ``single_param_constraint.py`` to handle
long-tail patterns (~7%) that deterministic regex rules cannot match.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.utils.llm_common import JSON_BLOCK_RE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SINGLE_PARAM_EXTRACT_PROMPT = """\
你是一个算子参数约束提取专家。请从以下参数的描述文本中，提取该参数自身的约束条件（单参数约束）。

## 参数信息
- 参数名：{param_name}
- 参数类型：{param_type}

## 参数描述
{param_text}

## 提取规则
1. 只提取该参数自身的约束，不要提取与其他参数之间的关联约束
2. 关注以下类型的约束：
   - 取值范围（如 >0, >=1, 0~1, [-1,1] 等）
   - 允许值枚举（如 只支持 0, 1, 2）
   - 数据类型限制（如 仅支持 float32 和 float16）
   - shape 维度数量限制（如 支持 1~8 维）
   - shape 各维度的取值限制（如 每个维度不超过 2^31）
   - 对齐要求（如 shape 必须 16 字节对齐）
   - 空 Tensor 限制（如 不支持空 Tensor）
   - TensorList 内部一致性（如 所有 Tensor 的 dtype 必须一致）
   - 其他单参数自身约束
3. expr_type 命名规范：
   - 取值范围 → "self_value_range"
   - 允许值枚举 → "self_value_enum"
   - 数据类型限制 → "self_dtype_restriction"
   - 维度数量限制 → "self_shape_dim_range"
   - shape 维度值限制 → "self_shape_dim_value"
   - 对齐要求 → "self_alignment"
   - 空 Tensor 限制 → "self_shape_nonempty"
   - TensorList 一致性 → "self_consistency"
   - 其他 → "self_other"
4. expr 字段用 Python 风格的表达式描述约束，用参数名本身引用该参数
5. source_citation 填写原文中描述该约束的原始文本片段
6. platform 字段：
   - 适用于所有平台 → 填空字符串 ""
   - 仅适用于特定平台 → 填写平台名称
   - 适用于多个平台 → 用"、"分隔

## 输出格式
严格按以下 JSON 格式返回，不要添加任何其他文字：
[
  {{
    "expr_type": "self_value_range",
    "expr": "{param_name} > 0",
    "description": "{param_name} 必须大于0",
    "source_citation": "取值范围：大于0",
    "params": ["{param_name}"],
    "platform": ""
  }}
]

如果没有提取到任何约束，返回空数组 []

{semantic_rules_context}
"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_response(text: str) -> list[dict[str, Any]]:
    """Parse the LLM response text into a list of constraint dicts.

    Handles:
    - Raw JSON array
    - JSON array wrapped in ```json ... ``` code fences
    - Malformed responses (returns empty list)

    Each returned dict contains:
        expr_type, expr, description, source_citation, params, platform
    """
    text = text.strip()
    if not text:
        return []

    # Try to extract JSON from code fences first
    m = JSON_BLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("parse_response: failed to parse JSON: %s", text[:200])
        return []

    if not isinstance(data, list):
        return []

    results: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Require at minimum expr_type and description
        if not item.get("expr_type") and not item.get("description"):
            continue
        results.append({
            "expr_type": item.get("expr_type", ""),
            "expr": item.get("expr", ""),
            "description": item.get("description", ""),
            "source_citation": item.get("source_citation", ""),
            "params": item.get("params", []),
            "platform": item.get("platform", ""),
        })

    return results
