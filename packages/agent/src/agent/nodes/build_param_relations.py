"""BuildParamRelations node: enrich param_relations with expr_type/expr and group by platform.

Implements three-layer protection for expression generation accuracy:
- Phase 0: Deterministic validation (AST syntax + reference checks)
- Phase 1: Prompt enhancement (Few-shot examples + confidence scoring)
- Phase 2: Failure remediation (enhanced retry + semantic verification)

Emits NODE_PROGRESS events for the frontend constraint detail panel
(data_ready → extract_done → complete).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import RELATION_OBJECT_BUILD_PROMPT
from agent.core.llm import create_llm
from agent.utils.expr_validation import _simplify_expr
from agent.utils.expr_validation import validate_expr as _validate_expr
from agent.utils.expr_validation import validate_none_guard
from agent.utils.llm_common import CONCURRENCY_LIMIT, parse_json_response
from agent.runtime.context import get_context
from agent.runtime.events import EventType, Span, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# Phase 2a: Few-shot examples for enhanced retry
FEW_SHOT_EXAMPLES = {
    "syntax_implies": {
        "bad": "x.shape implies y.shape",
        "good": "(y.shape == x.shape) if True else True",
        "note": "Python 没有 implies 运算符，用 if/else 表达蕴含",
    },
    "syntax_null": {
        "bad": "x.shape[0] == null",
        "good": "x.shape[0] == None",
        "note": "Python 使用 None，不是 null",
    },
    "ref_hallucination": {
        "bad": "x.shape == z.shape (params=['x','y'])",
        "good": "x.shape == y.shape",
        "note": "只能引用 params 列表中的参数",
    },
    "attr_hallucination": {
        "bad": "len(x.shape) == len(y.dims)",
        "good": "len(x.shape) == len(y.shape)",
        "note": "合法属性只有: shape, dtype, format, range_value",
    },
    "condition_direction": {
        "bad": "(x.shape[0]==y.shape[0]) if axis==0 else False",
        "good": "(x.shape[0]==y.shape[0]) if axis==0 else True",
        "note": "条件不满足时返回 True（约束不适用）",
    },
    "quantifier": {
        "bad": "x.shape[0] > 0",
        "good": "all(d > 0 for d in x.shape)",
        "note": "'所有维度' 必须用 all() 表达全称量词",
    },
    "syntax_tuple": {
        "bad": "tuple(x.shape) == tuple(y.shape)",
        "good": "list(x.shape) == list(y.shape)",
        "note": "禁止使用 tuple()，用 list() 代替，或直接用 x.shape == y.shape 比较",
    },
    # FIX-16: Additional examples for None guard, conditional, enum
    "none_guard": {
        "bad": "biasOptional.shape[0] == N",
        "good": "(biasOptional.shape[0] == N) if biasOptional is not None else True",
        "note": "可选参数的属性引用必须有 None 守卫包装",
    },
    "conditional_branch": {
        "bad": "N1 == 2 * K2",
        "good": "(N1 == 2 * K2) if (activation.range_value in [geglu, swiglu, reglu]) else True",
        "note": "条件约束必须保留条件守卫，条件不满足时返回 True",
    },
    "enum_completeness": {
        "bad": "groupType.range_value in [0, 1]",
        "good": "groupType.range_value in [-1, 0, 2]",
        "note": "枚举值必须与文档完全一致，不可遗漏",
    },
}

# Phase 2b: Semantic verification prompt
EXPR_VERIFY_PROMPT = """\
你是一个参数约束表达式审查专家。

## 任务
验证以下表达式是否正确反映了自然语言描述的约束关系。

## 自然语言描述
{description}

## 原始引用
{source_citation}

## 参数列表
{params}

## 参数 shape 信息
{param_shapes_text}

## 待验证的表达式
expr_type: {expr_type}
expr: {expr}

## 检查清单
1. expr 中引用的参数是否都在参数列表中？
2. 条件逻辑方向是否正确？（"当A时B" → B if A else True）
3. 量词是否正确？（"所有维度" → all(), "任意一维" → any()）
4. 边界条件是否正确？（开区间 vs 闭区间）
5. 广播关系的 expr 是否正确？（允许维度为 1）
6. expr_type 是否与描述的关系类型匹配？参照以下对照表：
   - expr 含 param.range_value in [...] 且仅引用1个参数 → self_value_enum
   - expr 含 param.shape[i] == ... 且引用条件参数 → shape_value_dependency
   - expr 含 param.dtype == ... → type_equality
   - expr 含 (param is not None) == (param is not None) → presence_dependency
   - expr 含 param % N == 0 → self_alignment
7. 维度索引是否正确？（对照"参数 shape 信息"，确认 shape[i] 引用的确实是描述中所指的维度；当参数有多种 shape 形式时，应使用负索引 shape[-N]）
8. 可选参数（名称含 Optional）的属性引用是否有 is not None 守卫？
   即 expr 应包含 (paramName is not None) 的条件包装
9. 维度数 vs 维大小：当 source_citation/description 出现"N维""N-M维""维度数""shape size"
   等**维数**语义时，expr 必须用 len(x.shape)，不得用 shape[i]（第i维的**大小**）。
   错误: "2-6维" → 2 <= x.shape[0] <= 6
   正确: "2-6维" → 2 <= len(x.shape) <= 6

## 输出
严格按以下 JSON 返回：
{{"is_correct": true, "reason": "表达式正确"}}
或
{{"is_correct": false, "reason": "错误原因", "corrected_expr": "修正后的表达式", "corrected_expr_type": "修正后的expr_type"}}
"""


def _format_signatures(sigs: list[dict], params: list[dict] | None = None) -> str:
    """Build a concise signature text for LLM context, enriched with shape info.

    When params is provided, Tensor parameters are annotated with their
    shape constraints (e.g. "[shape: (T,N,C)或(T,C)]"), which is critical
    for the LLM to choose correct dimension indices in expressions.
    """
    if not sigs:
        return "（无函数签名信息）"

    # Build shape lookup: (function_name, param_name) → shape string
    shape_map: dict[tuple[str, str], str] = {}
    if params:
        for p in params:
            shape = (p.get("shape", "") or "").strip()
            if shape:
                key = (p.get("function_name", ""), p.get("param_name", ""))
                shape_map[key] = shape

    lines: list[str] = []
    for sig in sigs:
        fn = sig.get("function_name", "")
        sig_params = sig.get("parameters", [])
        param_strs = []
        for p in sig_params:
            name = p.get("name", "")
            ptype = p.get("type", "")
            s = f"{name}: {ptype}"
            shape = shape_map.get((fn, name), "")
            if shape:
                s += f" [shape: {shape}]"
            param_strs.append(s)
        lines.append(f"{fn}({', '.join(param_strs)})")
    return "\n".join(lines)


def _build_param_shapes_text(params: list[dict]) -> str:
    """Build a compact shape reference table for LLM context.

    Lists each parameter with its shape (if any) so the LLM can choose
    correct dimension indices (e.g. shape[-1] for the last dim).
    """
    if not params:
        return "（无参数 shape 信息）"
    lines: list[str] = []
    for p in params:
        name = p.get("param_name", "")
        shape = (p.get("shape", "") or "").strip()
        ptype = (p.get("param_type", "") or "").strip()
        if shape:
            lines.append(f"- {name} ({ptype}): shape = {shape}")
    if not lines:
        return "（参数无 shape 信息）"
    return "\n".join(lines)


def _parse_relation_object_response(text: str) -> dict[str, str]:
    """Parse LLM response into {expr_type, expr, confidence, uncertainty_reason} dict."""
    data = parse_json_response(text, dict)
    if data is not None:
        return {
            "expr_type": data.get("expr_type", ""),
            "expr": data.get("expr", ""),
            "confidence": data.get("confidence", "high"),
            "uncertainty_reason": data.get("uncertainty_reason", ""),
        }
    logger.warning("BuildParamRelations: failed to parse LLM response: %s", text[:200])
    return {"expr_type": "", "expr": "", "confidence": "high", "uncertainty_reason": ""}


# ---------------------------------------------------------------------------
# Phase 0 validation is now in agent.utils.expr_validation
# (imported as _validate_expr, _validate_expr_refs, _validate_expr_syntax)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2a: Enhanced retry with Few-shot examples
# ---------------------------------------------------------------------------


def _select_relevant_example(error: str, expr: str) -> str:
    """Select the most relevant Few-shot example based on error type.

    R17: use precise error-message keyword matching instead of broad
    expr-content matching, to avoid false-matching legal expressions.
    """
    error_lower = error.lower()
    expr_lower = expr.lower()

    # R17: precise error-message matching for FIX-16 examples
    if "without none guard" in error_lower:
        ex = FEW_SHOT_EXAMPLES["none_guard"]
    elif "contradiction" in error_lower:
        ex = FEW_SHOT_EXAMPLES["conditional_branch"]
    elif "enum" in error_lower and "in [" in expr_lower:
        ex = FEW_SHOT_EXAMPLES["enum_completeness"]
    elif "implies" in expr_lower:
        ex = FEW_SHOT_EXAMPLES["syntax_implies"]
    elif "null" in expr_lower:
        ex = FEW_SHOT_EXAMPLES["syntax_null"]
    elif "tuple" in expr_lower:
        ex = FEW_SHOT_EXAMPLES["syntax_tuple"]
    elif "unknown parameter" in error_lower:
        ex = FEW_SHOT_EXAMPLES["ref_hallucination"]
    elif "unknown attribute" in error_lower:
        ex = FEW_SHOT_EXAMPLES["attr_hallucination"]
    else:
        # Default: condition direction example
        ex = FEW_SHOT_EXAMPLES["condition_direction"]

    return f"""
## 参考示例（避免类似错误）
错误示例: {ex['bad']}
正确示例: {ex['good']}
注意: {ex['note']}
"""


async def _extract_one_with_hint(
    llm: ChatOpenAI,
    rel: dict,
    signatures_text: str,
    param_shapes_text: str,
    example_hint: str,
    implicit_params_text: str = "",
) -> dict[str, str]:
    """Extract with Few-shot example hint for retry."""
    prompt = RELATION_OBJECT_BUILD_PROMPT.format(
        signatures_text=signatures_text,
        param_shapes_text=param_shapes_text,
        implicit_params_text=implicit_params_text,
        relation_type=rel.get("relation_type", ""),
        params=json.dumps(rel.get("params", []), ensure_ascii=False),
        description=rel.get("description", ""),
        source_citation=rel.get("source_citation", ""),
    )
    prompt += "\n" + example_hint
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_relation_object_response(text)


async def _extract_with_retry(
    llm: ChatOpenAI,
    rel: dict,
    signatures_text: str,
    param_shapes_text: str,
    sem: asyncio.Semaphore,
    implicit_params_text: str = "",
    external_constants: set[str] | None = None,
    implicit_param_names: set[str] | None = None,
    param_optional_map: dict[str, bool] | None = None,
) -> dict[str, str]:
    """Phase 2a: Extract with enhanced retry (max 2 attempts).

    On validation failure, inject relevant Few-shot example before retrying.
    FIX-9: also validates None guards for optional params (R14: standalone).
    """
    last_error = ""
    last_expr = ""

    for attempt in range(settings.expr_max_retries + 1):
        async with sem:
            try:
                if attempt == 0:
                    prompt = RELATION_OBJECT_BUILD_PROMPT.format(
                        signatures_text=signatures_text,
                        param_shapes_text=param_shapes_text,
                        implicit_params_text=implicit_params_text,
                        relation_type=rel.get("relation_type", ""),
                        params=json.dumps(rel.get("params", []), ensure_ascii=False),
                        description=rel.get("description", ""),
                        source_citation=rel.get("source_citation", ""),
                    )
                    response = await llm.ainvoke(prompt)
                    text = response.content if hasattr(response, "content") else str(response)
                    result = _parse_relation_object_response(text)
                else:
                    example_hint = _select_relevant_example(last_error, last_expr)
                    result = await _extract_one_with_hint(
                        llm, rel, signatures_text, param_shapes_text, example_hint,
                        implicit_params_text=implicit_params_text,
                    )

                expr = result.get("expr", "")
                params = rel.get("params", [])

                # Phase 0 validation
                is_valid, error = _validate_expr(
                    expr, params, external_constants, implicit_param_names,
                )
                # FIX-9: None guard validation (R14: standalone, alongside
                # _validate_expr; does NOT modify validate_expr signature)
                if is_valid:
                    guard_ok, guard_error = validate_none_guard(
                        expr, params, param_optional_map,
                    )
                    if not guard_ok:
                        is_valid = False
                        error = guard_error

                if is_valid:
                    return result

                last_error = error
                last_expr = expr

                if attempt < settings.expr_max_retries:
                    logger.warning(
                        "BuildParamRelations: expr validation failed (attempt %d/%d) "
                        "for relation id=%s: %s — retrying with example",
                        attempt + 1, settings.expr_max_retries + 1, rel.get("id", "?"), error,
                    )
                else:
                    logger.error(
                        "BuildParamRelations: expr failed after %d attempts "
                        "for relation id=%s: %s — storing empty expr",
                        settings.expr_max_retries + 1, rel.get("id", "?"), error,
                    )
                    return {
                        "expr_type": result.get("expr_type", ""),
                        "expr": "",
                        "_validation_error": error,
                    }
            except Exception:
                logger.warning(
                    "BuildParamRelations: LLM failed for relation id=%s",
                    rel.get("id", "?"),
                )
                return {"expr_type": "", "expr": ""}

    return {"expr_type": "", "expr": ""}


# ---------------------------------------------------------------------------
# Phase 2b: Semantic verification
# ---------------------------------------------------------------------------


async def _call_verify_llm(
    llm: ChatOpenAI,
    rel: dict,
    expr_result: dict,
    param_shapes_text: str,
    sem: asyncio.Semaphore,
) -> dict:
    """Phase 2b: Call verification LLM to check semantic correctness."""
    async with sem:
        prompt = EXPR_VERIFY_PROMPT.format(
            description=rel.get("description", ""),
            source_citation=rel.get("source_citation", ""),
            params=json.dumps(rel.get("params", []), ensure_ascii=False),
            param_shapes_text=param_shapes_text,
            expr_type=expr_result.get("expr_type", ""),
            expr=expr_result.get("expr", ""),
        )
        response = await llm.ainvoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        data = parse_json_response(text, dict)
        if isinstance(data, dict):
            return data
        logger.warning("BuildParamRelations: failed to parse verify response: %s", text[:200])
        return {"is_correct": True, "reason": "parse_failed, accepting original"}


async def _verify_and_fix(
    llm: ChatOpenAI,
    rel: dict,
    expr_result: dict,
    param_shapes_text: str,
    sem: asyncio.Semaphore,
) -> dict[str, str]:
    """Phase 2b: Verify and fix expression with loop-back validation.

    If verification LLM returns corrected_expr, it must pass Phase 0 again.
    """
    try:
        verify_result = await _call_verify_llm(llm, rel, expr_result, param_shapes_text, sem)
    except Exception:
        logger.warning(
            "BuildParamRelations: semantic verification failed for relation id=%s",
            rel.get("id", "?"),
        )
        return expr_result  # Accept original on verification failure

    if verify_result.get("is_correct", True):
        return expr_result

    # Verification failed, use corrected_expr
    corrected = verify_result.get("corrected_expr", "")
    if not corrected:
        logger.warning(
            "BuildParamRelations: verification failed but no corrected_expr "
            "for relation id=%s",
            rel.get("id", "?"),
        )
        return expr_result  # Accept original if no correction

    # Critical: corrected_expr must pass Phase 0 validation
    is_valid, error = _validate_expr(corrected, rel.get("params", []))
    if not is_valid:
        logger.warning(
            "BuildParamRelations: corrected_expr failed validation for "
            "relation id=%s: %s — accepting original",
            rel.get("id", "?"), error,
        )
        return expr_result  # Accept original if correction is invalid

    logger.info(
        "BuildParamRelations: corrected expr for relation id=%s: %s",
        rel.get("id", "?"), verify_result.get("reason", ""),
    )
    # Use corrected_expr_type if provided (Phase 2b can fix expr_type
    # misclassification, e.g. shape_value_dependency → self_value_enum)
    corrected_type = verify_result.get("corrected_expr_type", "")
    return {
        "expr_type": corrected_type if corrected_type else expr_result.get("expr_type", ""),
        "expr": corrected,
        "_corrected": True,
        "_correction_reason": verify_result.get("reason", ""),
    }


def _needs_rank_verify(rel: dict, result: dict) -> bool:
    """Fix 2C-c: narrow Phase-2b forced re-verification to rank-vs-dim cases.

    Only triggers when a single-param shape_dependency / shape_value_dependency
    expr uses shape[i] while the source text mentions 维 (rank semantics) —
    the canonical misuse where "支持N-M维" is mistranslated to a shape[i]
    range instead of len(x.shape). Avoids re-running the verify LLM on every
    high-confidence shape_dependency (which would risk corrupting correct exprs).
    """
    if not result.get("expr"):
        return False
    et = result.get("expr_type", "")
    if et not in ("shape_dependency", "shape_value_dependency"):
        return False
    if "shape[" not in result["expr"]:
        return False
    src = (rel.get("source_citation", "") or "") + (rel.get("description", "") or "")
    return "维" in src


async def _batch_extract_relation_objects(
    relations: list[dict],
    signatures_text: str,
    param_shapes_text: str,
    implicit_params_text: str = "",
    external_constants: set[str] | None = None,
    implicit_param_names: set[str] | None = None,
    param_optional_map: dict[str, bool] | None = None,
) -> list[dict[str, str]]:
    """Batch LLM extraction with three-layer protection.

    Integration:
    - Phase 1: Prompt enhancement (Few-shot + confidence) — in RELATION_OBJECT_BUILD_PROMPT
    - Phase 2a: Enhanced retry on Phase 0 validation failure
    - Phase 2b: Semantic verification for low-confidence results

    Returns a list of dicts per relation, with optional metadata keys:
    - _validation_error: Phase 0 validation error message
    - _corrected: True if expr was corrected by Phase 2b
    - _correction_reason: Why the expr was corrected
    """
    if not relations:
        return []

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    try:
        llm = create_llm()
    except Exception:
        logger.exception("BuildParamRelations: failed to create LLM")
        return [{"expr_type": "", "expr": ""}] * len(relations)

    async def _process_one(rel: dict) -> dict[str, str]:
        from agent.nodes.build_param_constraint.complexity_classify import is_complex_relation
        from agent.nodes.build_param_constraint.complex_relation_agent import generate_expr_via_agent

        if is_complex_relation(rel):
            # Type 4: complex conditional -> DeepAgent + skill.md
            result = await generate_expr_via_agent(
                rel, signatures_text, param_shapes_text, implicit_params_text,
            )
            # Phase 0 safety net (Agent has no tools; validate in Python)
            expr = result.get("expr", "")
            if expr:
                is_valid, error = _validate_expr(
                    expr, rel.get("params", []),
                    external_constants, implicit_param_names,
                )
                # FIX-9: None guard validation for complex relations too
                if is_valid:
                    guard_ok, guard_error = validate_none_guard(
                        expr, rel.get("params", []), param_optional_map,
                    )
                    if not guard_ok:
                        is_valid = False
                        error = guard_error
                if not is_valid:
                    # Phase 3 Item 8: attempt post-generation simplification
                    # before giving up. If the expr was rejected for excessive
                    # redundancy (FFNV3 [50] copy-paste pattern), _simplify_expr
                    # factors out the repeated sub-expression and may produce a
                    # valid form without another LLM round.
                    if settings.expr_simplify:
                        simplified = _simplify_expr(expr)
                        if simplified != expr:
                            is_valid2, _ = _validate_expr(
                                simplified, rel.get("params", []),
                                external_constants, implicit_param_names,
                            )
                            if is_valid2:
                                result["expr"] = simplified
                                result["_simplified"] = True
                                logger.info(
                                    "ComplexRelationAgent: simplified expr "
                                    "validated for id=%s (%d -> %d chars)",
                                    rel.get("id", "?"), len(expr), len(simplified),
                                )
                                return result
                    # Phase 3 Item 8: simplified form still invalid (or not
                    # simplified) -> retry Agent once with error hint guiding
                    # factored form. DeepAgent calls are costly, so only 1 retry
                    # (unlike simple relations which retry up to expr_max_retries).
                    logger.warning(
                        "ComplexRelationAgent expr invalid for id=%s: %s — retrying with hint",
                        rel.get("id", "?"), error,
                    )
                    retry_result = await _retry_complex_with_hint(
                        rel, error, signatures_text,
                        param_shapes_text, implicit_params_text,
                    )
                    expr2 = retry_result.get("expr", "")
                    if expr2:
                        is_valid3, _ = _validate_expr(
                            expr2, rel.get("params", []),
                            external_constants, implicit_param_names,
                        )
                        if is_valid3:
                            return retry_result
                    # Retry still failed -> empty expr + marker (original behaviour)
                    result["expr"] = ""
                    result["_validation_error"] = error
        else:
            # Type 1: simple param relation -> current single-shot LLM
            result = await _extract_with_retry(
                llm, rel, signatures_text, param_shapes_text, sem,
                implicit_params_text=implicit_params_text,
                external_constants=external_constants,
                implicit_param_names=implicit_param_names,
                param_optional_map=param_optional_map,
            )

            # Check if semantic verification is needed (Phase 2b)
            confidence = result.get("confidence", "high")
            if confidence == "low" and result.get("expr"):
                # Force semantic verification for low confidence
                result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)
            elif confidence == "medium" and result.get("expr"):
                # Phase 2b verification for medium confidence — enables
                # expr_type correction and None-guard checks that the
                # initial single-shot LLM may have missed.
                result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)
            elif confidence == "high" and _needs_rank_verify(rel, result):
                # Fix 2C-c: narrowed forced re-verification — only the
                # rank-vs-dim misuse ("N维" → shape[i] instead of len(shape))
                # is worth re-running the verify LLM on a high-confidence
                # result; broader triggers risk corrupting correct exprs.
                result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)

        return result

    results = await asyncio.gather(*[_process_one(r) for r in relations])
    return list(results)


async def _retry_complex_with_hint(
    rel: dict,
    error: str,
    signatures_text: str,
    param_shapes_text: str,
    implicit_params_text: str,
) -> dict[str, str]:
    """Retry a complex relation (type 4) after Phase 0 validation failure.

    Injects the validation error as a hint into the user message, asking
    the Agent to regenerate using the factored form (all((expr) if cond
    else True for ...)) per the mutual_exclusion.md / SKILL.md rules.

    Only ONE retry — DeepAgent calls are costly. On failure, callers fall
    back to empty expr + ``_validation_error`` marker (original behaviour).
    """
    from agent.nodes.build_param_constraint.complex_relation_agent import (
        _get_complex_relation_agent,
        _extract_ai_text,
        _parse_agent_response,
    )

    hint = (
        "## 上次生成失败\n"
        "上次生成的 expr 校验失败：" + error + "\n"
        "请按因式分解规则重新生成简洁表达式：\n"
        "- 多场景互斥用 all((expr) if (cond) else True for ...) 形式\n"
        "- 禁止重复子表达式 3+ 次\n"
        "- expr 总长度不超过 500 字符\n"
    )
    user_msg = (
        "## Relation\n"
        "- relation_type: " + rel.get("relation_type", "") + "\n"
        "- params: " + json.dumps(rel.get("params", []), ensure_ascii=False) + "\n"
        "- description: " + rel.get("description", "") + "\n"
        "- source_citation: " + rel.get("source_citation", "") + "\n\n"
        + hint + "\n"
        "## Function Signatures\n" + signatures_text + "\n\n"
        "## Parameter Shapes\n" + param_shapes_text + "\n\n"
        + implicit_params_text + "\n"
    )
    try:
        agent = _get_complex_relation_agent()
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_msg}],
        })
        ai_text = await _extract_ai_text(result)
        parsed = _parse_agent_response(ai_text)
    except Exception:
        logger.exception("ComplexRelationAgent retry failed")
        return {"expr_type": "", "expr": "", "confidence": "low"}

    # Apply simplification to the retry result as well
    if settings.expr_simplify:
        expr = parsed.get("expr", "")
        if expr and len(expr) > 500:
            simplified = _simplify_expr(expr)
            if simplified != expr:
                parsed["expr"] = simplified
                parsed["_simplified"] = True
    return parsed


# ---------------------------------------------------------------------------
# Constant substitution (only replace known constant values, not dim vars)
# ---------------------------------------------------------------------------


def _substitute_dim_vars(
    expr: str,
    mappings: list[dict],
    external_constants: set[str] | None = None,
) -> str:
    """Replace only constant dimension values (e.g. k0 -> 16).

    Named dimension variables (BS, H, N, etc.) are kept as-is — they are
    treated as implicit parameters referenced by name in expressions.

    External constants are also kept as-is.
    """
    if not expr or not mappings:
        return expr

    substitutions: dict[str, str] = {}

    for m in mappings:
        var = m["var_name"]
        if m.get("is_constant"):
            val = str(m.get("constant_value", 0))
            substitutions[f"{var}.range_value"] = val
            substitutions[var] = val

    # Apply substitutions (longest key first to avoid partial matches)
    for old, new in sorted(substitutions.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(old, new)

    return expr


async def build_param_relations_node(state: PipelineState) -> dict[str, Any]:
    """Build relation_object for each param_relation row and group by platform.

    Flow:
    1. Query param_relations, function_signatures, platform_support
    2. LLM batch extract expr_type + expr from description
    3. Assemble relation_object per row and persist to DB
    4. Group by platform (platform="" → all supported platforms)
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("BuildParamRelations: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("BuildParamRelations: no doc_id, skipping")
        return {"error": None}

    try:
        # Step 1: Query data sources
        relations = await _mcp_client.query_param_relations(doc_id)
        sigs = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        params = await _mcp_client.query_params_by_doc_id(doc_id)

        if not relations:
            logger.info("BuildParamRelations: no relations, skipping")
            return {"error": None}

        # Setup NODE_PROGRESS emission for the frontend constraint detail panel.
        ctx = get_context()
        _progress_span = Span(
            span_id="progress",
            parent_span_id=ctx.current_span_id if ctx else None,
            span_type=SpanType.NODE,
            name="build_param_relations",
        )
        _emit = lambda evt, data: (
            ctx.manager.emit(evt, ctx.run_id, _progress_span, {
                "agent_id": "constraint",
                "node_id": "build_param_relations",
                **data,
            }) if ctx and ctx.manager else None
        )

        _emit(EventType.NODE_PROGRESS, {
            "message": f"已查询到 {len(relations)} 条参数关系，开始提取表达式",
            "phase": "data_ready",
            "relations_count": len(relations),
        })

        # Step 1b: Get shape dimension mappings from state (for prompt + safety net)
        mappings = state.get("implicit_params", [])
        implicit_params_text = ""
        if mappings:
            from agent.nodes.param_relation_extract.prompts import (
                format_implicit_params_context,
            )
            implicit_params_text = format_implicit_params_context(mappings)

        # Step 1c: Get platform constants and build external constant names
        platform_constants = state.get("platform_constants", [])
        external_const_names: set[str] = set()
        for pc in platform_constants:
            external_const_names.add(pc["const_name"])
        for m in mappings:
            if m.get("is_external_constant"):
                external_const_names.add(m["var_name"])
        if external_const_names:
            logger.info(
                "BuildParamRelations: external constants: %s (doc_id=%s)",
                sorted(external_const_names), doc_id,
            )

        # Step 1d: Build set of implicit param names (named dim variables)
        # Used to allow them in expression validation
        implicit_param_names: set[str] = set()
        for m in mappings:
            if not m.get("is_external_constant") and not m.get("is_constant"):
                implicit_param_names.add(m["var_name"])

        # FIX-9: Build param_optional_map from DB is_optional field (R6).
        # Used by validate_none_guard to detect missing None guards.
        param_optional_map: dict[str, bool] = {}
        for p in params or []:
            pn = p.get("param_name", "")
            if pn:
                param_optional_map[pn] = bool(p.get("is_optional", False))

        # Step 2: Build signature context (enriched with shape info)
        signatures_text = _format_signatures(sigs, params or [])
        param_shapes_text = _build_param_shapes_text(params or [])

        # Step 3: LLM batch extract expr_type + expr
        # Skip relations that already have a non-empty expr (from constraint_extract
        # Pass 1-4 regex templates or Pass 5 agent).  Only call LLM for relations
        # with empty expr.
        needs_llm: list[tuple[int, dict]] = []  # (original_index, relation)
        llm_results: list[dict[str, str] | None] = [None] * len(relations)

        for i, rel in enumerate(relations):
            obj = rel.get("relation_object", {})
            if isinstance(obj, str):
                try:
                    obj = json.loads(obj)
                except (json.JSONDecodeError, TypeError):
                    obj = {}
            existing_expr = obj.get("expr", "") if isinstance(obj, dict) else ""
            if existing_expr:
                llm_results[i] = {
                    "expr_type": obj.get("expr_type", ""),
                    "expr": existing_expr,
                    "confidence": "high",
                }
            else:
                needs_llm.append((i, rel))

        if needs_llm:
            llm_needed = [r for _, r in needs_llm]
            extracted = await _batch_extract_relation_objects(
                llm_needed, signatures_text, param_shapes_text,
                implicit_params_text=implicit_params_text,
                external_constants=external_const_names,
                implicit_param_names=implicit_param_names,
                param_optional_map=param_optional_map,
            )
            for (idx, _), result in zip(needs_llm, extracted):
                llm_results[idx] = result
            logger.info(
                "BuildParamRelations: %d/%d relations already have expr, "
                "LLM generated expr for %d",
                len(relations) - len(needs_llm), len(relations), len(needs_llm),
            )
        else:
            logger.info("BuildParamRelations: all relations already have expr, skipping LLM")

        # Aggregate validation phase results for NODE_PROGRESS (extract_done).
        valid_count = 0
        corrected_count = 0
        error_count = 0
        ast_failed = 0
        ref_failed = 0
        semantic_passed = 0
        semantic_corrected = 0
        semantic_failed = 0
        semantic_skipped = 0
        for r in llm_results:
            phases = r.get("_validation_phases") or {}
            ast = phases.get("ast_syntax") or {}
            ref = phases.get("param_refs") or {}
            sem = phases.get("semantic") or {}
            if r.get("_validation_error"):
                error_count += 1
            elif r.get("_corrected"):
                corrected_count += 1
            else:
                valid_count += 1
            if ast.get("status") == "failed":
                ast_failed += 1
            if ref.get("status") == "failed":
                ref_failed += 1
            s_status = sem.get("status")
            if s_status == "passed":
                semantic_passed += 1
            elif s_status == "corrected":
                semantic_corrected += 1
            elif s_status == "failed":
                semantic_failed += 1
            elif s_status == "skipped":
                semantic_skipped += 1

        _emit(EventType.NODE_PROGRESS, {
            "message": f"表达式提取完成: {valid_count}/{len(relations)} 有效, "
                       f"{corrected_count} 已修正, {error_count} 有校验错误",
            "phase": "extract_done",
            "relations_count": len(relations),
            "valid_count": valid_count,
            "corrected_count": corrected_count,
            "error_count": error_count,
            "ast_failed_count": ast_failed,
            "ref_failed_count": ref_failed,
            "semantic_passed_count": semantic_passed,
            "semantic_corrected_count": semantic_corrected,
            "semantic_failed_count": semantic_failed,
            "semantic_skipped_count": semantic_skipped,
        })

        # Step 4: Assemble relation_object and persist
        updates: list[dict] = []

        for rel, llm_out in zip(relations, llm_results):
            original_expr = llm_out.get("expr", "")
            final_expr = original_expr

            # Only substitute constant values (e.g. k0 -> 16), not dim vars
            if mappings and original_expr:
                substituted = _substitute_dim_vars(
                    original_expr, mappings, external_const_names,
                )
                if substituted != original_expr:
                    is_valid, _ = _validate_expr(
                        substituted, rel.get("params", []),
                        external_const_names, implicit_param_names,
                    )
                    if is_valid:
                        final_expr = substituted

            # Keep original params (including implicit/non-operator params)
            clean_params = rel.get("params", [])

            relation_object = {
                "expr_type": llm_out.get("expr_type", ""),
                "expr": final_expr,
                "relation_params": clean_params,
                "src_text": rel.get("source_citation", ""),
            }

            # Log metadata for debugging (not stored in DB)
            if llm_out.get("_validation_error"):
                logger.warning(
                    "BuildParamRelations: relation id=%s has validation error: %s",
                    rel.get("id", "?"), llm_out["_validation_error"],
                )
            if llm_out.get("_corrected"):
                logger.info(
                    "BuildParamRelations: relation id=%s expr was corrected: %s",
                    rel.get("id", "?"), llm_out.get("_correction_reason", ""),
                )
            updates.append({
                "id": rel["id"],
                "relation_object": json.dumps(relation_object, ensure_ascii=False),
            })

        result = await _mcp_client.update_param_relation_objects(doc_id, updates)
        logger.info(
            "BuildParamRelations: updated %d/%d relations (doc_id=%s)",
            result.get("updated", 0), len(updates), doc_id,
        )

        # Step 5: Group by platform
        from agent.utils.platform_utils import resolve_target_platforms

        supported_platforms = [
            p["platform_name"] for p in platforms if p.get("is_supported") == 1
        ]

        grouped: dict[str, list[dict]] = {}
        for rel, upd in zip(relations, updates):
            obj = json.loads(upd["relation_object"])
            platform_str = rel.get("platform", "")
            targets = resolve_target_platforms(platform_str, supported_platforms)
            for plat in targets:
                grouped.setdefault(plat, []).append(obj)

        logger.info(
            "BuildParamRelations: grouped into %d platforms (doc_id=%s)",
            len(grouped), doc_id,
        )

        _emit(EventType.NODE_PROGRESS, {
            "message": f"已按平台分组: {len(grouped)} 个平台, {len(relations)} 条关系",
            "phase": "complete",
            "platforms_count": len(grouped),
            "relations_count": len(relations),
        })

        # Build per-relation validation_results for the frontend
        # ExtractorAgent constraint detail panel (cs_relations / cs_check_relations).
        validation_results: list[dict] = []
        for rel, llm_out in zip(relations, llm_results):
            phases = llm_out.get("_validation_phases") or {}
            ast = phases.get("ast_syntax") or {}
            ref = phases.get("param_refs") or {}
            sem = phases.get("semantic") or {}

            # syntax_valid: only False when a phase actually failed.
            any_failed = (
                ast.get("status") == "failed"
                or ref.get("status") == "failed"
                or sem.get("status") == "failed"
            )
            syntax_valid = not any_failed

            # has_validation: any phase actually ran (vs all skipped).
            has_validation = any(
                p.get("status") in ("passed", "failed", "corrected", "recovered")
                for p in (ast, ref, sem)
            )

            validation_results.append({
                "relation_id": rel.get("id"),
                "relation_type": rel.get("relation_type", ""),
                "params": rel.get("params", []),
                "expr_type": llm_out.get("expr_type", ""),
                "expr": llm_out.get("expr", ""),
                "confidence": llm_out.get("confidence", ""),
                "syntax_valid": syntax_valid,
                "validation_error": llm_out.get("_validation_error", ""),
                "corrected": bool(llm_out.get("_corrected")),
                "correction_reason": llm_out.get("_correction_reason", ""),
                "has_validation": has_validation,
                "phase_ast_syntax": {
                    "status": ast.get("status", "skipped"),
                    "error": ast.get("error", ""),
                },
                "phase_param_refs": {
                    "status": ref.get("status", "skipped"),
                    "error": ref.get("error", ""),
                },
                "phase_semantic": {
                    "status": sem.get("status", "skipped"),
                    "reason": sem.get("reason", ""),
                },
            })

        return {
            "error": None,
            "relations_count": len(relations),
            "platforms_count": len(grouped),
            "validation_results": validation_results,
        }

    except Exception as e:
        logger.exception("BuildParamRelations failed for %s", operator_name)
        return {"error": str(e)}
