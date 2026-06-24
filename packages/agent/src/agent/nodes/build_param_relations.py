"""BuildParamRelations node: enrich param_relations with expr_type/expr and group by platform.

Implements three-layer protection for expression generation accuracy:
- Phase 0: Deterministic validation (AST syntax + reference checks)
- Phase 1: Prompt enhancement (Few-shot examples + confidence scoring)
- Phase 2: Failure remediation (enhanced retry + semantic verification)
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import RELATION_OBJECT_BUILD_PROMPT
from agent.runtime.context import get_context
from agent.runtime.events import EventType, Span, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_CONCURRENCY_LIMIT = 5

# Phase 0: Allowed attributes and builtin names for reference validation
_ALLOWED_ATTRS = {"shape", "dtype", "format", "range_value"}
_BUILTIN_NAMES = {
    "True", "False", "None", "len", "range",
    "all", "any", "int", "float", "str", "bool", "set",
}

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
6. expr_type 是否与描述的关系类型匹配？
7. 维度索引是否正确？（对照"参数 shape 信息"，确认 shape[i] 引用的确实是描述中所指的维度；当参数有多种 shape 形式时，应使用负索引 shape[-N]）

## 输出
严格按以下 JSON 返回：
{{"is_correct": true, "reason": "表达式正确"}}
或
{{"is_correct": false, "reason": "错误原因", "corrected_expr": "修正后的表达式"}}
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
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "expr_type": data.get("expr_type", ""),
                "expr": data.get("expr", ""),
                "confidence": data.get("confidence", "high"),
                "uncertainty_reason": data.get("uncertainty_reason", ""),
            }
    except json.JSONDecodeError:
        pass

    # Regex fallback
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict):
                return {
                    "expr_type": data.get("expr_type", ""),
                    "expr": data.get("expr", ""),
                    "confidence": data.get("confidence", "high"),
                    "uncertainty_reason": data.get("uncertainty_reason", ""),
                }
        except json.JSONDecodeError:
            pass

    logger.warning("BuildParamRelations: failed to parse LLM response: %s", text[:200])
    return {"expr_type": "", "expr": "", "confidence": "high", "uncertainty_reason": ""}


# ---------------------------------------------------------------------------
# Phase 0: Deterministic validation (zero LLM cost)
# ---------------------------------------------------------------------------


def _validate_expr_syntax(expr: str) -> tuple[bool, str]:
    """Phase 0a: Validate expr is a legal Python expression.

    Returns:
        (is_valid, error_message)
    """
    if not expr:
        logger.error("[ValidateExprSyntax] PASS (empty expr, skipped)")
        return True, ""  # Empty expression is allowed
    try:
        ast.parse(expr, mode="eval")
        logger.error("[ValidateExprSyntax] PASS — expr='%s' — AST语法校验通过", expr)
        return True, ""
    except SyntaxError as e:
        err = f"SyntaxError at line {e.lineno}: {e.msg}"
        logger.error("[ValidateExprSyntax] FAIL — expr='%s' — %s", expr, err)
        return False, err


def _validate_expr_refs(expr: str, params: list[str]) -> tuple[bool, str]:
    """Phase 0b: Validate parameter names and attributes in expr.

    Checks:
    1. All Name nodes must be in params or Python builtins
    2. All Attribute nodes must be in _ALLOWED_ATTRS
    3. Comprehension variables (e.g., 'd' in 'all(d > 0 for d in x.shape)') are allowed
    """
    if not expr:
        logger.error("[ValidateExprRefs] PASS (empty expr, skipped) params=%s", params)
        return True, ""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        logger.error("[ValidateExprRefs] FAIL — expr='%s' — Invalid syntax (AST parse failed)", expr)
        return False, "Invalid syntax"

    param_set = set(params)

    # Collect all comprehension variables (bound by for loops in comprehensions)
    comprehension_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            for generator in node.generators:
                if isinstance(generator.target, ast.Name):
                    comprehension_vars.add(generator.target.id)
                elif isinstance(generator.target, ast.Tuple):
                    for elt in generator.target.elts:
                        if isinstance(elt, ast.Name):
                            comprehension_vars.add(elt.id)

    # Collect all referenced names and attributes for logging
    ref_names: list[str] = []
    ref_attrs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            ref_names.append(node.id)
        if isinstance(node, ast.Attribute):
            ref_attrs.append(node.attr)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if (
                node.id not in param_set
                and node.id not in _BUILTIN_NAMES
                and node.id not in comprehension_vars
            ):
                err = f"Unknown parameter: '{node.id}'"
                logger.error("[ValidateExprRefs] FAIL — expr='%s' params=%s — %s | ref_names=%s ref_attrs=%s comprehension_vars=%s",
                             expr, params, err, ref_names, ref_attrs, comprehension_vars)
                return False, err
        if isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_ATTRS:
                err = f"Unknown attribute: '.{node.attr}'"
                logger.error("[ValidateExprRefs] FAIL — expr='%s' params=%s — %s | ref_names=%s ref_attrs=%s allowed_attrs=%s",
                             expr, params, err, ref_names, ref_attrs, _ALLOWED_ATTRS)
                return False, err
    logger.error("[ValidateExprRefs] PASS — expr='%s' params=%s — 参数名和属性均合法 | ref_names=%s ref_attrs=%s comprehension_vars=%s",
                 expr, params, ref_names, ref_attrs, comprehension_vars)
    return True, ""


def _validate_expr(expr: str, params: list[str]) -> tuple[bool, str]:
    """Phase 0: Comprehensive validation (syntax + references)."""
    is_valid, error = _validate_expr_syntax(expr)
    if not is_valid:
        return False, error
    is_valid, error = _validate_expr_refs(expr, params)
    if not is_valid:
        return False, error
    return True, ""


# ---------------------------------------------------------------------------
# Phase 2a: Enhanced retry with Few-shot examples
# ---------------------------------------------------------------------------


def _select_relevant_example(error: str, expr: str) -> str:
    """Select the most relevant Few-shot example based on error type."""
    error_lower = error.lower()
    expr_lower = expr.lower()

    if "implies" in expr_lower:
        ex = FEW_SHOT_EXAMPLES["syntax_implies"]
    elif "null" in expr_lower:
        ex = FEW_SHOT_EXAMPLES["syntax_null"]
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
) -> dict[str, str]:
    """Extract with Few-shot example hint for retry."""
    prompt = RELATION_OBJECT_BUILD_PROMPT.format(
        signatures_text=signatures_text,
        param_shapes_text=param_shapes_text,
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
) -> dict[str, str]:
    """Phase 2a: Extract with enhanced retry (max 2 attempts).

    On validation failure, inject relevant Few-shot example before retrying.

    Returns a dict that also tracks per-phase validation outcomes in
    `_validation_phases` so the frontend can show the AST syntax check,
    parameter reference check, and (downstream) semantic check results:

    - `_validation_phases.ast_syntax`:
        {status: "passed"} when ast.parse() succeeds,
        {status: "failed", error: "..."} when SyntaxError,
        {status: "skipped"} when expr is empty.
    - `_validation_phases.param_refs`:
        {status: "passed"} when all Name/Attribute references are valid,
        {status: "failed", error: "..."} when a reference is invalid,
        {status: "skipped"} when ast_syntax failed or expr is empty.
    """
    last_error = ""
    last_expr = ""

    # 默认 phase 结果：空表达式 → 全部跳过
    def _make_phases(ast_status="skipped", ast_err="",
                      ref_status="skipped", ref_err=""):
        return {
            "ast_syntax": {"status": ast_status, "error": ast_err},
            "param_refs": {"status": ref_status, "error": ref_err},
        }

    for attempt in range(settings.expr_max_retries + 1):
        async with sem:
            try:
                if attempt == 0:
                    prompt = RELATION_OBJECT_BUILD_PROMPT.format(
                        signatures_text=signatures_text,
                        param_shapes_text=param_shapes_text,
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
                    )

                expr = result.get("expr", "")
                params = rel.get("params", [])

                # Phase 0 校验：分两步执行，并分别记录 phase 结果
                if not expr:
                    # 空表达式：所有 phase 标为 skipped
                    result["_validation_phases"] = _make_phases()
                    return result

                # Phase 0a: AST 语法校验
                ast_ok, ast_err = _validate_expr_syntax(expr)
                if not ast_ok:
                    logger.warning(
                        "BuildParamRelations: AST 语法校验失败 (attempt %d/%d) "
                        "for relation id=%s: %s",
                        attempt + 1, settings.expr_max_retries + 1, rel.get("id", "?"), ast_err,
                    )
                    last_error = ast_err
                    last_expr = expr
                    if attempt < settings.expr_max_retries:
                        continue
                    # 重试耗尽，记录失败并返回
                    result["_validation_phases"] = _make_phases(
                        ast_status="failed", ast_err=ast_err,
                    )
                    return {
                        "expr_type": result.get("expr_type", ""),
                        "expr": "",
                        "_validation_error": ast_err,
                        "_validation_phases": result["_validation_phases"],
                    }

                # Phase 0b: 参数引用校验
                ref_ok, ref_err = _validate_expr_refs(expr, params)
                if not ref_ok:
                    logger.warning(
                        "BuildParamRelations: 参数引用校验失败 (attempt %d/%d) "
                        "for relation id=%s: %s",
                        attempt + 1, settings.expr_max_retries + 1, rel.get("id", "?"), ref_err,
                    )
                    last_error = ref_err
                    last_expr = expr
                    if attempt < settings.expr_max_retries:
                        continue
                    # 重试耗尽，记录失败并返回
                    result["_validation_phases"] = _make_phases(
                        ast_status="passed",
                        ref_status="failed", ref_err=ref_err,
                    )
                    return {
                        "expr_type": result.get("expr_type", ""),
                        "expr": "",
                        "_validation_error": ref_err,
                        "_validation_phases": result["_validation_phases"],
                    }

                # 两步都通过
                result["_validation_phases"] = _make_phases(
                    ast_status="passed",
                    ref_status="passed",
                )
                return result

            except Exception:
                logger.warning(
                    "BuildParamRelations: LLM failed for relation id=%s",
                    rel.get("id", "?"),
                )
                return {
                    "expr_type": "",
                    "expr": "",
                    "_validation_phases": _make_phases(),
                }

    # 理论上不会走到这里，兜底返回
    return {"expr_type": "", "expr": "", "_validation_phases": _make_phases()}


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
    rel_id = rel.get("id", "?")
    logger.error("[Phase2b][VerifyLLM] ===== START relation_id=%s =====", rel_id)
    logger.error("  input_expr_type=%s", expr_result.get('expr_type', ''))
    logger.error("  input_expr=%s", expr_result.get('expr', ''))
    logger.error("  input_description=%s", (rel.get('description', '') or '')[:200])
    logger.error("  input_source_citation=%s", (rel.get('source_citation', '') or '')[:200])
    logger.error("  input_params=%s", json.dumps(rel.get('params', []), ensure_ascii=False))

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
        logger.error("[Phase2b][VerifyLLM] relation_id=%s raw_response=%s", rel_id, text[:500])

        # Parse JSON response
        match = _JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(1)
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                logger.error("[Phase2b][VerifyLLM] relation_id=%s parsed_result=%s", rel_id, json.dumps(data, ensure_ascii=False))
                logger.error("  is_correct=%s reason=%s", data.get('is_correct'), data.get('reason', ''))
                logger.error("  corrected_expr=%s", data.get('corrected_expr', ''))
                logger.error("[Phase2b][VerifyLLM] ===== END relation_id=%s =====", rel_id)
                return data
        except json.JSONDecodeError:
            pass
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                if isinstance(data, dict):
                    logger.error("[Phase2b][VerifyLLM] relation_id=%s parsed_result(regex)=%s", rel_id, json.dumps(data, ensure_ascii=False))
                    logger.error("  is_correct=%s reason=%s", data.get('is_correct'), data.get('reason', ''))
                    logger.error("  corrected_expr=%s", data.get('corrected_expr', ''))
                    logger.error("[Phase2b][VerifyLLM] ===== END relation_id=%s =====", rel_id)
                    return data
            except json.JSONDecodeError:
                pass
        logger.warning("BuildParamRelations: failed to parse verify response: %s", text[:200])
        logger.error("[Phase2b][VerifyLLM] relation_id=%s PARSE_FAILED, accepting original", rel_id)
        logger.error("[Phase2b][VerifyLLM] ===== END relation_id=%s =====", rel_id)
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

    The returned dict always carries `_validation_phases.semantic` with:
    - {status: "passed", reason: "..."} when the LLM confirmed correctness
    - {status: "corrected", reason: "..."} when an expr was corrected and
      the corrected version passed Phase 0 loop-back
    - {status: "failed", reason: "..."} when verification found an issue
      but the correction was missing or invalid (original kept as-is)
    - {status: "skipped"} when the LLM call itself errored out
    """
    rel_id = rel.get("id", "?")
    logger.error("[Phase2b][VerifyAndFix] ===== START relation_id=%s =====", rel_id)
    logger.error("  original_expr=%s", expr_result.get('expr', ''))
    logger.error("  original_expr_type=%s", expr_result.get('expr_type', ''))

    # 透传上游 phase 结果
    prev_phases = dict(expr_result.get("_validation_phases") or {})

    try:
        verify_result = await _call_verify_llm(llm, rel, expr_result, param_shapes_text, sem)
    except Exception:
        logger.warning(
            "BuildParamRelations: semantic verification failed for relation id=%s",
            rel.get("id", "?"),
        )
        logger.error("[Phase2b][VerifyAndFix] relation_id=%s LLM_CALL_FAILED, accepting original", rel_id)
        logger.error("[Phase2b][VerifyAndFix] ===== END relation_id=%s =====", rel_id)
        prev_phases["semantic"] = {
            "status": "skipped",
            "reason": "语义校验 LLM 调用失败，沿用原表达式",
        }
        expr_result["_validation_phases"] = prev_phases
        return expr_result  # Accept original on verification failure

    is_correct = verify_result.get("is_correct", True)
    reason = verify_result.get("reason", "")
    corrected = verify_result.get("corrected_expr", "")

    logger.error("[Phase2b][VerifyAndFix] relation_id=%s verify_result:", rel_id)
    logger.error("  is_correct=%s", is_correct)
    logger.error("  reason=%s", reason)
    logger.error("  corrected_expr=%s", corrected)

    if is_correct:
        logger.error("[Phase2b][VerifyAndFix] relation_id=%s → ACCEPT ORIGINAL (semantically correct)", rel_id)
        logger.error("[Phase2b][VerifyAndFix] ===== END relation_id=%s =====", rel_id)
        prev_phases["semantic"] = {"status": "passed", "reason": reason or "语义校验通过"}
        expr_result["_validation_phases"] = prev_phases
        return expr_result

    # Verification failed, use corrected_expr
    if not corrected:
        logger.warning(
            "BuildParamRelations: verification failed but no corrected_expr "
            "for relation id=%s",
            rel.get("id", "?"),
        )
        logger.error("[Phase2b][VerifyAndFix] relation_id=%s → ACCEPT ORIGINAL (no corrected_expr provided)", rel_id)
        logger.error("[Phase2b][VerifyAndFix] ===== END relation_id=%s =====", rel_id)
        prev_phases["semantic"] = {
            "status": "failed",
            "reason": reason or "语义校验未通过，且未提供修正表达式",
        }
        expr_result["_validation_phases"] = prev_phases
        return expr_result  # Accept original if no correction

    # Critical: corrected_expr must pass Phase 0 validation
    logger.error("[Phase2b][VerifyAndFix] relation_id=%s LOOP-BACK: validating corrected_expr with Phase 0...", rel_id)
    is_valid, error = _validate_expr(corrected, rel.get("params", []))
    if not is_valid:
        logger.warning(
            "BuildParamRelations: corrected_expr failed validation for "
            "relation id=%s: %s — accepting original",
            rel.get("id", "?"), error,
        )
        logger.error("[Phase2b][VerifyAndFix] relation_id=%s → ACCEPT ORIGINAL (corrected_expr failed Phase 0: %s)", rel_id, error)
        logger.error("[Phase2b][VerifyAndFix] ===== END relation_id=%s =====", rel_id)
        prev_phases["semantic"] = {
            "status": "failed",
            "reason": f"{reason or '语义校验未通过'}；修正表达式未通过 AST 语法/参数引用校验：{error}",
        }
        expr_result["_validation_phases"] = prev_phases
        return expr_result  # Accept original if correction is invalid

    logger.info(
        "BuildParamRelations: corrected expr for relation id=%s: %s",
        rel.get("id", "?"), verify_result.get("reason", ""),
    )
    logger.error("[Phase2b][VerifyAndFix] relation_id=%s → ACCEPT CORRECTED", rel_id)
    logger.error("  original_expr=%s", expr_result.get('expr', ''))
    logger.error("  corrected_expr=%s", corrected)
    logger.error("  correction_reason=%s", reason)
    logger.error("  corrected_expr passed Phase 0 validation")
    logger.error("[Phase2b][VerifyAndFix] ===== END relation_id=%s =====", rel_id)
    prev_phases["semantic"] = {"status": "corrected", "reason": reason or "语义校验后修正"}
    new_result = {
        "expr_type": expr_result.get("expr_type", ""),
        "expr": corrected,
        "_corrected": True,
        "_correction_reason": verify_result.get("reason", ""),
        "_validation_phases": prev_phases,
    }
    return new_result


async def _batch_extract_relation_objects(
    relations: list[dict],
    signatures_text: str,
    param_shapes_text: str,
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

    sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

    try:
        llm = ChatOpenAI(
            api_key=settings.active_api_key,
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.1,
        )
    except Exception:
        logger.exception("BuildParamRelations: failed to create LLM")
        return [{"expr_type": "", "expr": ""}] * len(relations)

    async def _process_one(rel: dict) -> dict[str, str]:
        # Phase 1 + 2a: Generate with enhanced retry
        result = await _extract_with_retry(llm, rel, signatures_text, param_shapes_text, sem)

        # Check if semantic verification is needed (Phase 2b)
        confidence = result.get("confidence", "high")
        rel_id = rel.get("id", "?")

        # 初始化 semantic phase 默认状态
        prev_phases = dict(result.get("_validation_phases") or {})

        # Phase 2b: Semantic verification
        # Trigger conditions:
        # 1. confidence == "low" and has expr (original logic)
        # 2. settings.force_phase2b == True (for analysis/debugging)
        force_phase2b = getattr(settings, "force_phase2b", False)

        if confidence == "low" and result.get("expr"):
            # Force semantic verification for low confidence
            logger.error("[Phase2b] relation_id=%s — confidence=low, triggering semantic verification", rel_id)
            result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)
        elif force_phase2b and result.get("expr"):
            # Force Phase2b for all relations (for analysis)
            logger.error("[Phase2b] relation_id=%s — force_phase2b=True, confidence=%s, triggering semantic verification", rel_id, confidence)
            result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)
        else:
            # Phase2b skipped
            logger.error("[Phase2b] relation_id=%s — SKIPPED (confidence=%s, force_phase2b=%s)", rel_id, confidence, force_phase2b)
            prev_phases["semantic"] = {
                "status": "skipped",
                "reason": f"confidence={confidence} 且未强制开启，跳过语义校验",
            }
            result["_validation_phases"] = prev_phases
        # For medium confidence, verification is optional (disabled by default)
        # Uncomment below to enable:
        # elif confidence == "medium" and result.get("expr"):
        #     result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)

        return result

    results = await asyncio.gather(*[_process_one(r) for r in relations])
    return list(results)


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
    print(f"[Backend][BuildParamRelations] state_input:", repr({"doc_id": doc_id, "operator_name": operator_name}))

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

        ctx = get_context()
        _progress_span = Span(span_id="progress", parent_span_id=ctx.current_span_id if ctx else None, span_type=SpanType.NODE, name="build_param_relations")
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

        # Step 2: Build signature context (enriched with shape info)
        signatures_text = _format_signatures(sigs, params or [])
        param_shapes_text = _build_param_shapes_text(params or [])

        # Step 3: LLM batch extract expr_type + expr (three-layer protection)
        llm_results = await _batch_extract_relation_objects(
            relations, signatures_text, param_shapes_text,
        )

        # Compute validation stats for progress reporting
        valid_count = sum(1 for r in llm_results if r.get("expr"))
        corrected_count = sum(1 for r in llm_results if r.get("_corrected"))
        error_count = sum(1 for r in llm_results if r.get("_validation_error"))

        # 三段式校验的分段计数（用于进度事件 + 前端展示）
        ast_failed = sum(
            1 for r in llm_results
            if (r.get("_validation_phases") or {}).get("ast_syntax", {}).get("status") == "failed"
        )
        ref_failed = sum(
            1 for r in llm_results
            if (r.get("_validation_phases") or {}).get("param_refs", {}).get("status") == "failed"
        )
        semantic_passed = sum(
            1 for r in llm_results
            if (r.get("_validation_phases") or {}).get("semantic", {}).get("status") == "passed"
        )
        semantic_corrected = sum(
            1 for r in llm_results
            if (r.get("_validation_phases") or {}).get("semantic", {}).get("status") == "corrected"
        )
        semantic_failed = sum(
            1 for r in llm_results
            if (r.get("_validation_phases") or {}).get("semantic", {}).get("status") == "failed"
        )
        semantic_skipped = sum(
            1 for r in llm_results
            if (r.get("_validation_phases") or {}).get("semantic", {}).get("status") == "skipped"
        )

        # LLM 提取结果总览（汇总，不展开每条）
        print(f"[Backend][BuildParamRelations] LLM extract: total={len(relations)} valid={valid_count} corrected={corrected_count} errors={error_count}")

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
            relation_object = {
                "expr_type": llm_out.get("expr_type", ""),
                "expr": llm_out.get("expr", ""),
                "relation_params": rel.get("params", []),
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

        # 关系对象已组装并持久化（详细 dump 已省略）

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

        # 平台分组已完成（详细 dump 已省略）

        _emit(EventType.NODE_PROGRESS, {
            "message": f"已按平台分组: {len(grouped)} 个平台, {len(relations)} 条关系",
            "phase": "complete",
            "platforms_count": len(grouped),
            "relations_count": len(relations),
        })

        validation_results = []
        for rel, llm_out in zip(relations, llm_results):
            phases = llm_out.get("_validation_phases") or {}
            # 兼容旧数据：当 _validation_phases 不存在时（旧版本或空 expr 路径），
            # 退化为单条 validation_error；前端按需展示。
            ast = phases.get("ast_syntax") or {}
            ref = phases.get("param_refs") or {}
            sem = phases.get("semantic") or {}

            # syntax_valid 重写：只看 phase 是否有真实失败
            #   - 任一 phase 是 "failed" → False
            #   - 全部 phase 都是 "skipped"（如 expr 为空，无可校验内容）→ True（不当作错误）
            #   - 其它情况（含 passed / corrected）→ True
            any_failed = (
                ast.get("status") == "failed"
                or ref.get("status") == "failed"
                or sem.get("status") == "failed"
            )
            syntax_valid = not any_failed

            # has_validation：是否真的有 phase 实际跑过（非全 skipped）
            # 前端据此区分 "通过" 与 "跳过"
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
                # 兼容字段：syntax_valid 仅在 phase 真失败时为 False
                "syntax_valid": syntax_valid,
                "validation_error": llm_out.get("_validation_error", ""),
                "corrected": bool(llm_out.get("_corrected")),
                "correction_reason": llm_out.get("_correction_reason", ""),
                # 前端用 has_validation 区分"通过"和"跳过"
                "has_validation": has_validation,
                # 三段式校验结果（Phase 0a / 0b / 2b），供前端按段渲染
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
