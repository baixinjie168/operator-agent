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
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import RELATION_OBJECT_BUILD_PROMPT
from agent.core.llm import create_llm
from agent.utils.expr_validation import validate_expr as _validate_expr
from agent.utils.expr_validation import validate_expr_refs as _validate_expr_refs
from agent.utils.expr_validation import validate_expr_syntax as _validate_expr_syntax
from agent.utils.llm_common import CONCURRENCY_LIMIT, JSON_BLOCK_RE
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
    match = JSON_BLOCK_RE.search(text)
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
# Phase 0 validation is now in agent.utils.expr_validation
# (imported as _validate_expr, _validate_expr_refs, _validate_expr_syntax)
# ---------------------------------------------------------------------------


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
) -> dict[str, str]:
    """Phase 2a: Extract with enhanced retry (max 2 attempts).

    On validation failure, inject relevant Few-shot example before retrying.
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
        # Parse JSON response
        match = JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(1)
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
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
    return {
        "expr_type": expr_result.get("expr_type", ""),
        "expr": corrected,
        "_corrected": True,
        "_correction_reason": verify_result.get("reason", ""),
    }


async def _batch_extract_relation_objects(
    relations: list[dict],
    signatures_text: str,
    param_shapes_text: str,
    implicit_params_text: str = "",
    external_constants: set[str] | None = None,
    implicit_param_names: set[str] | None = None,
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
                if not is_valid:
                    logger.warning(
                        "ComplexRelationAgent expr invalid for id=%s: %s",
                        rel.get("id", "?"), error,
                    )
                    result["expr"] = ""
        else:
            # Type 1: simple param relation -> current single-shot LLM
            result = await _extract_with_retry(
                llm, rel, signatures_text, param_shapes_text, sem,
                implicit_params_text=implicit_params_text,
                external_constants=external_constants,
                implicit_param_names=implicit_param_names,
            )

            # Check if semantic verification is needed (Phase 2b)
            confidence = result.get("confidence", "high")
            if confidence == "low" and result.get("expr"):
                # Force semantic verification for low confidence
                result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)
            # For medium confidence, verification is optional (disabled by default)
            # Uncomment below to enable:
            # elif confidence == "medium" and result.get("expr"):
        #     result = await _verify_and_fix(llm, rel, result, param_shapes_text, sem)

        return result

    results = await asyncio.gather(*[_process_one(r) for r in relations])
    return list(results)


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

        # Step 2: Build signature context (enriched with shape info)
        signatures_text = _format_signatures(sigs, params or [])
        param_shapes_text = _build_param_shapes_text(params or [])

        # Step 3: LLM batch extract expr_type + expr
        llm_results = await _batch_extract_relation_objects(
            relations, signatures_text, param_shapes_text,
            implicit_params_text=implicit_params_text,
            external_constants=external_const_names,
            implicit_param_names=implicit_param_names,
        )

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

        return {"error": None}

    except Exception as e:
        logger.exception("BuildParamRelations failed for %s", operator_name)
        return {"error": str(e)}
