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
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT, JSON_BLOCK_RE

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# Phase 0: Allowed attributes and builtin names for reference validation
_ALLOWED_ATTRS = {"shape", "dtype", "format", "range_value"}
_BUILTIN_NAMES = {
    "True", "False", "None", "len", "range",
    "all", "any", "int", "float", "str", "bool", "set",
    "min", "max", "list",
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
# Phase 0: Deterministic validation (zero LLM cost)
# ---------------------------------------------------------------------------


def _validate_expr_syntax(expr: str) -> tuple[bool, str]:
    """Phase 0a: Validate expr is a legal Python expression.

    Returns:
        (is_valid, error_message)
    """
    if not expr:
        return True, ""  # Empty expression is allowed
    try:
        ast.parse(expr, mode="eval")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _validate_expr_refs(
    expr: str,
    params: list[str],
    external_constants: set[str] | None = None,
) -> tuple[bool, str]:
    """Phase 0b: Validate parameter names and attributes in expr.

    Checks:
    1. All Name nodes must be in params, Python builtins, or external_constants
    2. All Attribute nodes must be in _ALLOWED_ATTRS
    3. Comprehension variables (e.g., 'd' in 'all(d > 0 for d in x.shape)') are allowed
    """
    if not expr:
        return True, ""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False, "Invalid syntax"

    param_set = set(params)
    ext_set = external_constants or set()

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

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if (
                node.id not in param_set
                and node.id not in _BUILTIN_NAMES
                and node.id not in comprehension_vars
                and node.id not in ext_set
            ):
                return False, f"Unknown parameter: '{node.id}'"
        if isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_ATTRS:
                return False, f"Unknown attribute: '.{node.attr}'"
    return True, ""


def _validate_expr(
    expr: str,
    params: list[str],
    external_constants: set[str] | None = None,
) -> tuple[bool, str]:
    """Phase 0: Comprehensive validation (syntax + references)."""
    is_valid, error = _validate_expr_syntax(expr)
    if not is_valid:
        return False, error
    is_valid, error = _validate_expr_refs(expr, params, external_constants)
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
    shape_dim_mappings_text: str = "",
) -> dict[str, str]:
    """Extract with Few-shot example hint for retry."""
    prompt = RELATION_OBJECT_BUILD_PROMPT.format(
        signatures_text=signatures_text,
        param_shapes_text=param_shapes_text,
        shape_dim_mappings_text=shape_dim_mappings_text,
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
    shape_dim_mappings_text: str = "",
    external_constants: set[str] | None = None,
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
                        shape_dim_mappings_text=shape_dim_mappings_text,
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
                        shape_dim_mappings_text=shape_dim_mappings_text,
                    )

                expr = result.get("expr", "")
                params = rel.get("params", [])

                # Phase 0 validation
                is_valid, error = _validate_expr(
                    expr, params, external_constants,
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
    shape_dim_mappings_text: str = "",
    external_constants: set[str] | None = None,
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
        # Phase 1 + 2a: Generate with enhanced retry
        result = await _extract_with_retry(
            llm, rel, signatures_text, param_shapes_text, sem,
            shape_dim_mappings_text=shape_dim_mappings_text,
            external_constants=external_constants,
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
# Phase 5: Safety net — dim var substitution with multi-level fallback
# ---------------------------------------------------------------------------


def _substitute_dim_vars(
    expr: str,
    mappings: list[dict],
    external_constants: set[str] | None = None,
) -> str:
    """Replace dim_var.range_value (and bare dim_var) with tensor.shape[i].

    Level 1 safety net: catches cases where the LLM still uses old notation
    despite updated prompts. For variables mapping to multiple tensors,
    uses the first mapping found.

    External constants are skipped — they should remain as-is in the expr.
    """
    if not expr or not mappings:
        return expr

    ext_set = external_constants or set()
    seen_vars: set[str] = set()
    substitutions: dict[str, str] = {}

    for m in mappings:
        var = m["var_name"]
        if var in seen_vars:
            continue
        # Skip external constants — don't substitute them
        if m.get("is_external_constant") or var in ext_set:
            continue
        seen_vars.add(var)

        if m.get("is_constant"):
            val = str(m.get("constant_value", 0))
            substitutions[f"{var}.range_value"] = val
            substitutions[var] = val
        else:
            replacement = f'{m["tensor_param"]}.shape[{m["dim_index"]}]'
            substitutions[f"{var}.range_value"] = replacement
            substitutions[var] = replacement

    # Apply substitutions (longest key first to avoid partial matches)
    for old, new in sorted(substitutions.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(old, new)

    return expr


def _try_alternate_mappings(
    expr: str,
    mappings: list[dict],
    rel: dict,
    external_constants: set[str] | None = None,
) -> str | None:
    """Try different tensor mapping combinations for ambiguous variables.

    Level 2 safety net: when a variable maps to multiple tensors
    (e.g. b → self.shape[0] | mat2.shape[0]), try each combination
    and return the first one that passes validation.
    """
    from itertools import product as iter_product

    # Collect all replacement options per variable
    var_options: dict[str, set[str]] = {}
    for m in mappings:
        var = m["var_name"]
        if m.get("is_constant"):
            replacement = str(m.get("constant_value", 0))
        else:
            replacement = f'{m["tensor_param"]}.shape[{m["dim_index"]}]'
        var_options.setdefault(var, set()).add(replacement)

    # Only process variables that appear in expr and have multiple options
    ambiguous: dict[str, list[str]] = {}
    for var, opts in var_options.items():
        if len(opts) > 1 and (f"{var}.range_value" in expr or f" {var}" in expr):
            ambiguous[var] = list(opts)

    if not ambiguous:
        return None

    # Try all combinations (typically very few: 2-4 combinations)
    for combo in iter_product(*ambiguous.values()):
        candidate = expr
        for var, replacement in zip(ambiguous.keys(), combo):
            candidate = candidate.replace(f"{var}.range_value", replacement)
            # Also replace bare var name (but not as substring of other words)
            candidate = re.sub(
                r"\b" + re.escape(var) + r"\b", replacement, candidate
            )
        is_valid, _ = _validate_expr(
            candidate, rel.get("params", []), external_constants,
        )
        if is_valid:
            return candidate

    return None


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

        # Step 1b: Get shape dimension mappings from state (for prompt + safety net)
        mappings = state.get("shape_dim_mappings", [])
        shape_dim_mappings_text = ""
        if mappings:
            from agent.nodes.param_relation_extract.prompts import (
                format_shape_dim_mappings_context,
            )
            shape_dim_mappings_text = format_shape_dim_mappings_context(mappings)

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

        # Step 1d: Build set of valid function signature parameter names
        # Used to filter out named dimension variables from relation_params
        from agent.nodes.param_relation_extract.shape_dim_mapping_extract import (
            _camel_to_snake,
        )
        sig_param_names: set[str] = set()
        for sig in sigs:
            for p in sig.get("parameters", []):
                name = p.get("name", "")
                if name:
                    sig_param_names.add(name)
                    sig_param_names.add(_camel_to_snake(name))

        # Step 2: Build signature context (enriched with shape info)
        signatures_text = _format_signatures(sigs, params or [])
        param_shapes_text = _build_param_shapes_text(params or [])

        # Step 3: LLM batch extract expr_type + expr
        llm_results = await _batch_extract_relation_objects(
            relations, signatures_text, param_shapes_text,
            shape_dim_mappings_text=shape_dim_mappings_text,
            external_constants=external_const_names,
        )

        # Step 4: Assemble relation_object and persist
        # Phase 5 safety net: substitute dim vars in expressions
        updates: list[dict] = []
        substitution_stats = {"total": 0, "level1": 0, "level2": 0, "level4": 0}

        for rel, llm_out in zip(relations, llm_results):
            original_expr = llm_out.get("expr", "")
            final_expr = original_expr
            sub_level = 0
            sub_error = ""

            if mappings and original_expr:
                substitution_stats["total"] += 1

                # Level 1: direct substitution + validation
                substituted = _substitute_dim_vars(
                    original_expr, mappings, external_const_names,
                )
                is_valid, sub_error = _validate_expr(
                    substituted, rel.get("params", []), external_const_names,
                )

                if is_valid and substituted != original_expr:
                    sub_level = 1
                    substitution_stats["level1"] += 1
                    final_expr = substituted
                elif not is_valid:
                    # Level 2: try alternate tensor mappings
                    alt_expr = _try_alternate_mappings(
                        original_expr, mappings, rel, external_const_names,
                    )
                    if alt_expr:
                        is_valid2, sub_error = _validate_expr(
                            alt_expr, rel.get("params", []),
                            external_const_names,
                        )
                        if is_valid2:
                            sub_level = 2
                            substitution_stats["level2"] += 1
                            final_expr = alt_expr

                if sub_level == 0 and original_expr != final_expr:
                    # Level 4: fall back to original expr (already validated in Phase 2a)
                    sub_level = 4
                    substitution_stats["level4"] += 1
                    final_expr = original_expr
                    logger.warning(
                        "BuildParamRelations: substitution Level 4 fallback "
                        "for relation id=%s: %s",
                        rel.get("id", "?"), sub_error,
                    )

            # Step A: Clean relation_params — only keep function signature
            # params and external constants (filter out named dim variables
            # like BS, H, N that the LLM incorrectly included)
            clean_params = [
                p for p in rel.get("params", [])
                if p in sig_param_names or p in external_const_names
            ]

            # Step B: Detect unsubstituted named variables in expr
            # (e.g. LLM used a variable not in shape_dim_mappings)
            if final_expr:
                remaining = re.findall(
                    r'\b([A-Za-z]\w*)\.range_value', final_expr,
                )
                unsubstituted = [
                    v for v in remaining
                    if v not in external_const_names
                    and v not in sig_param_names
                ]
                if unsubstituted:
                    logger.warning(
                        "BuildParamRelations: unsubstituted named vars %s "
                        "in expr '%s' for relation id=%s — clearing expr",
                        unsubstituted, final_expr, rel.get("id", "?"),
                    )
                    final_expr = ""

            relation_object = {
                "expr_type": llm_out.get("expr_type", ""),
                "expr": final_expr,
                "relation_params": clean_params,
                "src_text": rel.get("source_citation", ""),
            }

            # Log substitution metadata for observability (not stored in DB)
            if mappings and original_expr and sub_level > 0:
                logger.debug(
                    "BuildParamRelations: substitution level=%d fallback=%s "
                    "for relation id=%s: original='%s' final='%s'%s",
                    sub_level, sub_level == 4, rel.get("id", "?"),
                    original_expr, final_expr,
                    f" error='{sub_error}'" if sub_error else "",
                )

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

        # Log substitution stats for observability
        if substitution_stats["total"] > 0:
            logger.info(
                "BuildParamRelations: dim var substitution stats: "
                "total=%d, level1=%d, level2=%d, level4_fallback=%d (doc_id=%s)",
                substitution_stats["total"],
                substitution_stats["level1"],
                substitution_stats["level2"],
                substitution_stats["level4"],
                doc_id,
            )

        return {"error": None}

    except Exception as e:
        logger.exception("BuildParamRelations failed for %s", operator_name)
        return {"error": str(e)}
