"""Tests for build_param_relations.py: three-layer expression protection.

Covers:
- Phase 0: AST syntax validation + reference validation
- Phase 2a: Few-shot example selection
- Phase 2b: Semantic verification with loop-back
- Integration: retry mechanism with mocked LLM
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.nodes.build_param_relations import (
    _select_relevant_example,
    _validate_expr,
    _validate_expr_refs,
    _validate_expr_syntax,
)


# ---------------------------------------------------------------------------
# Phase 0a: AST syntax validation
# ---------------------------------------------------------------------------


class TestValidateExprSyntax:
    def test_valid_expression(self):
        is_valid, error = _validate_expr_syntax("x.shape == y.shape")
        assert is_valid
        assert error == ""

    def test_valid_conditional(self):
        expr = (
            "(scale.shape[0] == x.shape[axis.range_value]) "
            "if len(scale.shape) == 1 else True"
        )
        is_valid, error = _validate_expr_syntax(expr)
        assert is_valid

    def test_valid_with_all(self):
        is_valid, error = _validate_expr_syntax("all(d > 0 for d in x.shape)")
        assert is_valid

    def test_empty_expression_allowed(self):
        is_valid, error = _validate_expr_syntax("")
        assert is_valid
        assert error == ""

    def test_syntax_error_implies(self):
        is_valid, error = _validate_expr_syntax("x.shape implies y.shape")
        assert not is_valid
        assert "SyntaxError" in error

    def test_syntax_error_missing_paren(self):
        is_valid, error = _validate_expr_syntax("(x.shape[0] == y.shape[0]")
        assert not is_valid

    def test_syntax_error_assignment(self):
        is_valid, error = _validate_expr_syntax("x.dtype = y.dtype")
        assert not is_valid

    def test_valid_none_comparison(self):
        is_valid, error = _validate_expr_syntax("x.shape[0] == None")
        assert is_valid

    def test_valid_boolean_expr(self):
        expr = "(x.dtype == 'FLOAT16') if (y.dtype == 'FLOAT16') else True"
        is_valid, error = _validate_expr_syntax(expr)
        assert is_valid


# ---------------------------------------------------------------------------
# Phase 0b: Reference validation
# ---------------------------------------------------------------------------


class TestValidateExprRefs:
    def test_valid_params(self):
        is_valid, error = _validate_expr_refs("x.shape == y.shape", ["x", "y"])
        assert is_valid

    def test_valid_with_builtins(self):
        is_valid, error = _validate_expr_refs("all(d > 0 for d in x.shape)", ["x"])
        assert is_valid

    def test_valid_with_len(self):
        is_valid, error = _validate_expr_refs("len(x.shape) == len(y.shape)", ["x", "y"])
        assert is_valid

    def test_empty_expression_allowed(self):
        is_valid, error = _validate_expr_refs("", ["x", "y"])
        assert is_valid

    def test_hallucinated_param(self):
        is_valid, error = _validate_expr_refs("x.shape == z.shape", ["x", "y"])
        assert not is_valid
        assert "Unknown parameter" in error

    def test_hallucinated_attribute(self):
        is_valid, error = _validate_expr_refs("len(x.shape) == len(y.dims)", ["x", "y"])
        assert not is_valid
        assert "Unknown attribute" in error

    def test_valid_dtype_attr(self):
        is_valid, error = _validate_expr_refs("x.dtype == y.dtype", ["x", "y"])
        assert is_valid

    def test_valid_format_attr(self):
        is_valid, error = _validate_expr_refs("x.format == y.format", ["x", "y"])
        assert is_valid

    def test_valid_range_value_attr(self):
        is_valid, error = _validate_expr_refs("x.range_value > 0", ["x"])
        assert is_valid

    def test_invalid_attr(self):
        is_valid, error = _validate_expr_refs("x.size == y.size", ["x", "y"])
        assert not is_valid
        assert "Unknown attribute" in error

    def test_builtin_true_false_none(self):
        is_valid, error = _validate_expr_refs("True if x.shape[0] > 0 else False", ["x"])
        assert is_valid


# ---------------------------------------------------------------------------
# Phase 0: Combined validation
# ---------------------------------------------------------------------------


class TestValidateExpr:
    def test_valid_expr(self):
        is_valid, error = _validate_expr("x.shape == y.shape", ["x", "y"])
        assert is_valid

    def test_syntax_error_first(self):
        is_valid, error = _validate_expr("x.shape implies y.shape", ["x", "y"])
        assert not is_valid
        assert "SyntaxError" in error

    def test_ref_error_second(self):
        is_valid, error = _validate_expr("x.shape == z.shape", ["x", "y"])
        assert not is_valid
        assert "Unknown parameter" in error

    def test_empty_passes(self):
        is_valid, error = _validate_expr("", ["x", "y"])
        assert is_valid


# ---------------------------------------------------------------------------
# Phase 2a: Few-shot example selection
# ---------------------------------------------------------------------------


class TestSelectRelevantExample:
    def test_implies_error(self):
        hint = _select_relevant_example("", "x.shape implies y.shape")
        assert "implies" in hint

    def test_null_error(self):
        hint = _select_relevant_example("", "x.shape[0] == null")
        assert "null" in hint

    def test_unknown_param_error(self):
        hint = _select_relevant_example("Unknown parameter", "x.shape == z.shape")
        assert "params" in hint

    def test_unknown_attr_error(self):
        hint = _select_relevant_example("Unknown attribute", "x.dims == y.dims")
        assert "shape" in hint

    def test_default_example(self):
        hint = _select_relevant_example("Some other error", "some expr")
        assert "True" in hint


# ---------------------------------------------------------------------------
# Integration: retry mechanism with mocked LLM
# ---------------------------------------------------------------------------


class TestExtractWithRetry:
    @pytest.mark.asyncio
    async def test_valid_expr_no_retry(self):
        from agent.nodes.build_param_relations import _extract_with_retry
        import asyncio

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "expr_type": "shape_equality",
            "expr": "x.shape == y.shape",
            "confidence": "high",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response

        rel = {"id": 1, "params": ["x", "y"], "relation_type": "shape",
               "description": "test", "source_citation": "test"}
        sem = asyncio.Semaphore(5)
        result = await _extract_with_retry(mock_llm, rel, "sig", sem)

        assert result["expr"] == "x.shape == y.shape"
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_syntax_error(self):
        from agent.nodes.build_param_relations import _extract_with_retry
        import asyncio

        bad_response = MagicMock()
        bad_response.content = json.dumps({
            "expr_type": "shape_equality",
            "expr": "x.shape implies y.shape",
            "confidence": "high",
        })
        good_response = MagicMock()
        good_response.content = json.dumps({
            "expr_type": "shape_equality",
            "expr": "x.shape == y.shape",
            "confidence": "high",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [bad_response, good_response]

        rel = {"id": 2, "params": ["x", "y"], "relation_type": "shape",
               "description": "test", "source_citation": "test"}
        sem = asyncio.Semaphore(5)
        result = await _extract_with_retry(mock_llm, rel, "sig", sem)

        assert result["expr"] == "x.shape == y.shape"
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_hallucinated_param(self):
        from agent.nodes.build_param_relations import _extract_with_retry
        import asyncio

        bad_response = MagicMock()
        bad_response.content = json.dumps({
            "expr_type": "shape_equality",
            "expr": "x.shape == z.shape",
            "confidence": "high",
        })
        good_response = MagicMock()
        good_response.content = json.dumps({
            "expr_type": "shape_equality",
            "expr": "x.shape == y.shape",
            "confidence": "high",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [bad_response, good_response]

        rel = {"id": 3, "params": ["x", "y"], "relation_type": "shape",
               "description": "test", "source_citation": "test"}
        sem = asyncio.Semaphore(5)
        result = await _extract_with_retry(mock_llm, rel, "sig", sem)

        assert result["expr"] == "x.shape == y.shape"
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        from agent.nodes.build_param_relations import _extract_with_retry
        import asyncio

        bad_response = MagicMock()
        bad_response.content = json.dumps({
            "expr_type": "shape_equality",
            "expr": "x.shape implies y.shape",
            "confidence": "high",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = bad_response

        rel = {"id": 4, "params": ["x", "y"], "relation_type": "shape",
               "description": "test", "source_citation": "test"}
        sem = asyncio.Semaphore(5)
        result = await _extract_with_retry(mock_llm, rel, "sig", sem)

        assert result["expr"] == ""
        assert "_validation_error" in result
        assert mock_llm.ainvoke.call_count == 3


# ---------------------------------------------------------------------------
# Integration: semantic verification
# ---------------------------------------------------------------------------


class TestVerifyAndFix:
    @pytest.mark.asyncio
    async def test_correct_expr_passes(self):
        from agent.nodes.build_param_relations import _verify_and_fix
        import asyncio

        verify_response = MagicMock()
        verify_response.content = json.dumps({
            "is_correct": True,
            "reason": "ok",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = verify_response

        rel = {"id": 5, "params": ["x", "y"]}
        expr_result = {
            "expr_type": "shape_equality",
            "expr": "x.shape == y.shape",
            "confidence": "low",
        }
        sem = asyncio.Semaphore(5)
        result = await _verify_and_fix(mock_llm, rel, expr_result, sem)

        assert result["expr"] == "x.shape == y.shape"
        assert "_corrected" not in result

    @pytest.mark.asyncio
    async def test_incorrect_expr_corrected(self):
        from agent.nodes.build_param_relations import _verify_and_fix
        import asyncio

        verify_response = MagicMock()
        verify_response.content = json.dumps({
            "is_correct": False,
            "reason": "condition direction",
            "corrected_expr": "(x.shape[0] == y.shape[0]) if axis == 0 else True",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = verify_response

        rel = {"id": 6, "params": ["x", "y", "axis"]}
        expr_result = {
            "expr_type": "shape_dependency",
            "expr": "(x.shape[0] == y.shape[0]) if axis == 0 else False",
            "confidence": "low",
        }
        sem = asyncio.Semaphore(5)
        result = await _verify_and_fix(mock_llm, rel, expr_result, sem)

        assert result["expr"] == "(x.shape[0] == y.shape[0]) if axis == 0 else True"
        assert result["_corrected"] is True

    @pytest.mark.asyncio
    async def test_corrected_expr_invalid_fallback(self):
        from agent.nodes.build_param_relations import _verify_and_fix
        import asyncio

        verify_response = MagicMock()
        verify_response.content = json.dumps({
            "is_correct": False,
            "reason": "error",
            "corrected_expr": "x.shape implies y.shape",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = verify_response

        rel = {"id": 7, "params": ["x", "y"]}
        expr_result = {
            "expr_type": "shape_equality",
            "expr": "x.shape == y.shape",
            "confidence": "low",
        }
        sem = asyncio.Semaphore(5)
        result = await _verify_and_fix(mock_llm, rel, expr_result, sem)

        assert result["expr"] == "x.shape == y.shape"
        assert "_corrected" not in result

    @pytest.mark.asyncio
    async def test_corrected_expr_hallucinated_param_fallback(self):
        from agent.nodes.build_param_relations import _verify_and_fix
        import asyncio

        verify_response = MagicMock()
        verify_response.content = json.dumps({
            "is_correct": False,
            "reason": "error",
            "corrected_expr": "x.shape == z.shape",
        })
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = verify_response

        rel = {"id": 8, "params": ["x", "y"]}
        expr_result = {
            "expr_type": "shape_equality",
            "expr": "x.shape == y.shape",
            "confidence": "low",
        }
        sem = asyncio.Semaphore(5)
        result = await _verify_and_fix(mock_llm, rel, expr_result, sem)

        assert result["expr"] == "x.shape == y.shape"
