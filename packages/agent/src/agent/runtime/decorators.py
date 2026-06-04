"""@traced_node decorator — wraps LangGraph nodes with automatic span + event emission.

Business code stays clean: no SSE, no events, no span management.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any

from agent.runtime.context import RuntimeContext, get_context, set_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)

# Agent IDs for backward-compatible SSE events
_AGENT_MAP: dict[str, str] = {
    "init_doc": "doc",
    "parse_params": "doc",
    "product_support": "doc",
    "function_explanation_extract": "doc",
    "function_signature_extract": "doc",
    "src_content_extract": "doc",
    "param_desc_extract": "doc",
    "shape_extract": "doc",
    "dtype_extract": "doc",
    "optional_extract": "doc",
    "dformat_extract": "doc",
    "param_attr_extract": "doc",
    "array_length_extract": "doc",
    "allowed_range_extract": "doc",
    "determinism_extract": "doc",
    "return_code_extract": "doc",
    "dtype_combo_extract": "doc",
    "param_relation_extract": "doc",
}


def traced_node(node_id: str):
    """Decorator: wraps a LangGraph node function with automatic span + event lifecycle.

    Usage:
        @traced_node("init_doc")
        async def init_doc_node(state: PipelineState) -> dict: ...
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state, config=None):
            ctx = get_context()
            if ctx is None:
                return await fn(state, config)

            run = ctx.manager.get_run(ctx.run_id)
            if not run:
                return await fn(state, config)

            agent_id = _AGENT_MAP.get(node_id, "doc")
            span = ctx.manager.open_span(
                run_id=ctx.run_id,
                parent_span_id=ctx.current_span_id,
                span_type=SpanType.NODE,
                name=node_id,
            )

            ctx.manager.emit(EventType.NODE_START, ctx.run_id, span, {
                "agent_id": agent_id,
                "node_id": node_id,
                "message": f"{_node_label(node_id)} 开始...",
                "step_index": 0,
                "progress_pct": 0,
                "progress_text": "开始",
            })

            try:
                node_ctx = RuntimeContext(ctx.run_id, ctx.manager)
                node_ctx.trace_id = ctx.trace_id
                node_ctx.current_span_id = span.span_id
                node_ctx.current_node_id = node_id

                async def _run_node():
                    set_context(node_ctx)
                    return await fn(state)

                node_task = asyncio.create_task(_run_node())
                await asyncio.sleep(0)
                result = await node_task

                node_status = result.get("status", "") if isinstance(result, dict) else ""
                node_error = result.get("error") if isinstance(result, dict) else None

                if node_status == "unchanged":
                    ctx.manager.close_span(ctx.run_id, span, SpanStatus.SUCCESS, output=result)
                    ctx.manager.emit(EventType.NODE_SKIPPED, ctx.run_id, span, {
                        "agent_id": agent_id,
                        "node_id": node_id,
                        "message": "文档未变更，跳过解析",
                        "progress_pct": 100,
                        "progress_text": "跳过",
                    })
                elif node_status == "error":
                    ctx.manager.close_span(ctx.run_id, span, SpanStatus.ERROR, output=result, error=node_error)
                    ctx.manager.emit(EventType.NODE_ERROR, ctx.run_id, span, {
                        "agent_id": agent_id,
                        "node_id": node_id,
                        "message": node_error or "节点执行失败",
                        "error": node_error,
                    })
                elif node_error and isinstance(node_error, str):
                    ctx.manager.close_span(ctx.run_id, span, SpanStatus.ERROR, output=result, error=node_error)
                    ctx.manager.emit(EventType.NODE_ERROR, ctx.run_id, span, {
                        "agent_id": agent_id,
                        "node_id": node_id,
                        "message": node_error,
                        "error": node_error,
                    })
                else:
                    ctx.manager.close_span(ctx.run_id, span, SpanStatus.SUCCESS, output=result)
                    ctx.manager.emit(EventType.NODE_SUCCESS, ctx.run_id, span, {
                        "agent_id": agent_id,
                        "node_id": node_id,
                        "message": _node_done_msg(node_id, result),
                        "step_index": 99,
                        "progress_pct": _node_progress_pct(node_id),
                        "progress_text": "完成",
                        "meta": _node_meta(node_id, result),
                        "output": result,
                    })

                await asyncio.sleep(0)
                return result

            except asyncio.CancelledError:
                node_task.cancel()
                ctx.manager.close_span(ctx.run_id, span, SpanStatus.ERROR, error="cancelled")
                ctx.manager.emit(EventType.NODE_ERROR, ctx.run_id, span, {
                    "agent_id": agent_id,
                    "node_id": node_id,
                    "message": "节点执行被取消",
                    "error": "cancelled",
                })
                raise

            except Exception as e:
                logger.exception("Node %s failed", node_id)
                ctx.manager.close_span(ctx.run_id, span, SpanStatus.ERROR, error=str(e))
                ctx.manager.emit(EventType.NODE_ERROR, ctx.run_id, span, {
                    "agent_id": agent_id,
                    "node_id": node_id,
                    "message": str(e),
                    "error": str(e),
                })
                await asyncio.sleep(0)
                return {"error": str(e)}

        return wrapper

    return decorator


# ── Helpers ──────────────────────────────────────────────────────────────────

def _node_label(node_id: str) -> str:
    labels = {
        "init_doc": "文档初始化",
        "parse_params": "参数解析",
        "product_support": "支持产品",
        "function_explanation_extract": "算子功能说明",
        "function_signature_extract": "函数原型",
        "src_content_extract": "原文片段",
        "param_desc_extract": "参数描述",
        "shape_extract": "维度",
        "dtype_extract": "数据类型",
        "optional_extract": "可选性",
        "dformat_extract": "数据格式",
        "param_attr_extract": "非连续Tensor",
        "array_length_extract": "数组长度",
        "allowed_range_extract": "取值范围",
        "determinism_extract": "确定性计算",
        "return_code_extract": "返回码",
        "dtype_combo_extract": "数据类型组合",
        "param_relation_extract": "参数约束关系",
    }
    return labels.get(node_id, node_id)


def _node_done_msg(node_id: str, result: dict) -> str:
    if node_id == "init_doc":
        sc = len(result.get("sections", []))
        v = result.get("version", 0)
        return f"文档初始化完成。v{v}, {sc} sections"
    elif node_id == "parse_params":
        pc = len(result.get("parameters", []))
        return f"参数解析完成。{pc} 个参数"
    elif node_id == "product_support":
        ps = len(result.get("product_support", []))
        return f"支持产品提取完成。{ps} 个产品"
    elif node_id == "function_explanation_extract":
        return "算子功能说明提取完成"
    elif node_id == "function_signature_extract":
        return "函数原型提取完成"
    elif node_id == "src_content_extract":
        return "原文片段提取完成"
    elif node_id == "param_desc_extract":
        return "参数描述提取完成"
    elif node_id == "shape_extract":
        return "维度提取完成"
    elif node_id == "dtype_extract":
        return "数据类型提取完成"
    elif node_id == "optional_extract":
        return "可选性提取完成"
    elif node_id == "dformat_extract":
        return "数据格式提取完成"
    elif node_id == "param_attr_extract":
        return "非连续Tensor提取完成"
    elif node_id == "array_length_extract":
        return "数组长度提取完成"
    elif node_id == "allowed_range_extract":
        return "取值范围提取完成"
    elif node_id == "determinism_extract":
        return "确定性计算提取完成"
    elif node_id == "return_code_extract":
        return "返回码提取完成"
    elif node_id == "dtype_combo_extract":
        return "数据类型组合提取完成"
    elif node_id == "param_relation_extract":
        return "参数约束关系提取完成"
    return f"{_node_label(node_id)} 完成"


def _node_meta(node_id: str, result: dict) -> str | None:
    if node_id == "init_doc":
        return f"v{result.get('version')} | {len(result.get('sections', []))} sections"
    elif node_id == "parse_params":
        return f"{len(result.get('parameters', []))} 参数"
    elif node_id == "product_support":
        return f"{len(result.get('product_support', []))} 产品"
    return None


def _node_progress_pct(node_id: str) -> int:
    pcts = {
        "init_doc": 10,
        "parse_params": 20,
        "product_support": 30,
        "function_explanation_extract": 40,
        "function_signature_extract": 45,
        "src_content_extract": 50,
        "param_desc_extract": 55,
        "shape_extract": 60,
        "dtype_extract": 65,
        "optional_extract": 70,
        "dformat_extract": 72,
        "param_attr_extract": 74,
        "array_length_extract": 76,
        "allowed_range_extract": 78,
        "determinism_extract": 80,
        "return_code_extract": 85,
        "dtype_combo_extract": 90,
        "param_relation_extract": 95,
    }
    return pcts.get(node_id, 50)
