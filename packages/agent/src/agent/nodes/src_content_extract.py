"""SrcContentExtract node: extract original source text for each parameter via LLM."""

import asyncio
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import SRC_CONTENT_EXTRACT_PROMPT
from agent.runtime.context import get_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5

_STEP_NAME = "src_content_extract"
_STEP_LABEL = "源文本提取"


def _parse_src_content(raw: str) -> str:
    if not raw or raw.strip() in ("（无）", "(无)", "无"):
        return ""
    return raw.strip()


async def src_content_extract_node(state: PipelineState) -> dict[str, Any]:
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("SrcContentExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("SrcContentExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        if not params:
            logger.info("SrcContentExtract: no parameters for doc_id=%s, skipping", doc_id)
            return {"error": None}

        ws_section = await _mcp_client.get_section(doc_id, "params_get_workspace")
        ws_content = ws_section.get("content", "") if ws_section else ""

        exe_section = await _mcp_client.get_section(doc_id, "params_execute")
        exe_content = exe_section.get("content", "") if exe_section else ""

        ws_params = [p for p in params if p.get("function_name", "").endswith("GetWorkspaceSize")]
        exe_params = [p for p in params if not p.get("function_name", "").endswith("GetWorkspaceSize")]

        llm = _create_llm()
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)
        ctx = get_context()

        async def _extract_one(param: dict, content: str) -> dict | None:
            param_name = param["param_name"]
            function_name = param.get("function_name", "")
            parent_span_id = ctx.current_span_id if ctx else None
            param_span = None

            if ctx and ctx.manager:
                param_span = ctx.manager.open_span(
                    run_id=ctx.run_id,
                    parent_span_id=parent_span_id,
                    span_type=SpanType.NODE,
                    name=f"{_STEP_NAME}:{function_name}:{param_name}",
                )
                ctx.manager.emit(EventType.PARAM_STEP_START, ctx.run_id, param_span, {
                    "agent_id": "doc",
                    "node_id": _STEP_NAME,
                    "param_name": param_name,
                    "function_name": function_name,
                    "step_name": _STEP_NAME,
                    "message": f"参数 {param_name} {_STEP_LABEL} 开始...",
                })

            try:
                async with sem:
                    raw = await _extract_src(llm, param_name, content)
                src_content = _parse_src_content(raw)

                if ctx and ctx.manager and param_span:
                    ctx.manager.close_span(ctx.run_id, param_span, SpanStatus.SUCCESS)
                    ctx.manager.emit(EventType.PARAM_STEP_COMPLETE, ctx.run_id, param_span, {
                        "agent_id": "doc",
                        "node_id": _STEP_NAME,
                        "param_name": param_name,
                        "function_name": function_name,
                        "step_name": _STEP_NAME,
                        "message": f"参数 {param_name} {_STEP_LABEL} 完成",
                        "result_preview": src_content if src_content else "",
                        "has_result": bool(src_content),
                    })

                if not src_content:
                    return None
                return {
                    "function_name": function_name,
                    "param_name": param_name,
                    "param_type": param.get("param_type", ""),
                    "src_content": src_content,
                }
            except Exception as exc:
                if ctx and ctx.manager and param_span:
                    ctx.manager.close_span(ctx.run_id, param_span, SpanStatus.ERROR, error=str(exc))
                    ctx.manager.emit(EventType.PARAM_STEP_ERROR, ctx.run_id, param_span, {
                        "agent_id": "doc",
                        "node_id": _STEP_NAME,
                        "param_name": param_name,
                        "function_name": function_name,
                        "step_name": _STEP_NAME,
                        "message": f"参数 {param_name} {_STEP_LABEL} 失败: {exc}",
                        "error": str(exc),
                    })
                return None

        all_updates: list[dict] = []

        def _emit_skip(param: dict, reason: str) -> None:
            param_name = param["param_name"]
            function_name = param.get("function_name", "")
            if ctx and ctx.manager:
                skip_span = ctx.manager.open_span(
                    run_id=ctx.run_id,
                    parent_span_id=ctx.current_span_id if ctx else None,
                    span_type=SpanType.NODE,
                    name=f"{_STEP_NAME}:{function_name}:{param_name}",
                )
                ctx.manager.close_span(ctx.run_id, skip_span, SpanStatus.SUCCESS)
                ctx.manager.emit(EventType.PARAM_STEP_ERROR, ctx.run_id, skip_span, {
                    "agent_id": "doc",
                    "node_id": _STEP_NAME,
                    "param_name": param_name,
                    "function_name": function_name,
                    "step_name": _STEP_NAME,
                    "message": f"参数 {param_name} {_STEP_LABEL} 已跳过: {reason}",
                    "error": reason,
                })

        if ws_params:
            if ws_content:
                results = await asyncio.gather(*[_extract_one(p, ws_content) for p in ws_params])
                all_updates.extend(r for r in results if r is not None)
            else:
                logger.warning(
                    "SrcContentExtract: params_get_workspace section not found for doc_id=%s, "
                    "skipping %d GetWorkspaceSize parameters",
                    doc_id,
                    len(ws_params),
                )
                for p in ws_params:
                    _emit_skip(p, "文档中未找到 GetWorkspaceSize 参数说明段落")

        if exe_params:
            if exe_content:
                results = await asyncio.gather(*[_extract_one(p, exe_content) for p in exe_params])
                all_updates.extend(r for r in results if r is not None)
            else:
                logger.warning(
                    "SrcContentExtract: params_execute section not found for doc_id=%s, skipping %d Execute parameters",
                    doc_id,
                    len(exe_params),
                )
                for p in exe_params:
                    _emit_skip(p, "文档中未找到执行函数参数说明段落")

        if all_updates:
            result = await _mcp_client.update_param_src_content(doc_id, all_updates)
            logger.info(
                "SrcContentExtract: updated src_content for %d/%d parameters (doc_id=%s)",
                result.get("updated", 0),
                len(params),
                doc_id,
            )
        else:
            logger.info("SrcContentExtract: no source content extracted for doc_id=%s", doc_id)

        return {"parameters": all_updates, "error": None}

    except Exception as e:
        logger.exception("SrcContentExtract failed for %s", operator_name)
        return {"error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_src(llm: ChatOpenAI, param_name: str, content: str) -> str:
    """Call LLM to extract source content for a single parameter."""
    prompt = SRC_CONTENT_EXTRACT_PROMPT.format(param_name=param_name, content=content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()
