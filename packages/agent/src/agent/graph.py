"""LangGraph agent for operator document processing.

Uses an LLM provider (Z.AI, DeepSeek, etc.) via ``ChatOpenAI`` and ``StateGraph``
to assemble a ReAct agent that orchestrates document parsing and section analysis.
The active provider is selected via the ``LLM_PROVIDER`` setting.

Also provides a deterministic pipeline graph for structured processing:
InitDoc → ParseParams → PersistParams.
"""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.core.config import settings
from agent.nodes.init_doc import init_doc_node
from agent.nodes.parse_params import parse_params_node
from agent.nodes.persist_params import persist_params_node
from agent.nodes.state import PipelineState
from agent.prompts import SYSTEM_PROMPT
from agent.tools.document_tools import (
    check_document_version,
    get_parsed_document,
    parse_document,
    save_document,
    save_parsed_result,
)
from agent.tools.section_tools import parse_all_sections, parse_section

logger = logging.getLogger(__name__)


def _create_llm() -> "ChatOpenAI":
    """Create a ChatOpenAI instance for the active LLM provider."""
    from langchain_openai import ChatOpenAI

    logger.info("Using LLM provider: %s, model: %s", settings.llm_provider, settings.active_model)
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=settings.llm_temperature,
    )


def create_operator_agent() -> CompiledStateGraph:
    """Build the operator document processing agent.

    Returns a LangGraph ``CompiledStateGraph`` that can be invoked with
    ``.ainvoke()`` or ``.astream()``.
    """
    llm = _create_llm()
    tools = [
        parse_document,
        check_document_version,
        save_document,
        save_parsed_result,
        get_parsed_document,
        parse_section,
        parse_all_sections,
    ]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        system = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(name="operator-agent")


def create_pipeline_graph() -> CompiledStateGraph:
    """Build a deterministic pipeline for document processing.

    Flow: InitDoc → ParseParams → PersistParams

    Returns a LangGraph ``CompiledStateGraph`` using ``PipelineState``.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("init_doc", init_doc_node)
    graph.add_node("parse_params", parse_params_node)
    graph.add_node("persist_params", persist_params_node)
    graph.add_edge(START, "init_doc")
    graph.add_edge("init_doc", "parse_params")
    graph.add_edge("parse_params", "persist_params")
    graph.add_edge("persist_params", END)
    return graph.compile(name="operator-pipeline")
