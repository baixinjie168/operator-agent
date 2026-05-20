"""LangGraph agent for operator document processing.

Uses Z.AI (OpenAI-compatible API) via ``ChatOpenAI`` and ``create_react_agent``
to assemble a LangGraph agent that orchestrates document parsing and section analysis.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agent.core.config import settings
from agent.tools.document_tools import (
    check_document_version,
    get_parsed_document,
    parse_document,
    save_document,
    save_parsed_result,
)
from agent.tools.section_tools import parse_all_sections, parse_section

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a CANN operator document processing agent. When you receive a document \
to process, follow this workflow:

1. Call `parse_document` to split the document into structured sections.
2. For each section, call `parse_section` to extract detailed information.
3. Call `save_parsed_result` to persist the results.

Always process all sections and save results before responding with a summary.

Available section types:
- product_support: Hardware product support matrix
- function_prototype: C/C++ function signatures
- function_description: Functional description of the operator
- constraints: Usage constraints and limitations
- params_get_workspace: Parameters for GetWorkspaceSize API
- params_execute: Parameters for Execute API
- return_codes_get_workspace: Return codes for GetWorkspaceSize
- return_codes_execute: Return codes for Execute API
- get_workspace_size: GetWorkspaceSize function prototype
- execute_api: Execute function prototype
- usage_example: Code examples
"""


def _create_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for the Z.AI API."""
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def create_operator_agent() -> CompiledStateGraph:
    """Build the operator document processing agent.

    Returns a LangGraph ``CompiledStateGraph`` that can be invoked with
    ``.ainvoke()`` or ``.astream()``.
    """
    llm = _create_llm()
    return create_react_agent(
        model=llm,
        tools=[
            parse_document,
            check_document_version,
            save_document,
            save_parsed_result,
            get_parsed_document,
            parse_section,
            parse_all_sections,
        ],
        prompt=SYSTEM_PROMPT,
        name="operator-agent",
    )
