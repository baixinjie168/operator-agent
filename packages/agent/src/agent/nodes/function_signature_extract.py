"""FunctionSignatureExtract node: extract function signatures and parameters via LLM.

Replaces the old separate parse_params node.  A single LLM call extracts
both structured function signatures (for the function_signatures table) and
a flat parameter list (for the parameters table / state.parameters).
"""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import FUNCTION_SIGNATURE_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


async def function_signature_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract function signatures and flat parameter list from function_prototype.

    Flow:
    1. Get function_prototype section content via MCP
    2. Call LLM to extract structured signatures
    3. Normalize param types (strip const/pointer)
    4. Save signatures to function_signatures table via MCP
    5. Flatten signatures to parameter list
    6. Save parameters to parameters table via MCP
    7. Return parameters in state for downstream nodes
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("FunctionSignatureExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("FunctionSignatureExtract: no doc_id in state, skipping")
        return {"parameters": [], "error": None}

    try:
        section = await _mcp_client.get_section(doc_id, "function_prototype")
        if not section:
            logger.warning("FunctionSignatureExtract: no function_prototype section for doc_id=%s", doc_id)
            return {"parameters": [], "error": None}

        content = section.get("content", "")
        if not content:
            logger.warning("FunctionSignatureExtract: empty function_prototype content for doc_id=%s", doc_id)
            return {"parameters": [], "error": None}

        signatures = await _extract_signatures_via_llm(content)
        signatures = _normalize_param_types(signatures)
        if not signatures:
            logger.info("FunctionSignatureExtract: LLM returned no results for doc_id=%s", doc_id)
            return {"function_signatures": [], "parameters": [], "error": None}

        logger.info(
            "FunctionSignatureExtract: extracted %d signatures for %s",
            len(signatures),
            operator_name,
        )

        # Save signatures to function_signatures table
        result = await _mcp_client.save_function_signatures(doc_id, signatures)
        logger.info(
            "FunctionSignatureExtract: saved %d signatures for doc_id=%s",
            result.get("saved", 0),
            doc_id,
        )

        # Flatten to parameters list and save to parameters table
        parameters = _signatures_to_parameters(signatures)
        if parameters:
            await _mcp_client.save_parameters(doc_id, parameters)
            logger.info(
                "FunctionSignatureExtract: saved %d parameters for doc_id=%s",
                len(parameters),
                doc_id,
            )

        return {"function_signatures": signatures, "parameters": parameters, "error": None}

    except Exception as e:
        logger.exception("FunctionSignatureExtract failed for %s", operator_name)
        return {"parameters": [], "error": str(e)}


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        temperature=0.1,
    )


async def _extract_signatures_via_llm(content: str) -> list[dict]:
    """Call LLM to extract function signatures from content."""
    llm = _create_llm()
    prompt = FUNCTION_SIGNATURE_EXTRACT_PROMPT.format(content=content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_json_response(text)


def _normalize_param_types(signatures: list[dict]) -> list[dict]:
    """Strip const and pointer modifiers from parameters.type.

    Ensures type field contains only the base type name (e.g. "aclTensor"),
    not the full C declaration (e.g. "const aclTensor *").
    """
    for sig in signatures:
        for param in sig.get("parameters", []):
            ptype = param.get("type", "")
            # Strip const, pointer *, and reference &
            ptype = re.sub(r'\bconst\b', '', ptype)
            ptype = ptype.replace('*', '').replace('&', '').strip()
            param["type"] = ptype
    return signatures


def _parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)

    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            data = json.loads(array_match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def _signatures_to_parameters(signatures: list[dict]) -> list[dict]:
    """Flatten structured signatures into a flat parameters list.

    Each signature's parameters array is expanded into individual records
    with keys: function_name, param_name, param_type.

    The ``direction`` field is intentionally omitted so the DB stores an
    empty string as placeholder.  Downstream nodes (table_column_extract and
    llm_description_extract) will fill in the correct direction.
    """
    parameters: list[dict] = []
    for sig in signatures:
        func_name = sig.get("function_name", "")
        for param in sig.get("parameters", []):
            if isinstance(param, str):
                param_name = param
                param_type = ""
            else:
                param_name = param.get("name", "")
                # param_type is already normalized by _normalize_param_types
                param_type = param.get("type", "")
            parameters.append({
                "function_name": func_name,
                "param_name": param_name,
                "param_type": param_type,
            })
    return parameters
