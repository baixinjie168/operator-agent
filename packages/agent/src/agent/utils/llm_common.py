"""Shared LLM infrastructure for all pipeline extraction nodes.

Provides:
- JSON_BLOCK_RE: compiled regex for extracting JSON from LLM code blocks
- CONCURRENCY_LIMIT: default semaphore limit for parallel LLM calls
- create_llm: factory re-export from agent.core.llm
- parse_json_response: generic JSON extraction from LLM text output
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Regex: matches ```json ... ``` or ``` ... ``` code blocks
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Default concurrency limit for parallel LLM calls
CONCURRENCY_LIMIT = 5


def create_llm() -> ChatOpenAI:
    """Re-export of agent.core.llm.create_llm for convenience.

    Nodes can import directly from here instead of agent.core.llm
    to keep their import lists shorter.
    """
    from agent.core.llm import create_llm as _create_llm

    return _create_llm()


def parse_json_response(text: str, expected_type: type = dict) -> Any | None:
    """Extract structured JSON from LLM text output.

    Strategy:
    1. Try to find a ```json ... ``` code block and parse its content
    2. Fall back to parsing the stripped text directly
    3. Fall back to regex extraction of the first matching JSON structure

    Args:
        text: Raw LLM response text.
        expected_type: Expected Python type (dict, list, etc.).

    Returns:
        Parsed JSON object if successful, None otherwise.
    """
    # Phase 1: extract from code block
    match = JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    # Phase 2: direct parse
    try:
        data = json.loads(text)
        if isinstance(data, expected_type):
            return data
    except json.JSONDecodeError:
        pass

    # Phase 3: regex fallback (greedy — handles nested structures)
    if expected_type is dict:
        pattern = r"\{[\s\S]*\}"
    elif expected_type is list:
        pattern = r"\[[\s\S]*\]"
    else:
        logger.warning(
            "parse_json_response: unsupported expected_type=%s", expected_type,
        )
        return None

    obj_match = re.search(pattern, text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, expected_type):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning(
        "parse_json_response: failed to parse LLM response as %s: %s",
        expected_type.__name__,
        text[:200],
    )
    return None
