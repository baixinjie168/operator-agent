"""Shared LLM infrastructure for all pipeline extraction nodes.

Provides:
- JSON_BLOCK_RE: compiled regex for extracting JSON from LLM code blocks
- CONCURRENCY_LIMIT: default semaphore limit for parallel LLM calls
- parse_json_response: generic JSON extraction from LLM text output

Note: create_llm is now imported directly from agent.core.llm.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex: matches ```json ... ``` or ``` ... ``` code blocks
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Default concurrency limit for parallel LLM calls
CONCURRENCY_LIMIT = 5


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
