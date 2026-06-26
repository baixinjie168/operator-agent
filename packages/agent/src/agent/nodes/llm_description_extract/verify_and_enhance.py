"""VerifyAndEnhance node: LLM-based audit and enhancement.

Runs after validate_results. For each parameter whose description has
missing attributes or is too short, calls the LLM to verify against the
original document context. If the document contains the missing information,
replaces the description with an enhanced version; otherwise keeps the original.

Also builds a description_audit record for every result, tracking the
full extraction -> validation -> verification pipeline for human review
and process optimization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.nodes.llm_description_extract.extract_ws import (
    _parse_llm_response,
)
from agent.nodes.llm_description_extract.state import DescriptionExtractState
from agent.prompts import LLM_DESCRIPTION_VERIFY_PROMPT
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs_verification(result: dict) -> bool:
    """Check whether a result needs LLM verification.

    Triggers when:
    1. _validation.missing_attrs is non-empty
    2. _validation.desc_too_short is True
    """
    v = result.get("_validation", {})
    if v.get("missing_attrs"):
        return True
    if v.get("desc_too_short"):
        return True
    return False


def _build_audit_record(
    result: dict,
    validation: dict,
    verify_response: dict | None,
    enhanced: bool,
) -> dict:
    """Build a description_audit record for a single parameter.

    The audit tracks three stages:
    - extraction: the original llm_description from extract_ws/extract_exe
    - validation: the validate_results output (missing_attrs, desc_too_short)
    - verification: the verify_and_enhance outcome (enhanced or not)
    """
    audit: dict[str, Any] = {
        "extraction": {
            "original_description": result.get(
                "_original_description", result.get("llm_description", "")
            ),
            "original_src_content": result.get(
                "_original_src_content", result.get("src_content", "")
            ),
            "char_count": len(result.get(
                "_original_description", result.get("llm_description", "")
            )),
        },
        "validation": {
            "triggered": _needs_verification(result),
            "missing_attrs": validation.get("missing_attrs", []),
            "desc_too_short": validation.get("desc_too_short", False),
        },
        "verification": {
            "triggered": False,
        },
    }

    if verify_response is not None:
        audit["verification"] = {
            "triggered": True,
            "has_missing_info": verify_response.get("has_missing_info", False),
            "found_attrs": verify_response.get("found_attrs", []),
            "enhanced": enhanced,
            "enhanced_char_count": (
                len(verify_response.get("enhanced_description", ""))
                if enhanced
                else None
            ),
        }

    return audit


async def _verify_one(
    llm: ChatOpenAI,
    result: dict,
) -> dict:
    """Verify and optionally enhance a single parameter description.

    Also builds and attaches a description_audit record regardless of
    whether verification was triggered.
    """
    validation = result.get("_validation", {})
    context = result.get("_context", "")
    param_name = result.get("param_name", "")
    current_desc = result.get("llm_description", "")
    missing_attrs = validation.get("missing_attrs", [])

    # Snapshot original values before potential modification
    result["_original_description"] = current_desc
    result["_original_src_content"] = result.get("src_content", "")

    # Not triggered or no context available -> build audit and return
    if not _needs_verification(result) or not context.strip():
        result["description_audit"] = _build_audit_record(
            result, validation, verify_response=None, enhanced=False,
        )
        return result

    # Build verification prompt
    missing_attrs_text = (
        ", ".join(missing_attrs)
        if missing_attrs
        else "描述过短，需检查完整性"
    )

    prompt = LLM_DESCRIPTION_VERIFY_PROMPT.format(
        param_name=param_name,
        original_description=current_desc,
        missing_attrs=missing_attrs_text,
        document_context=context,
    )

    try:
        response = await llm.ainvoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.warning("VerifyEnhance: LLM call failed for %s", param_name)
        result["description_audit"] = _build_audit_record(
            result, validation, verify_response=None, enhanced=False,
        )
        return result

    parsed = _parse_llm_response(text)

    enhanced = False
    if parsed is not None:
        if parsed.get("has_missing_info") and parsed.get("enhanced_description"):
            enhanced_desc = parsed["enhanced_description"].strip()
            if len(enhanced_desc) > len(current_desc):
                logger.info(
                    "VerifyEnhance: %s enhanced (%d -> %d chars, found: %s)",
                    param_name,
                    len(current_desc),
                    len(enhanced_desc),
                    parsed.get("found_attrs", []),
                )
                result["llm_description"] = enhanced_desc
                # Append enhanced src_content
                enhanced_src = parsed.get("enhanced_src_content", "")
                if enhanced_src:
                    existing_src = result.get("src_content", "")
                    result["src_content"] = (
                        existing_src + "\n" + enhanced_src
                        if existing_src
                        else enhanced_src
                    )
                result["_enhanced"] = True
                enhanced = True
            else:
                logger.info(
                    "VerifyEnhance: %s enhanced desc not longer (%d <= %d), keeping original",
                    param_name,
                    len(enhanced_desc),
                    len(current_desc),
                )
        else:
            logger.info(
                "VerifyEnhance: %s - missing info not in document, keeping original",
                param_name,
            )
    else:
        logger.warning(
            "VerifyEnhance: failed to parse LLM response for %s", param_name,
        )

    # Build audit record
    result["description_audit"] = _build_audit_record(
        result, validation, verify_response=parsed, enhanced=enhanced,
    )
    return result


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

async def verify_and_enhance_node(
    state: DescriptionExtractState,
) -> dict[str, Any]:
    """Audit and enhance llm_descriptions for parameters that failed validation.

    For each result with _validation.missing_attrs or desc_too_short,
    calls the LLM to check if the original document contains the missing
    information. If yes, replaces the description with an enhanced version.

    Also builds description_audit records for all results.
    """
    ws_results = state.get("ws_results", [])
    exe_results = state.get("exe_results", [])
    all_results = ws_results + exe_results

    # Snapshot originals for audit before any modification
    for r in all_results:
        r["_original_description"] = r.get("llm_description", "")
        r["_original_src_content"] = r.get("src_content", "")

    # Determine which results need verification
    need_verify = [r for r in all_results if _needs_verification(r)]

    logger.info(
        "VerifyEnhance: %d/%d params need verification",
        len(need_verify),
        len(all_results),
    )

    if not need_verify:
        # Still build audit records for all results (not triggered)
        for r in all_results:
            validation = r.get("_validation", {})
            r["description_audit"] = _build_audit_record(
                r, validation, verify_response=None, enhanced=False,
            )
        return {
            "ws_results": ws_results,
            "exe_results": exe_results,
            "enhance_count": 0,
            "error": None,
        }

    try:
        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _task(r: dict) -> dict:
            async with sem:
                return await _verify_one(llm, r)

        await asyncio.gather(*[_task(r) for r in need_verify])

        # Build audit for results that did NOT need verification
        for r in all_results:
            if "description_audit" not in r:
                validation = r.get("_validation", {})
                r["description_audit"] = _build_audit_record(
                    r, validation, verify_response=None, enhanced=False,
                )

        enhanced_count = sum(1 for r in need_verify if r.get("_enhanced"))

        logger.info(
            "VerifyEnhance: enhanced %d/%d params",
            enhanced_count,
            len(need_verify),
        )

        return {
            "ws_results": ws_results,
            "exe_results": exe_results,
            "enhance_count": enhanced_count,
            "error": None,
        }

    except Exception:
        logger.exception("VerifyEnhance failed")
        # Ensure audit records exist even on failure
        for r in all_results:
            if "description_audit" not in r:
                validation = r.get("_validation", {})
                r["description_audit"] = _build_audit_record(
                    r, validation, verify_response=None, enhanced=False,
                )
        return {
            "ws_results": ws_results,
            "exe_results": exe_results,
            "enhance_count": 0,
            "error": "verify_enhance_failed",
        }
