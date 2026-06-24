"""Agent loop: iterative relation extraction with coverage-driven refinement.

Flow:
  Round 1: Chunked extraction (divide and conquer)
  Round 2: Dual coverage check (params + paragraphs)
  Round 3: Targeted extraction (uncovered params + paragraphs)
  Round 4: Self-reflection (max 2 rounds)
"""

import asyncio
import logging
import re
import time
from typing import Any

from langchain_openai import ChatOpenAI

from agent.nodes.param_relation_extract.chunked_extract import extract_relations_chunked
from agent.nodes.param_relation_extract.coverage import (
    find_uncovered_context_mentions,
    find_uncovered_params,
)
from agent.nodes.param_relation_extract.merge_relations import _dedup_relations
from agent.nodes.param_relation_extract.self_check import (
    extract_relations_for_param,
    extract_relations_for_paragraph,
    self_check_relations,
)

logger = logging.getLogger(__name__)

MAX_SELF_CHECK_ROUNDS = 2
MAX_WALL_TIME_PER_DOC = 300  # seconds


async def extract_relations_agent(
    section_content: str,
    param_names: list[str],
    llm: ChatOpenAI,
    implicit_params: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Agent-style iterative relation extraction with self-check loop.

    Returns:
        (all_relations, coverage_report)

    Any round failure gracefully degrades to already-extracted results.
    A wall-time guard (MAX_WALL_TIME_PER_DOC) ensures the function returns
    within the budget; on timeout, round-1 results are returned.
    """
    t0 = time.monotonic()

    try:
        all_relations, report = await asyncio.wait_for(
            _run_agent_rounds(section_content, param_names, llm, t0, implicit_params),
            timeout=MAX_WALL_TIME_PER_DOC,
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        logger.warning(
            "Agent loop timed out after %.1fs (limit %ds) — returning round-1 results",
            elapsed, MAX_WALL_TIME_PER_DOC,
        )
        # Fallback: re-run only round 1 to get at least partial results
        try:
            chunked = await extract_relations_chunked(section_content, llm)
            all_relations = _dedup_relations(chunked)
        except Exception:
            logger.exception("Round-1 fallback also failed")
            all_relations = []
        final_unc = find_uncovered_params(param_names, all_relations)
        report = {
            "round1": len(all_relations),
            "round3": 0,
            "round4": 0,
            "self_check_rounds": 0,
            "total_rounds": 1,
            "total": len(all_relations),
            "uncovered_params": final_unc,
            "coverage": f"{len(param_names) - len(final_unc)}/{len(param_names)}",
            "timed_out": True,
        }

    return _cleanup(all_relations), report


async def _run_agent_rounds(
    section_content: str,
    param_names: list[str],
    llm: ChatOpenAI,
    t0: float,
    implicit_params: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Core agent rounds (extracted so asyncio.wait_for can wrap it)."""
    all_relations: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "round1": 0,
        "round3": 0,
        "round4": 0,
        "self_check_rounds": 0,
        "total_rounds": 0,
    }
    has_targeted = False

    try:
        # ---- Round 1: Chunked extraction ----
        chunked = await extract_relations_chunked(section_content, llm, implicit_params)
        all_relations = _dedup_relations(chunked)
        report["round1"] = len(all_relations)
        logger.info("Agent round 1 (chunked): %d relations (%.1fs)",
                     len(all_relations), time.monotonic() - t0)

        if not section_content.strip() or not param_names:
            report["total_rounds"] = 1
            report["total"] = len(all_relations)
            report["uncovered_params"] = param_names
            report["coverage"] = f"0/{len(param_names)}"
            return all_relations, report

        # ---- Round 2: Dual coverage check ----
        uncovered_params = find_uncovered_params(param_names, all_relations)
        uncovered_paragraphs = find_uncovered_context_mentions(
            section_content, param_names, all_relations,
        )

        logger.info(
            "Agent round 2 (coverage): %d/%d params uncovered, %d paragraphs (%.1fs)",
            len(uncovered_params), len(param_names),
            len(uncovered_paragraphs), time.monotonic() - t0,
        )

        has_targeted = bool(uncovered_params or uncovered_paragraphs)

        # ---- Round 3: Targeted extraction ----
        if has_targeted:
            sem = asyncio.Semaphore(5)

            async def _extract_for_param(name: str) -> list[dict[str, Any]]:
                async with sem:
                    try:
                        return await extract_relations_for_param(
                            llm, section_content, name, param_names,
                        )
                    except Exception:
                        logger.warning("Targeted extraction failed for param %s", name)
                        return []

            async def _extract_for_paragraph(para: str) -> list[dict[str, Any]]:
                async with sem:
                    try:
                        mentioned = [
                            n for n in param_names
                            if re.search(
                                r"(?<![a-zA-Z0-9_])" + re.escape(n) + r"(?![a-zA-Z0-9_])",
                                para,
                            )
                        ]
                        return await extract_relations_for_paragraph(
                            llm, para, mentioned,
                        )
                    except Exception:
                        logger.warning("Paragraph targeted extraction failed")
                        return []

            tasks: list = []
            for name in uncovered_params:
                tasks.append(_extract_for_param(name))
            for para in uncovered_paragraphs:
                tasks.append(_extract_for_paragraph(para))

            results = await asyncio.gather(*tasks)
            for rels in results:
                all_relations.extend(rels)
            all_relations = _dedup_relations(all_relations)
            report["round3"] = len(all_relations) - report["round1"]
            logger.info("Agent round 3 (targeted): +%d relations (%.1fs)",
                        report["round3"], time.monotonic() - t0)

        # ---- Round 4: Self-reflection (max 2 rounds) ----
        for self_round in range(MAX_SELF_CHECK_ROUNDS):
            try:
                additional = await self_check_relations(
                    llm, all_relations, section_content,
                )
            except Exception:
                logger.warning("Self-check round %d failed, skipping", self_round + 1)
                break

            if not additional:
                break

            before = len(all_relations)
            all_relations = _dedup_relations(all_relations + additional)
            new_count = len(all_relations) - before
            report["round4"] += new_count
            report["self_check_rounds"] += 1
            logger.info(
                "Agent round 4.%d (self-check): +%d relations (%.1fs)",
                self_round + 1, new_count, time.monotonic() - t0,
            )

            if new_count == 0:
                break

    except Exception:
        # Top-level fallback: return whatever we have so far
        logger.exception(
            "Agent loop exception, falling back to %d extracted relations",
            len(all_relations),
        )

    # ---- Statistics ----
    report["total_rounds"] = (
        1 + (1 if has_targeted else 0) + report["self_check_rounds"]
    )

    final_uncovered = find_uncovered_params(param_names, all_relations)
    report["total"] = len(all_relations)
    report["uncovered_params"] = final_uncovered
    report["coverage"] = (
        f"{len(param_names) - len(final_uncovered)}/{len(param_names)}"
    )

    logger.info(
        "Agent complete: %d relations, coverage=%s, %d rounds (%.1fs)",
        report["total"], report["coverage"], report["total_rounds"],
        time.monotonic() - t0,
    )

    return all_relations, report


def _cleanup(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove internal marker fields."""
    for r in relations:
        r.pop("_source", None)
        r.pop("_source_chunk", None)
    return relations
