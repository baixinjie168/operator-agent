"""Public ``generate`` function — single entry point for case generation.

The :class:`agent.generators.TestCaseGenerator` facade calls
``generate(context, *, count, seed)`` to produce the list of test cases.
This module provides that contract; the actual sampling logic lives in
:mod:`agent.generators.case_builder` and friends.

To swap in the formal generation logic, replace the body of ``generate``
(and/or the helpers it delegates to).  ``TestCaseGenerator`` and the
public API in ``agent.generators.__init__`` stay unchanged.
"""

from __future__ import annotations

import random

from agent.generators.case_builder import build_single_case
from agent.generators.shape_groups import (
    build_fixed_values,
    build_shape_equal_groups,
)
from shared.models.test_case import GeneratorContext, TestCaseRecord


def generate(
    context: GeneratorContext,
    *,
    count: int,
    seed: int | None,
) -> list[TestCaseRecord]:
    """Return test cases for ALL supported platforms.

    For each supported platform, ``count`` cases are generated.
    Each case is tagged with its ``supported_product``.

    Contract:
        generate(context, *, count, seed) -> list[TestCaseRecord]

    Determinism:
        Same ``context`` + same ``seed`` => identical case list.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []

    rng = random.Random(seed)
    shape_groups = build_shape_equal_groups(
        {"constraints_in_parameters": context.constraints_in_parameters}
    )
    _ = build_fixed_values(
        {"constraints_in_parameters": context.constraints_in_parameters}
    )

    platforms = context.supported_platforms if context.supported_platforms else ["default"]

    cases: list[TestCaseRecord] = []
    global_idx = 0
    for platform in platforms:
        for i in range(count):
            case = build_single_case(
                rng,
                context,
                idx=global_idx,
                platform=platform,
                shape_groups=shape_groups,
            )
            # Tag the case with its product/platform
            case_data = case.model_dump()
            case_data["supported_product"] = platform
            case_data["id"] = global_idx
            cases.append(TestCaseRecord(**case_data))
            global_idx += 1
    return cases
