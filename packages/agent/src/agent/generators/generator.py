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
    """Return ``count`` test cases for the given parsed ``GeneratorContext``.

    Contract:
        generate(context, *, count, seed) -> list[TestCaseRecord]

    Determinism:
        Same ``context`` + same ``seed`` ⇒ identical case list.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []

    rng = random.Random(seed)
    shape_groups = build_shape_equal_groups(
        {"constraints_in_parameters": context.constraints_in_parameters}
    )
    # fixed_values is computed for API symmetry; not yet applied to scalar inputs.
    _ = build_fixed_values(
        {"constraints_in_parameters": context.constraints_in_parameters}
    )

    platform = _select_platform(context)
    cases: list[TestCaseRecord] = []
    for i in range(count):
        case = build_single_case(
            rng,
            context,
            idx=i,
            platform=platform,
            shape_groups=shape_groups,
        )
        cases.append(case)
    return cases


def _select_platform(context: GeneratorContext) -> str:
    if context.supported_platforms:
        return context.supported_platforms[0]
    return "default"
