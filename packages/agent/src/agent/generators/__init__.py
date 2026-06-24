"""Test case generator — public API.

This package is the single source of truth for the case generation logic.
The only external-facing entry point is :class:`TestCaseGenerator`
(``generate(count) -> list[TestCaseRecord]``); all helper modules below
are implementation details.

Migration plan: the formal generation code will be dropped into this
package, replacing ``generator.generate`` (and/or the helper modules it
delegates to).  No caller in ``routes/`` / ``nodes/`` / MCP needs to
change — the public API stays stable.

Public re-exports
-----------------

The names below are part of the stable API and are imported by node,
route, and MCP code.
"""

from __future__ import annotations

from agent.generators.case_builder import build_single_case
from agent.generators.dtype_picker import (
    DEFAULT_DTYPE,
    map_aclnn_dtype_to_pytorch,
    pick_dtype_for_param,
)
from agent.generators.facade import (
    DEFAULT_COUNT,
    DEFAULT_SEED,
    TestCaseGenerator,
)
from agent.generators.generator import generate
from agent.generators.result_parser import parse_result_json
from agent.generators.shape_groups import (
    build_fixed_values,
    build_shape_equal_groups,
)
from agent.generators.shape_sampler import sample_shape
from agent.generators.value_sampler import (
    sample_range_values,
    sample_scalar,
)

__all__ = [
    # Public API
    "DEFAULT_COUNT",
    "DEFAULT_DTYPE",
    "DEFAULT_SEED",
    "TestCaseGenerator",
    "build_fixed_values",
    "build_shape_equal_groups",
    "build_single_case",
    "generate",
    "map_aclnn_dtype_to_pytorch",
    "parse_result_json",
    "pick_dtype_for_param",
    "sample_range_values",
    "sample_scalar",
    "sample_shape",
]
