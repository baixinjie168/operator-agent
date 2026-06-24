"""Generate ``range_values`` for tensors and scalar values.

Adapted from the legacy ``_generate_range_values`` and the scalar block of
``generate_single_case``.  When real constraint data is available, ranges
should be derived from the operator's ``allowed_range_value`` constraint
instead of being randomly sampled.
"""

from __future__ import annotations

import random
from typing import Any

_FLOAT_DTYPES = frozenset({"float32", "float16", "bfloat16", "float", "double"})
_INT_DTYPES = frozenset({"int32", "int64", "int", "bool"})


def sample_range_values(rng: random.Random, dtype: str) -> list[float] | list[int]:
    """Return a ``[min, max]`` value range appropriate for ``dtype``."""
    dt = dtype.lower()
    if dt in _FLOAT_DTYPES:
        return [rng.uniform(-10.0, -1.0), rng.uniform(1.0, 10.0)]
    if dt in _INT_DTYPES:
        return [rng.randint(0, 100), rng.randint(100, 200)]
    return [-5.0, 5.0]


def sample_scalar(
    rng: random.Random,
    dtype: str,
    *,
    allowed: list[Any] | None = None,
) -> float | int | bool:
    """Return a scalar value: pick from ``allowed`` if non-empty, else by dtype."""
    if allowed:
        return rng.choice(allowed)

    dt = dtype.lower()
    if dt in ("int64_t", "int64", "int32", "int32_t", "int"):
        return rng.randint(0, 100)
    if dt in ("double", "float"):
        return rng.uniform(-1.0, 1.0)
    if dt in ("bool", "boolean"):
        return rng.choice([True, False])
    return 0
