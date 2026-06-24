"""Sample a random tensor shape.

Adapted from the legacy ``_generate_random_shape`` method.  Honors
``dimensions_value`` (e.g. ``[[2, 8]]``) as a hint for the ndim range.
"""

from __future__ import annotations

import random
from typing import Sequence


def sample_shape(
    rng: random.Random,
    *,
    ndim_min: int = 1,
    ndim_max: int = 8,
    max_elements: int = 100,
    dimensions_value: Sequence[Sequence[int]] | None = None,
) -> list[int]:
    """Generate a random shape whose dim count lies in ``[ndim_min, ndim_max]``.

    Args:
        rng: Seeded ``random.Random`` instance for reproducibility.
        ndim_min: Minimum rank (inclusive).
        ndim_max: Maximum rank (inclusive).
        max_elements: Soft cap on the product of dims.
        dimensions_value: Optional list of ``[min, max]`` ndim ranges from the
            result.json constraints — when present, the ndim is sampled from
            their union's intersection with ``[ndim_min, ndim_max]``.

    Returns:
        A list of positive integers, length in ``[ndim_min, ndim_max]``.
    """
    eff_ndim_min, eff_ndim_max = _resolve_ndim_range(
        ndim_min, ndim_max, dimensions_value
    )
    ndim = rng.randint(eff_ndim_min, eff_ndim_max)

    shape: list[int] = []
    remaining = max(1, max_elements)
    for i in range(ndim):
        if i < ndim - 1:
            upper = max(1, min(remaining, 10))
            dim_val = rng.randint(1, upper)
            shape.append(dim_val)
            remaining = max(1, remaining // dim_val)
        else:
            shape.append(max(1, remaining))
    return shape


def _resolve_ndim_range(
    ndim_min: int,
    ndim_max: int,
    dimensions_value: Sequence[Sequence[int]] | None,
) -> tuple[int, int]:
    """Intersect ``[ndim_min, ndim_max]`` with the union of ``dimensions_value`` ranges."""
    if not dimensions_value:
        return ndim_min, ndim_max
    lo, hi = ndim_min, ndim_max
    for entry in dimensions_value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            a, b = int(entry[0]), int(entry[1])
        except (TypeError, ValueError):
            continue
        if a > b:
            a, b = b, a
        lo = max(lo, a)
        hi = max(hi, b)
    lo = max(1, lo)
    hi = max(lo, hi)
    return lo, hi
