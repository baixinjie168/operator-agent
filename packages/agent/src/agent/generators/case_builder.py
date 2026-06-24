"""Assemble a single ``TestCaseRecord`` from a ``GeneratorContext``.

Produces a ``TestCaseRecord`` using random sampling for shapes, range_values,
and dtypes.  The deterministic seed (``TestCaseGenerator.seed``) is the
single source of randomness, so two generators with the same seed + context
produce identical cases.

Key choices:
* 50% probability drop for ``is_optional=true`` params (matches legacy).
* Shape-equality groups share a single sampled shape.
* ``dtype`` is taken from ``dtype_support[platform]`` first, then
  param-level dtype list, then default.

When the formal generation logic lands, replace the body of
``generator.generate`` and/or replace the helper functions in this module.
The ``TestCaseGenerator`` entry point and the public API surface stay the
same.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from agent.generators.dtype_picker import (
    map_aclnn_dtype_to_pytorch,
    pick_dtype_for_param,
)
from agent.generators.shape_sampler import sample_shape
from agent.generators.value_sampler import sample_range_values
from shared.models.test_case import (
    GeneratorContext,
    StandardSpec,
    TensorInputSpec,
    TestCaseRecord,
)


_OPTIONAL_DROP_PROB = 0.5
_MAX_SHAPE_ELEMENTS = 50
_NDIM_MIN = 1
_NDIM_MAX = 8


def build_single_case(
    rng: random.Random,
    context: GeneratorContext,
    *,
    idx: int,
    platform: str,
    shape_groups: Iterable[Iterable[str]] | None = None,
    force_include: Iterable[str] | None = None,
    force_exclude: Iterable[str] | None = None,
) -> TestCaseRecord:
    """Build one ``TestCaseRecord``.

    Args:
        rng: Seeded random source.
        context: Parsed operator constraints.
        idx: Case index (``id`` field of the record).
        platform: Platform key to use for dtype/shape resolution.
        shape_groups: Optional equivalence groups from
            ``build_shape_equal_groups``.  All members share one sampled shape.
        force_include: Param names that must always be included (test hook).
        force_exclude: Param names that must always be dropped (test hook).
    """
    forced_in = set(force_include or ())
    forced_out = set(force_exclude or ())
    shape_groups = list(shape_groups or [])

    # Sample one shape per group (deterministic order).
    group_shapes: dict[frozenset[str], list[int]] = {}
    for group in shape_groups:
        members = frozenset(group)
        if not members:
            continue
        first = next(iter(members))
        dims = _param_dimensions_value(context, first, platform)
        group_shapes[members] = sample_shape(
            rng,
            ndim_min=_NDIM_MIN,
            ndim_max=_NDIM_MAX,
            max_elements=_MAX_SHAPE_ELEMENTS,
            dimensions_value=dims,
        )

    inputs: list[TensorInputSpec] = []
    for param_name, per_plat in context.inputs.items():
        if param_name in forced_out:
            continue
        if param_name not in forced_in and _should_drop(
            rng, context, param_name, platform
        ):
            continue

        block = per_plat.get(platform) or {}
        # Choose shape: group shape if this param is in a group, else sample.
        group_shape = _lookup_group_shape(shape_groups, group_shapes, param_name)
        if group_shape is not None:
            shape: list[int] | None = list(group_shape)
        else:
            dims = _param_dimensions_value(context, param_name, platform)
            shape = sample_shape(
                rng,
                ndim_min=_NDIM_MIN,
                ndim_max=_NDIM_MAX,
                max_elements=_MAX_SHAPE_ELEMENTS,
                dimensions_value=dims,
            )

        aclnn_dtype = pick_dtype_for_param(platform, param_name, context.model_dump())
        torch_dtype = map_aclnn_dtype_to_pytorch(aclnn_dtype)
        range_values = sample_range_values(rng, torch_dtype)

        inputs.append(
            TensorInputSpec(
                name=param_name,
                type="tensor",
                required=not _is_optional(context, param_name, platform),
                dtype=torch_dtype,
                shape=shape,
                range_values=range_values,
                backward=False,
                align_32B=None,
                outlier_values=None,
            )
        )

    aclnn_lower = context.aclnn_name.lower()
    return TestCaseRecord(
        id=idx,
        name=context.operator_name,
        aclnn_name=context.aclnn_name,
        triton_name=None,
        version="v1.0",
        expected_error_msg=None,
        api="pytorch",
        api_type=f"function_{aclnn_lower}",
        aclnn_api_type=f"pyaclnn_aclnn_{aclnn_lower}",
        triton_api_type="triton_function",
        fusion_api_type="fusion_function",
        fusion_mode=None,
        dist_api_type="dist_function",
        backward=False,
        standard=StandardSpec(),
        outputs=None,
        inputs=inputs,
    )


def _should_drop(
    rng: random.Random,
    context: GeneratorContext,
    param_name: str,
    platform: str,
) -> bool:
    """Return True if the param should be dropped (50% for optional, never for required)."""
    if not _is_optional(context, param_name, platform):
        return False
    return rng.random() < _OPTIONAL_DROP_PROB


def _is_optional(context: GeneratorContext, param_name: str, platform: str) -> bool:
    per_plat = context.inputs.get(param_name) or {}
    block = per_plat.get(platform) or {}
    is_optional = (block.get("is_optional") or {}).get("value")
    return bool(is_optional) if is_optional is not None else False


def _param_dimensions_value(
    context: GeneratorContext, param_name: str, platform: str
) -> list[list[int]] | None:
    per_plat = context.inputs.get(param_name) or {}
    block = per_plat.get(platform) or {}
    dims = (block.get("dimensions") or {}).get("value")
    if not isinstance(dims, list) or not dims:
        return None
    return dims


def _lookup_group_shape(
    shape_groups: list[Iterable[str]],
    group_shapes: dict[frozenset[str], list[int]],
    param_name: str,
) -> list[int] | None:
    for group in shape_groups:
        members = frozenset(group)
        if param_name in members and members in group_shapes:
            return group_shapes[members]
    return None
