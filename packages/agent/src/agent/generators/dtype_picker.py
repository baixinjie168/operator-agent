"""Pick dtypes per (platform, param) and map them to PyTorch strings.

The dtype mapping table (``FLOAT32`` → ``float32`` etc.) is stable and
reusable.  Per-platform dtype picking is currently first-combo-wins; this
can be replaced with more sophisticated resolution later.

Adapted from the legacy ``_get_dtype_combinations_for_platform`` and
``_map_dtype_to_pytorch`` methods.
"""

from __future__ import annotations

from typing import Any

DEFAULT_DTYPE = "FLOAT32"
_FALLBACK_PLATFORM = "default"

# aclnn dtype → pytorch dtype string (matches legacy)
_ACLNN_TO_TORCH: dict[str, str] = {
    "FLOAT32": "float32",
    "FLOAT16": "float16",
    "BFLOAT16": "bfloat16",
    "INT32": "int32",
    "INT64": "int64",
    "BOOL": "bool",
    "DOUBLE": "float64",
}


def map_aclnn_dtype_to_pytorch(dtype: str) -> str:
    """Map an aclnn dtype (e.g. ``FLOAT32``) to a PyTorch string (``float32``)."""
    if not dtype:
        return dtype
    mapped = _ACLNN_TO_TORCH.get(dtype)
    return mapped if mapped is not None else dtype.lower()


def pick_dtype_for_param(platform: str, param_name: str, context: dict[str, Any]) -> str:
    """Pick the dtype to use for ``param_name`` on ``platform``.

    Resolution order:
    1. First combo in ``dtype_support[platform]`` containing the param.
    2. First combo in ``dtype_support[default]`` containing the param.
    3. First dtype in ``inputs[param_name][platform].dtype.value``.
    4. ``DEFAULT_DTYPE`` as last resort.
    """
    dtype_support = context.get("dtype_support") or {}

    for plat in (platform, _FALLBACK_PLATFORM):
        combos = dtype_support.get(plat) or []
        for combo in combos:
            if not isinstance(combo, dict):
                continue
            if param_name in combo and combo[param_name]:
                return str(combo[param_name])

    inputs = context.get("inputs") or {}
    param_per_plat = inputs.get(param_name) or {}
    plat_block = param_per_plat.get(platform) or {}
    dtype_block = plat_block.get("dtype") or {}
    dtypes = dtype_block.get("value") or []
    if dtypes:
        return str(dtypes[0])

    return DEFAULT_DTYPE
