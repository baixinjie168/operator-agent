"""Parse ``result.json`` into a typed ``GeneratorContext``.

The generator pipeline consumes a parsed ``GeneratorContext`` rather than the
raw JSON; this module does the one-time flattening from the nested per-platform
``result.json`` shape into the flat structure the rest of the generator uses.

Input shape (new format)::

    {
      "operator_name": "aclnnFoo",
      "product_support": ["P1", "P2"],
      "inputs": {
        "x1": {
          "P1": {
            "dtype": {"value": ["FLOAT32"], "src_text": ""},
            "dimensions": {"value": [[2, 8]], "src_text": "2-8"},
            "is_optional": {"value": false, "src_text": ""},
            "type": {"value": "const aclTensor", "src_text": ""},
            ...
          }
        }
      },
      "outputs": {...},
      "constraints_in_parameters": {"P1": [{...}, ...]},
      "dtype_support_description": {"P1": [{"x1": "FLOAT32", ...}]}
    }
"""

from __future__ import annotations

from typing import Any

from shared.models.test_case import GeneratorContext


_ACLNN_PREFIX = "aclnn"


def parse_result_json(result: dict[str, Any]) -> GeneratorContext:
    """Convert a raw ``result.json`` dict into a typed ``GeneratorContext``.

    Args:
        result: The dict loaded from ``document_versions.json_constraints``.

    Returns:
        A frozen ``GeneratorContext`` suitable for the generator facade.

    Raises:
        TypeError: If ``result`` is not a dict.
        ValueError: If ``operator_name`` is missing or empty.
    """
    if not isinstance(result, dict):
        raise TypeError(f"result must be a dict, got {type(result).__name__}")

    operator_name = result.get("operator_name")
    if not operator_name or not isinstance(operator_name, str):
        raise ValueError("result.operator_name is required and must be a non-empty string")

    aclnn_name = _strip_aclnn_prefix(operator_name)
    supported_platforms = list(result.get("product_support") or [])
    inputs = dict(result.get("inputs") or {})
    outputs = dict(result.get("outputs") or {})
    constraints_in_parameters = dict(result.get("constraints_in_parameters") or {})
    dtype_support = dict(result.get("dtype_support_description") or {})

    return GeneratorContext(
        operator_name=operator_name,
        aclnn_name=aclnn_name,
        supported_platforms=supported_platforms,
        inputs=inputs,
        outputs=outputs,
        constraints_in_parameters=constraints_in_parameters,
        dtype_support=dtype_support,
    )


def _strip_aclnn_prefix(operator_name: str) -> str:
    """Strip the ``aclnn`` prefix to get the bare operator name (e.g. ``AdaLayerNorm``)."""
    if operator_name.startswith(_ACLNN_PREFIX):
        return operator_name[len(_ACLNN_PREFIX):]
    return operator_name
