"""Parameter validation utilities shared across extraction nodes.

Provides zero-LLM-cost pre-filtering helpers:
- is_cross_reference: detect cross-parameter references ("与xxx一致")
- is_dash: detect dash variants (-, —, –, etc.)
- is_ws_function: detect GetWorkspaceSize function names
- is_bool_type / is_tensor_type: classify parameter types
- VALID_DTYPES: approved dtype whitelist
- EXCLUDED_PARAMS: two-stage API common params to skip
"""

from __future__ import annotations

import re

# Two-stage API common parameters (workspace, executor, stream) that are
# not real operator parameters — excluded from relation extraction and
# constraint assembly.
EXCLUDED_PARAMS = frozenset({
    "workspace", "workspaceSize", "executor", "stream",
})

# Matches cross-reference patterns like "与self一致", "同input相同",
# "和xxx保持一致", "与`xxx`一致" etc.
RELATIVE_REF_RE = re.compile(
    r"^(?:与|同|和|跟)"
    r".{1,20}"
    r"(?:一致|相同|一样|保持一致|保持一致|同)$",
)

# Dash variants that indicate "no value"
_DASH_VARIANTS = frozenset(("-", "—", "–", "－"))

# Approved dtype whitelist
VALID_DTYPES = frozenset({
    "FLOAT", "FLOAT32", "FLOAT16", "INT8", "INT32", "UINT8",
    "INT16", "UINT16", "UINT32", "INT64", "UINT64", "DOUBLE",
    "FLOAT64", "BOOL", "STRING", "COMPLEX64", "COMPLEX128",
    "BF16", "BFLOAT16", "INT", "UINT1", "COMPLEX32",
})


def is_cross_reference(value: str) -> bool:
    """Check whether *value* is a cross-parameter reference.

    Matches patterns like "与self一致", "同input相同", etc.
    Strips backticks before matching.
    """
    cleaned = value.strip().replace("`", "")
    return bool(RELATIVE_REF_RE.match(cleaned))


def is_dash(value: str) -> bool:
    """Check whether *value* is a dash variant (no value indicator)."""
    return value.strip() in _DASH_VARIANTS


def is_ws_function(function_name: str) -> bool:
    """Return True if *function_name* is a GetWorkspaceSize variant."""
    return "GetWorkspaceSize" in function_name


def is_bool_type(param_type: str) -> bool:
    """Check if parameter type is bool."""
    return param_type.lower() == "bool"


def is_tensor_type(param_type: str) -> bool:
    """Check if parameter type is a Tensor or TensorList.

    Tensor parameters have no scalar value range — ``allowed_range_value``
    only applies to scalar types (int64_t, double, aclScalar, etc.).
    """
    return "aclTensor" in param_type
