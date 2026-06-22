"""LLM prompt templates for the operator document processing pipeline.

Organized by functional domain:
- basic_info: function signatures, product support, function explanation
- llm_description: parameter description extraction and verification
- per_param: per-parameter attribute extraction (dtype, dformat, shape, etc.)
- relations: parameter relation/constraint expression building
- assembly: shape→dimensions conversion, allowed_range_value building
- extraction: specialized extraction (return codes, determinism, dtype combos)
"""

from agent.prompts.basic_info import (
    FUNCTION_EXPLANATION_EXTRACT_PROMPT,
    FUNCTION_SIGNATURE_EXTRACT_PROMPT,
    PRODUCT_SUPPORT_EXTRACT_PROMPT,
)
from agent.prompts.llm_description import (
    LLM_DESCRIPTION_EXTRACT_PROMPT,
    LLM_DESCRIPTION_VERIFY_PROMPT,
)
from agent.prompts.per_param import (
    ALLOWED_RANGE_EXTRACT_PROMPT,
    ARRAY_LENGTH_EXTRACT_PROMPT,
    DFORMAT_EXTRACT_PROMPT,
    DTYPE_EXTRACT_PROMPT,
    OPTIONAL_EXTRACT_PROMPT,
    SHAPE_EXTRACT_PROMPT,
)
from agent.prompts.relations import RELATION_OBJECT_BUILD_PROMPT
from agent.prompts.assembly import (
    ALLOWED_RANGE_VALUE_BUILD_PROMPT,
    SHAPE_TO_DIMENSIONS_PROMPT,
)
from agent.prompts.extraction import (
    DETERMINISM_EXTRACT_PROMPT,
    DTYPE_COMBO_TABLE_PROMPT,
    DTYPE_CONSTRAINT_TEXT_PROMPT,
    RETURN_CODE_EXTRACT_PROMPT,
)

__all__ = [
    # basic_info
    "FUNCTION_SIGNATURE_EXTRACT_PROMPT",
    "PRODUCT_SUPPORT_EXTRACT_PROMPT",
    "FUNCTION_EXPLANATION_EXTRACT_PROMPT",
    # llm_description
    "LLM_DESCRIPTION_EXTRACT_PROMPT",
    "LLM_DESCRIPTION_VERIFY_PROMPT",
    # per_param
    "DTYPE_EXTRACT_PROMPT",
    "OPTIONAL_EXTRACT_PROMPT",
    "DFORMAT_EXTRACT_PROMPT",
    "SHAPE_EXTRACT_PROMPT",
    "ARRAY_LENGTH_EXTRACT_PROMPT",
    "ALLOWED_RANGE_EXTRACT_PROMPT",
    # relations
    "RELATION_OBJECT_BUILD_PROMPT",
    # assembly
    "SHAPE_TO_DIMENSIONS_PROMPT",
    "ALLOWED_RANGE_VALUE_BUILD_PROMPT",
    # extraction
    "RETURN_CODE_EXTRACT_PROMPT",
    "DETERMINISM_EXTRACT_PROMPT",
    "DTYPE_COMBO_TABLE_PROMPT",
    "DTYPE_CONSTRAINT_TEXT_PROMPT",
]
