from enum import StrEnum


class DataType(StrEnum):
    FLOAT32 = "FLOAT32"
    FLOAT16 = "FLOAT16"
    BFLOAT16 = "BFLOAT16"
    INT8 = "INT8"
    UINT8 = "UINT8"
    INT32 = "INT32"
    INT64 = "INT64"
    BOOL = "BOOL"


class DataFormat(StrEnum):
    ND = "ND"
    NC = "NC"
    NCL = "NCL"
    NCHW = "NCHW"
    NCDHW = "NCDHW"
    NHWC = "NHWC"


class ParamDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class SectionType(StrEnum):
    TITLE = "title"
    PRODUCT_SUPPORT = "product_support"
    FUNCTION_DESCRIPTION = "function_description"
    FUNCTION_PROTOTYPE = "function_prototype"
    GET_WORKSPACE_SIZE = "get_workspace_size"
    EXECUTE_API = "execute_api"
    CONSTRAINTS = "constraints"
    USAGE_EXAMPLE = "usage_example"
    PARAMS_GET_WORKSPACE = "params_get_workspace"
    RETURN_CODES_GET_WORKSPACE = "return_codes_get_workspace"
    PARAMS_EXECUTE = "params_execute"
    RETURN_CODES_EXECUTE = "return_codes_execute"
    UNKNOWN = "unknown"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class TaskComplexity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LLMProvider(StrEnum):
    ZAI = "zai"
    DEEPSEEK = "deepseek"


class TestCaseCategory(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"


class PipelineStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class RelationType(StrEnum):
    DTYPE = "dtype"
    SHAPE = "shape"
    DFORMAT = "dformat"
    VALUE = "value"
    DTYPE_SHAPE = "dtype&shape"
    DTYPE_DFORMAT = "dtype&dformat"
    DFORMAT_SHAPE = "dformat&shape"
    DTYPE_DFORMAT_SHAPE = "dtype&dformat&shape"
    PRESENCE = "presence"
    SHAPE_VALUE = "shape&value"


class GeneratorParamKind(StrEnum):
    """Whether a parameter is a tensor or a scalar attribute."""

    TENSOR = "tensor"
    ATTRS = "attrs"


class ConstraintExprType(StrEnum):
    """expr_type values in constraints_in_parameters entries."""

    SHAPE_EQUALITY = "shape_equality"
    SHAPE_UNIFICATION = "shape_unification"
    FIXED_VALUE = "fixed_value"
    TYPE_DEPENDENCY = "type_dependency"
    SHAPE_CHOICE = "shape_choice"
