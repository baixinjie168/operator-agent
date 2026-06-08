
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.models.enums import DataFormat, DataType


class TensorSpec(BaseModel):
    shape: list[int]
    dtype: DataType
    format: DataFormat
    data_values: list[float] | None = None
    is_empty: bool = False


class TestCase(BaseModel):
    name: str
    category: str  # "positive" | "negative" | "boundary"
    description: str
    inputs: dict[str, TensorSpec]
    scalar_params: dict[str, float | int]
    expected_status: str
    violated_constraint: str | None = None


class TestFile(BaseModel):
    operator_name: str
    imports: list[str]
    helper_functions: list[str]
    test_cases: list[TestCase]
    generated_code: str
    constraint_coverage: dict[str, float]


# ── Generator output models (immutable, generated from result.json) ────────────


class TensorInputSpec(BaseModel):
    """Single tensor/attr input entry inside a generated test case."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: Literal["tensor", "attrs"]
    required: bool
    dtype: str  # pytorch dtype string, e.g. "float32"
    shape: list[int] | None = None
    range_values: list[float] | list[int] | bool | float | int | None = None
    backward: bool = False
    align_32B: int | None = None
    outlier_values: list[float] | None = None


class StandardSpec(BaseModel):
    """Tolerance/standard block embedded in a generated test case."""

    model_config = ConfigDict(frozen=True)

    acc: dict[str, float] = Field(default_factory=dict)
    perf: list[float] = Field(default_factory=lambda: [0.95, 0.95])


class TestCaseRecord(BaseModel):
    """Single generated test case record — matches the legacy aclnnAdaLayerNorm_cases.json format."""

    # Tell pytest not to try to collect this Pydantic model as a test class.
    __test__ = False
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    aclnn_name: str
    triton_name: str | None = None
    version: str = "v1.0"
    expected_error_msg: str | None = None
    api: str = "pytorch"
    api_type: str
    aclnn_api_type: str
    triton_api_type: str = "triton_function"
    fusion_api_type: str = "fusion_function"
    fusion_mode: str | None = None
    dist_api_type: str = "dist_function"
    backward: bool = False
    standard: StandardSpec
    outputs: list[dict] | None = None
    inputs: list[TensorInputSpec]


class GeneratorContext(BaseModel):
    """Intermediate, fully-typed representation derived from result.json.

    The generator modules operate on this context rather than the raw
    nested ``result.json`` structure.
    """

    model_config = ConfigDict(frozen=True)

    operator_name: str
    aclnn_name: str
    supported_platforms: list[str]
    inputs: dict[str, dict[str, dict]]  # {param: {platform: constraint}}
    outputs: dict[str, dict[str, dict]]
    constraints_in_parameters: dict[str, list[dict]] = Field(default_factory=dict)
    dtype_support: dict[str, list[dict]] = Field(default_factory=dict)

