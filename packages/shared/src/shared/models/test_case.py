
from typing import Literal

from pydantic import BaseModel, ConfigDict

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

