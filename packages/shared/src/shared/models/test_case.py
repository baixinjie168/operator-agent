
from pydantic import BaseModel

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
