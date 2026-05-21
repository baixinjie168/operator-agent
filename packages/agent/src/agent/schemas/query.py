from pydantic import BaseModel


class OperatorSummary(BaseModel):
    name: str
    source_url: str | None = None
    latest_version: int | None = None
    created_at: str | None = None


class OperatorListResponse(BaseModel):
    operators: list[OperatorSummary]


class OperatorDetailResponse(BaseModel):
    success: bool
    operator_name: str | None = None
    version: int | None = None
    parsed_data: dict | None = None
    error: str | None = None


class ParameterItem(BaseModel):
    id: int
    operator_name: str
    version: int
    function_name: str
    param_name: str
    param_type: str = ""
    direction: str = "input"
    description: str | None = None
    usage_notes: str | None = None
    data_type: str | None = None
    data_format: str | None = None
    shape: str | None = None
    attributes: dict | None = None


class ParameterListResponse(BaseModel):
    parameters: list[ParameterItem]
