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
    src_content: str = ""
    description: str | None = None
    data_type: str | None = None
    data_format: str | None = None
    shape: str | None = None
    is_optional: int = 0
    is_support_discontinuous: str = '{"value":"N/A","src_text":""}'
    array_length: str = "N/A"
    param_desc: str = ""
    allowed_range_value: str = "[]"
    attributes: dict | None = None


class ParameterListResponse(BaseModel):
    parameters: list[ParameterItem]


class ParamRelationItem(BaseModel):
    id: int
    operator_name: str
    version: int
    function_name: str
    relation_type: str
    precondition: str = "无"
    description: str
    params: list[str]
    param_optional: dict[str, bool] = {}
    source_citation: str


class ParamRelationListResponse(BaseModel):
    relations: list[ParamRelationItem]


class FunctionSignatureItem(BaseModel):
    id: int
    operator_name: str
    version: int
    function_name: str
    return_type: str = ""
    parameters: list[dict] = []
    full_signature: str = ""
    raw_code: str = ""


class FunctionSignatureListResponse(BaseModel):
    signatures: list[FunctionSignatureItem]


class PlatformSupportItem(BaseModel):
    id: int
    operator_name: str
    version: int
    platform_name: str
    is_supported: int
    deterministic_computing: dict = {"value": "", "src_text": ""}


class PlatformSupportListResponse(BaseModel):
    platforms: list[PlatformSupportItem]


class ReturnCodeItem(BaseModel):
    id: int
    operator_name: str
    version: int
    function_name: str
    return_value: str
    error_code: int
    descriptions: list[str] = []
    source_citation: str = ""


class ReturnCodeListResponse(BaseModel):
    return_codes: list[ReturnCodeItem]


class DeterminismItem(BaseModel):
    id: int
    operator_name: str
    version: int
    product: str
    value: bool
    src_text: str = ""


class DeterminismListResponse(BaseModel):
    determinism: list[DeterminismItem]


class DtypeComboResponse(BaseModel):
    operator_name: str
    function_name: str | None = None
    combos: dict[str, list[dict]] = {}


class DtypeComboItem(BaseModel):
    id: int
    operator_name: str
    version: int
    function_name: str
    platform: str
    combo: dict = {}


class DtypeComboListResponse(BaseModel):
    dtype_combos: list[DtypeComboItem]


class ConstraintsResultItem(BaseModel):
    id: int
    doc_id: int
    operator_name: str
    version: int
    product_support: list = []
    platform_support: list[str] = []
    function_explanation: dict = {}


class ConstraintsResultResponse(BaseModel):
    results: list[ConstraintsResultItem]
