from typing import Any

from pydantic import BaseModel

from shared.models.enums import ParamDirection, SectionType


class ProductSupport(BaseModel):
    product: str
    supported: bool


class FunctionSignature(BaseModel):
    return_type: str
    function_name: str
    parameters: list[str]
    raw_code: str


class ParsedSection(BaseModel):
    section_type: SectionType
    heading: str
    content: str
    line_start: int
    line_end: int
    metadata: dict[str, Any] = {}


class ParameterTableRow(BaseModel):
    cells: dict[str, str]


class ParsedOperatorDocument(BaseModel):
    operator_name: str
    cann_version: str
    source_url: str | None = None
    saved_date: str | None = None
    sections: list[ParsedSection]
    product_support: list[ProductSupport] = []
    function_signatures: list[FunctionSignature] = []


class ParsedParameter(BaseModel):
    """A single parameter extracted from a CANN operator function signature."""

    function_name: str
    param_name: str
    param_type: str = ""
    direction: ParamDirection = ParamDirection.INPUT
    description: str = ""
    usage_notes: str = ""
    data_type: str = ""
    data_format: str = ""
    shape: str = ""
    attributes: dict[str, Any] = {}
