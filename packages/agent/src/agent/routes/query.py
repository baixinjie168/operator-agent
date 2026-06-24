"""Query routes: list operators, retrieve parsed documents, and query parameters."""

from __future__ import annotations

from fastapi import APIRouter, Query

from agent.mcp_client import MCPClient
from agent.schemas.query import (
    ConstraintsResultResponse,
    DeterminismListResponse,
    DocumentContentResponse,
    DtypeComboListResponse,
    DtypeComboResponse,
    FunctionSignatureListResponse,
    JsonConstraintsResponse,
    OperatorDetailResponse,
    OperatorListResponse,
    ParameterListResponse,
    ParamRelationListResponse,
    PlatformSupportListResponse,
    ReturnCodeListResponse,
    UpdateJsonConstraintsRequest,
    UpdateJsonConstraintsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["query"])

_mcp_client = MCPClient()


@router.get("/operators", response_model=OperatorListResponse)
async def list_operators() -> OperatorListResponse:
    """List all registered operators with their latest version."""
    result = await _mcp_client.list_operators()
    operators = [
        {
            "name": item["name"],
            "source_url": item.get("source_url"),
            "latest_version": item.get("latest_version"),
            "created_at": item.get("created_at"),
        }
        for item in result
    ]
    return OperatorListResponse(operators=operators)


@router.get("/operators/{operator_name}", response_model=OperatorDetailResponse)
async def get_operator(operator_name: str, version: int | None = None) -> OperatorDetailResponse:
    """Retrieve a parsed operator document by name and optional version."""
    result = await _mcp_client.get_parsed(operator_name, version)
    if result is None:
        return OperatorDetailResponse(success=False, error=f"Operator '{operator_name}' not found")
    return OperatorDetailResponse(
        success=True,
        operator_name=result.get("operator_name"),
        version=version,
        parsed_data=result,
    )


@router.get("/operators/{operator_name}/document", response_model=DocumentContentResponse)
async def get_operator_document(
    operator_name: str, version: int | None = None,
) -> DocumentContentResponse:
    """Retrieve raw Markdown content of the operator document."""
    try:
        result = await _mcp_client.get_document_content(operator_name, version)
        if result is None:
            return DocumentContentResponse(
                success=False, error=f"Document for operator '{operator_name}' not found",
            )
        return DocumentContentResponse(
            success=True,
            operator_name=result.get("operator_name", operator_name),
            version=result.get("version"),
            content=result.get("content"),
        )
    except Exception as e:
        return DocumentContentResponse(success=False, error=str(e))


@router.get("/parameters", response_model=ParameterListResponse)
async def list_parameters(operator_name: str | None = Query(default=None)) -> ParameterListResponse:
    """Query parameters, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_parameters(operator_name)
        return ParameterListResponse(parameters=result)
    except Exception:
        return ParameterListResponse(parameters=[])


@router.get("/relations", response_model=ParamRelationListResponse)
async def list_relations(operator_name: str | None = Query(default=None)) -> ParamRelationListResponse:
    """Query parameter relations, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_param_relations_by_operator(operator_name)
        return ParamRelationListResponse(relations=result)
    except Exception:
        return ParamRelationListResponse(relations=[])


@router.get("/signatures", response_model=FunctionSignatureListResponse)
async def list_signatures(operator_name: str | None = Query(default=None)) -> FunctionSignatureListResponse:
    """Query function signatures, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_function_signatures_by_operator(operator_name)
        return FunctionSignatureListResponse(signatures=result)
    except Exception:
        return FunctionSignatureListResponse(signatures=[])


@router.get("/platforms", response_model=PlatformSupportListResponse)
async def list_platforms(operator_name: str | None = Query(default=None)) -> PlatformSupportListResponse:
    """Query platform support info, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_platform_support_by_operator(operator_name)
        return PlatformSupportListResponse(platforms=result)
    except Exception:
        return PlatformSupportListResponse(platforms=[])


@router.get("/return-codes", response_model=ReturnCodeListResponse)
async def list_return_codes(operator_name: str | None = Query(default=None)) -> ReturnCodeListResponse:
    """Query return codes, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_return_codes_by_operator(operator_name)
        return ReturnCodeListResponse(return_codes=result)
    except Exception:
        return ReturnCodeListResponse(return_codes=[])


@router.get("/determinism", response_model=DeterminismListResponse)
async def list_determinism(operator_name: str | None = Query(default=None)) -> DeterminismListResponse:
    """Query determinism records, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_determinism_by_operator(operator_name)
        return DeterminismListResponse(determinism=result)
    except Exception:
        return DeterminismListResponse(determinism=[])


@router.get("/dtype-combos", response_model=DtypeComboResponse)
async def list_dtype_combos(
    operator_name: str = Query(...),
    function_name: str | None = Query(default=None),
) -> DtypeComboResponse:
    """Query dtype combination records for a specific operator."""
    try:
        rows = await _mcp_client.query_dtype_combos_by_operator(operator_name)
        if function_name:
            rows = [r for r in rows if r.get("function_name") == function_name]

        # Group by platform
        from collections import defaultdict
        combos: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            combos[r["platform"]].append(r["combo"])

        fn = function_name or (rows[0]["function_name"] if rows else None)
        return DtypeComboResponse(
            operator_name=operator_name,
            function_name=fn,
            combos=dict(combos),
        )
    except Exception:
        return DtypeComboResponse(operator_name=operator_name, combos={})


@router.get("/dtype-combos-list", response_model=DtypeComboListResponse)
async def list_dtype_combos_flat(
    operator_name: str | None = Query(default=None),
) -> DtypeComboListResponse:
    """Query dtype combination records as a flat list, optionally filtered by operator name."""
    try:
        rows = await _mcp_client.query_dtype_combos_by_operator(operator_name)
        return DtypeComboListResponse(dtype_combos=rows)
    except Exception:
        return DtypeComboListResponse(dtype_combos=[])


@router.get("/constraints-result", response_model=ConstraintsResultResponse)
async def list_constraints_result(
    operator_name: str | None = Query(default=None),
) -> ConstraintsResultResponse:
    """Query assembled constraints results, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_constraints_result(operator_name)
        return ConstraintsResultResponse(results=result)
    except Exception:
        return ConstraintsResultResponse(results=[])


@router.get("/json-constraints", response_model=JsonConstraintsResponse)
async def get_json_constraints(
    operator_name: str = Query(...),
) -> JsonConstraintsResponse:
    """Retrieve json_constraints from the latest document version for an operator."""
    try:
        result = await _mcp_client.get_json_constraints(operator_name)
        if result is None:
            return JsonConstraintsResponse(success=False, error="No json_constraints found")
        return JsonConstraintsResponse(success=True, operator_name=operator_name, json_constraints=result)
    except Exception as e:
        return JsonConstraintsResponse(success=False, error=str(e))


@router.post("/json-constraints", response_model=UpdateJsonConstraintsResponse)
async def update_json_constraints(
    body: UpdateJsonConstraintsRequest,
) -> UpdateJsonConstraintsResponse:
    """Update json_constraints for the latest document version of an operator."""
    try:
        result = await _mcp_client.update_json_constraints_by_name(
            body.operator_name, body.json_constraints,
        )
        if not result.get("saved"):
            return UpdateJsonConstraintsResponse(
                success=False,
                operator_name=body.operator_name,
                error=result.get("error", "Save failed"),
            )
        return UpdateJsonConstraintsResponse(
            success=True,
            operator_name=body.operator_name,
            doc_id=result.get("doc_id"),
        )
    except Exception as e:
        return UpdateJsonConstraintsResponse(
            success=False, operator_name=body.operator_name, error=str(e),
        )
