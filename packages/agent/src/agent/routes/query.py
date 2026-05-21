"""Query routes: list operators, retrieve parsed documents, and query parameters."""

from __future__ import annotations

from fastapi import APIRouter, Query

from agent.mcp_client import MCPClient
from agent.schemas.query import (
    OperatorDetailResponse,
    OperatorListResponse,
    ParameterListResponse,
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


@router.get("/parameters", response_model=ParameterListResponse)
async def list_parameters(operator_name: str | None = Query(default=None)) -> ParameterListResponse:
    """Query parameters, optionally filtered by operator name."""
    try:
        result = await _mcp_client.query_parameters(operator_name)
        return ParameterListResponse(parameters=result)
    except Exception:
        return ParameterListResponse(parameters=[])
