"""HTTP routes for the GeneratorAgent: generate / fetch / list test cases."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from agent.mcp_client import MCPClient
from agent.schemas.cases import (
    GenerateCasesRequest,
    GenerateCasesResponse,
    GetCasesResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["cases"])

_mcp_client = MCPClient()


def _cases_dir() -> Path:
    """Resolve the on-disk cases directory (parallel to ``agent.core.config``)."""
    return Path(__file__).resolve().parents[4] / "cases"


@router.post("/cases/generate", response_model=GenerateCasesResponse)
async def generate_cases(body: GenerateCasesRequest) -> GenerateCasesResponse:
    """Generate ``count`` test cases for ``operator_name`` from saved constraints.

    Returns the on-disk path of the generated ``{operator_name}_cases.json``.
    """
    logger.info(
        "POST /cases/generate: op=%s count=%d seed=%s",
        body.operator_name, body.count, body.seed,
    )
    try:
        constraints = await _mcp_client.get_json_constraints(body.operator_name)
        if not constraints:
            return GenerateCasesResponse(
                success=False,
                operator_name=body.operator_name,
                error=f"json_constraints not found for {body.operator_name}",
            )

        from agent.generators import TestCaseGenerator, parse_result_json
        import json

        context = parse_result_json(constraints)
        cases = TestCaseGenerator(context, seed=body.seed).generate(count=body.count)
        cases_json = json.dumps(
            [c.model_dump() for c in cases], ensure_ascii=False,
        )
        save_result = await _mcp_client.save_test_cases(
            operator_name=body.operator_name,
            cases_json=cases_json,
            source="generated",
        )
        return GenerateCasesResponse(
            success=True,
            operator_name=body.operator_name,
            cases_count=len(cases),
            output_path=save_result.get("output_path"),
        )

    except Exception as e:
        logger.exception("generate_cases failed for %s", body.operator_name)
        return GenerateCasesResponse(
            success=False,
            operator_name=body.operator_name,
            error=str(e),
        )


@router.get("/cases/{operator_name}", response_model=GetCasesResponse)
async def get_cases(operator_name: str) -> GetCasesResponse:
    """Fetch the most recent saved test cases for ``operator_name``."""
    result = await _mcp_client.get_test_cases(operator_name)
    if not result:
        return GetCasesResponse(operator_name=operator_name, found=False)

    return GetCasesResponse(
        operator_name=operator_name,
        found=True,
        cases=result.get("cases", []),
    )


@router.get("/cases/{operator_name}/download")
async def download_cases(operator_name: str) -> FileResponse:
    """Download the on-disk ``cases/{operator_name}_cases.json`` file."""
    path = _cases_dir() / f"{operator_name}_cases.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    return FileResponse(
        path=path,
        media_type="application/json",
        filename=path.name,
    )


@router.get("/cases")
async def list_case_operators(
    limit: int = Query(default=50, ge=1, le=500),
):
    """List operator names that have saved test cases (most recent first)."""
    rows = await _mcp_client.list_test_case_operators()
    return {"operators": rows[:limit]}
