from __future__ import annotations

from fastapi import APIRouter

from ...agent import run_design_agent
from ...schemas import DesignRequest, DesignRunResponse
from ...runtime_store import store_design_run

router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/agent/design-run", response_model=DesignRunResponse)
async def create_design_run(request: DesignRequest) -> DesignRunResponse:
    response = await run_design_agent(request)
    store_design_run(response)
    return response
