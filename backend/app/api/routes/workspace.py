from __future__ import annotations

from fastapi import APIRouter

from ...api.errors import not_found
from ...schemas import ApiErrorPayload, WorkspaceDetail
from ...service import get_workspace_detail

router = APIRouter(prefix="/api", tags=["workspace"])


@router.get("/workspace", response_model=WorkspaceDetail)
def read_default_workspace() -> WorkspaceDetail:
    return get_workspace_detail()


@router.get(
    "/runs/{run_id}/workspace",
    response_model=WorkspaceDetail,
    responses={404: {"model": ApiErrorPayload}},
)
def read_workspace_for_run(run_id: str) -> WorkspaceDetail:
    workspace = get_workspace_detail(run_id)
    if workspace.activeRun.id != run_id:
        raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return workspace
