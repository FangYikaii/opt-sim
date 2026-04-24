from __future__ import annotations

from fastapi import APIRouter

from ...api.errors import not_found
from ...schemas import ApiErrorPayload, RunSummary, TargetAsset, WorkspaceDraft
from ...service import (
    get_run_draft,
    get_run_summary,
    list_runs,
    list_targets_for_run,
)

router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs", response_model=list[RunSummary])
def read_runs() -> list[RunSummary]:
    return list_runs()


@router.get(
    "/runs/{run_id}",
    response_model=RunSummary,
    responses={404: {"model": ApiErrorPayload}},
)
def read_run(run_id: str) -> RunSummary:
    run = get_run_summary(run_id)
    if run is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return run


@router.get(
    "/runs/{run_id}/draft",
    response_model=WorkspaceDraft,
    responses={404: {"model": ApiErrorPayload}},
)
def read_run_draft(run_id: str) -> WorkspaceDraft:
    draft = get_run_draft(run_id)
    if draft is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return draft


@router.get(
    "/runs/{run_id}/targets",
    response_model=list[TargetAsset],
    responses={404: {"model": ApiErrorPayload}},
)
def read_run_targets(run_id: str) -> list[TargetAsset]:
    targets = list_targets_for_run(run_id)
    if targets is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return targets
