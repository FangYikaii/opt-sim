from __future__ import annotations

from fastapi import APIRouter

from ...api.errors import not_found
from ...schemas import (
    ApiErrorPayload,
    CandidateSolution,
    ConstraintCheck,
    ExportEstimate,
    TimelineEvent,
)
from ...service import (
    get_export_estimate_for_run,
    list_candidates_for_run,
    list_constraints_for_run,
    list_timeline_for_run,
)

router = APIRouter(prefix="/api", tags=["review"])


@router.get(
    "/runs/{run_id}/timeline",
    response_model=list[TimelineEvent],
    responses={404: {"model": ApiErrorPayload}},
)
def read_run_timeline(run_id: str) -> list[TimelineEvent]:
    timeline = list_timeline_for_run(run_id)
    if timeline is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return timeline


@router.get(
    "/runs/{run_id}/candidates",
    response_model=list[CandidateSolution],
    responses={404: {"model": ApiErrorPayload}},
)
def read_run_candidates(run_id: str) -> list[CandidateSolution]:
    candidates = list_candidates_for_run(run_id)
    if candidates is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return candidates


@router.get(
    "/runs/{run_id}/constraints",
    response_model=list[ConstraintCheck],
    responses={404: {"model": ApiErrorPayload}},
)
def read_run_constraints(run_id: str) -> list[ConstraintCheck]:
    constraints = list_constraints_for_run(run_id)
    if constraints is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return constraints


@router.get(
    "/runs/{run_id}/export-estimate",
    response_model=ExportEstimate,
    responses={404: {"model": ApiErrorPayload}},
)
def read_run_export_estimate(run_id: str) -> ExportEstimate:
    estimate = get_export_estimate_for_run(run_id)
    if estimate is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return estimate
