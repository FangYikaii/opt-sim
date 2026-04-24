from __future__ import annotations

from fastapi import APIRouter

from ...api.errors import not_found
from ...schemas import ApiErrorPayload, ArtifactDetail, ArtifactSummary
from ...service import get_artifact_detail, list_artifacts_for_run

router = APIRouter(prefix="/api", tags=["artifacts"])


@router.get(
    "/runs/{run_id}/artifacts",
    response_model=list[ArtifactSummary],
    responses={404: {"model": ApiErrorPayload}},
)
def read_run_artifacts(run_id: str) -> list[ArtifactSummary]:
    artifacts = list_artifacts_for_run(run_id)
    if artifacts is None:
      raise not_found("RUN_NOT_FOUND", f"Run not found: {run_id}")
    return artifacts


@router.get(
    "/artifacts/{artifact_id}",
    response_model=ArtifactDetail,
    responses={404: {"model": ApiErrorPayload}},
)
def read_artifact(artifact_id: str) -> ArtifactDetail:
    artifact = get_artifact_detail(artifact_id)
    if artifact is None:
      raise not_found("ARTIFACT_NOT_FOUND", f"Artifact not found: {artifact_id}")
    return artifact
