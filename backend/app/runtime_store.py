from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import (
    ArtifactDetail,
    ArtifactMetadataItem,
    ArtifactSummary,
    DesignRunResponse,
    RunSummary,
    WorkspaceDetail,
    WorkspaceProject,
)


@dataclass
class RuntimeRunBundle:
    workspace: WorkspaceDetail
    artifacts: list[ArtifactSummary]
    artifact_details: dict[str, ArtifactDetail]


PROJECT = WorkspaceProject(
    id="project-opt-sim",
    name="Opt-Sim Structural Color Desk",
    description="Business-facing structural color planning workspace built around validated reproduction routes.",
)

_runtime_runs: dict[str, RuntimeRunBundle] = {}


def _build_artifacts_for_run(run: RunSummary) -> tuple[list[ArtifactSummary], dict[str, ArtifactDetail]]:
    summaries = [
        ArtifactSummary(
            id=f"{run.id}-report",
            name="simulation-report.md",
            type="report",
            status="ready",
        ),
        ArtifactSummary(
            id=f"{run.id}-export-plan",
            name="export-plan.json",
            type="export",
            status="pending",
        ),
    ]
    details = {
        f"{run.id}-report": ArtifactDetail(
            id=f"{run.id}-report",
            runId=run.id,
            name="simulation-report.md",
            type="report",
            status="ready",
            description="Algorithm result summary and candidate review notes for the live demo run.",
            metadata=[
                ArtifactMetadataItem(label="run_id", value=run.id),
                ArtifactMetadataItem(label="status", value=run.status),
            ],
        ),
        f"{run.id}-export-plan": ArtifactDetail(
            id=f"{run.id}-export-plan",
            runId=run.id,
            name="export-plan.json",
            type="export",
            status="pending",
            description="Preview-only export plan generated from the selected candidate.",
            metadata=[
                ArtifactMetadataItem(label="run_id", value=run.id),
                ArtifactMetadataItem(label="status", value="pending approval"),
            ],
        ),
    }
    return summaries, details


def list_runtime_runs() -> list[RunSummary]:
    return [bundle.workspace.activeRun for bundle in _runtime_runs.values()]


def get_runtime_workspace(run_id: str) -> WorkspaceDetail | None:
    bundle = _runtime_runs.get(run_id)
    return bundle.workspace if bundle else None


def get_runtime_artifacts(run_id: str) -> list[ArtifactSummary] | None:
    bundle = _runtime_runs.get(run_id)
    return bundle.artifacts if bundle else None


def get_runtime_artifact_detail(artifact_id: str) -> ArtifactDetail | None:
    for bundle in _runtime_runs.values():
      if artifact_id in bundle.artifact_details:
        return bundle.artifact_details[artifact_id]
    return None


def store_design_run(response: DesignRunResponse) -> WorkspaceDetail:
    timestamp = datetime.now().strftime("%H:%M:%S")
    active_run = response.activeRun.model_copy(update={"updatedAt": timestamp})
    artifacts, artifact_details = _build_artifacts_for_run(active_run)
    workspace = WorkspaceDetail(
        project=PROJECT,
        activeRun=active_run,
        draft=response.draft,
        runs=[active_run] + list_runtime_runs(),
        targets=response.targets,
        artifacts=artifacts,
        timeline=response.timeline,
        candidates=response.candidates,
        constraints=response.constraints,
        exportEstimate=response.exportEstimate,
    )
    _runtime_runs[active_run.id] = RuntimeRunBundle(
        workspace=workspace,
        artifacts=artifacts,
        artifact_details=artifact_details,
    )
    return workspace
