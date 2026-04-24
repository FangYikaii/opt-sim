from __future__ import annotations

from .algorithm_overview import get_algorithm_overview as build_algorithm_overview
from .models import (
    AlgorithmOverview,
    ArtifactDetail,
    ArtifactSummary,
    CandidateSolution,
    ConstraintCheck,
    ExportEstimate,
    RunSummary,
    TargetAsset,
    TimelineEvent,
    WorkspaceDetail,
    WorkspaceDraft,
    WorkspaceProject,
)
from .runtime_store import (
    PROJECT,
    get_runtime_artifact_detail,
    get_runtime_artifacts,
    get_runtime_workspace,
    list_runtime_runs,
)


def get_algorithm_overview() -> AlgorithmOverview:
    return build_algorithm_overview()


def get_project() -> WorkspaceProject:
    return PROJECT


def list_runs() -> list[RunSummary]:
    return list_runtime_runs()


def get_run_summary(run_id: str) -> RunSummary | None:
    workspace = get_runtime_workspace(run_id)
    if workspace is None:
        return None
    return workspace.activeRun


def get_run_draft(run_id: str) -> WorkspaceDraft | None:
    workspace = get_runtime_workspace(run_id)
    if workspace is None:
        return None
    return workspace.draft


def list_targets_for_run(run_id: str) -> list[TargetAsset] | None:
    workspace = get_runtime_workspace(run_id)
    if workspace is None:
        return None
    return workspace.targets


def list_timeline_for_run(run_id: str) -> list[TimelineEvent] | None:
    workspace = get_runtime_workspace(run_id)
    if workspace is None:
        return None
    return workspace.timeline


def list_candidates_for_run(run_id: str) -> list[CandidateSolution] | None:
    workspace = get_runtime_workspace(run_id)
    if workspace is None:
        return None
    return workspace.candidates


def list_constraints_for_run(run_id: str) -> list[ConstraintCheck] | None:
    workspace = get_runtime_workspace(run_id)
    if workspace is None:
        return None
    return workspace.constraints


def list_artifacts_for_run(run_id: str) -> list[ArtifactSummary] | None:
    return get_runtime_artifacts(run_id)


def get_export_estimate_for_run(run_id: str) -> ExportEstimate | None:
    workspace = get_runtime_workspace(run_id)
    if workspace is None:
        return None
    return workspace.exportEstimate


def get_workspace_detail(run_id: str | None = None) -> WorkspaceDetail:
    if run_id:
        workspace = get_runtime_workspace(run_id)
        if workspace is not None:
            return workspace

    runs = list_runtime_runs()
    if not runs:
        raise KeyError("No runtime runs available")

    workspace = get_runtime_workspace(runs[0].id)
    if workspace is None:
        raise KeyError("Default runtime workspace missing")
    return workspace


def get_artifact_detail(artifact_id: str) -> ArtifactDetail | None:
    return get_runtime_artifact_detail(artifact_id)
