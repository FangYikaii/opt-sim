from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

from .models import (
    ActiveModelInfo,
    AgentConfigurationSummary,
    ArtifactDetail,
    ArtifactMetadataItem,
    ArtifactSummary,
    CandidateSolution,
    DecisionSupport,
    DesignRunResponse,
    ExportEstimate,
    RunSummary,
    WorkspaceDetail,
    WorkspaceDraft,
    WorkspaceProject,
)


@dataclass
class RuntimeRunBundle:
    workspace: WorkspaceDetail
    artifacts: list[ArtifactSummary]
    artifact_details: dict[str, ArtifactDetail]


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_RUNS_DIR = Path(
    os.getenv("OPT_SIM_RUNTIME_RUNS_DIR", ROOT / ".runtime-store" / "runs")
)

PROJECT = WorkspaceProject(
    id="project-opt-sim",
    name="Opt-Sim Structural Color Desk",
    description="Business-facing structural color planning workspace built around validated reproduction routes.",
)


def _runtime_run_path(run_id: str) -> Path:
    return RUNTIME_RUNS_DIR / f"{run_id}.json"


def _serialize_bundle(bundle: RuntimeRunBundle) -> dict[str, object]:
    return {
        "workspace": bundle.workspace.model_dump(mode="json"),
        "artifacts": [artifact.model_dump(mode="json") for artifact in bundle.artifacts],
        "artifact_details": {
            artifact_id: detail.model_dump(mode="json")
            for artifact_id, detail in bundle.artifact_details.items()
        },
    }


def _deserialize_bundle(payload: dict[str, object]) -> RuntimeRunBundle:
    artifacts_payload = payload.get("artifacts", [])
    detail_payload = payload.get("artifact_details", {})
    return RuntimeRunBundle(
        workspace=WorkspaceDetail.model_validate(payload.get("workspace", {})),
        artifacts=[ArtifactSummary.model_validate(item) for item in artifacts_payload],
        artifact_details={
            str(artifact_id): ArtifactDetail.model_validate(detail)
            for artifact_id, detail in dict(detail_payload).items()
        },
    )


def _persist_runtime_runs() -> None:
    RUNTIME_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for run_id, bundle in _runtime_runs.items():
        _runtime_run_path(run_id).write_text(
            json.dumps(_serialize_bundle(bundle), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _refresh_run_history() -> None:
    ordered_runs = [
        bundle.workspace.activeRun.model_copy(deep=True)
        for bundle in reversed(_runtime_runs.values())
    ]
    for run_id, bundle in list(_runtime_runs.items()):
        _runtime_runs[run_id] = RuntimeRunBundle(
            workspace=bundle.workspace.model_copy(
                update={"runs": [run.model_copy(deep=True) for run in ordered_runs]}
            ),
            artifacts=bundle.artifacts,
            artifact_details=bundle.artifact_details,
        )


def _load_persisted_runtime_runs() -> dict[str, RuntimeRunBundle]:
    if not RUNTIME_RUNS_DIR.exists():
        return {}

    bundles: dict[str, RuntimeRunBundle] = {}
    for path in sorted(RUNTIME_RUNS_DIR.glob("run-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            bundle = _deserialize_bundle(payload)
        except Exception:
            continue
        bundles[bundle.workspace.activeRun.id] = bundle
    return bundles


_runtime_runs: dict[str, RuntimeRunBundle] = _load_persisted_runtime_runs()
_refresh_run_history()


def _selected_candidate(candidates: list[CandidateSolution]) -> CandidateSolution | None:
    if not candidates:
        return None
    return next((candidate for candidate in candidates if candidate.selected), candidates[0])


def _candidate_metric(candidate: CandidateSolution | None, label: str, default: str = "n/a") -> str:
    if candidate is None:
        return default
    metric = next((item for item in candidate.metrics if item.label == label), None)
    return metric.value if metric is not None else default


def _build_artifacts_for_run(
    run: RunSummary,
    *,
    draft: WorkspaceDraft,
    candidates: list[CandidateSolution],
    export_estimate: ExportEstimate,
    active_model: ActiveModelInfo,
    decision_support: DecisionSupport,
) -> tuple[list[ArtifactSummary], dict[str, ArtifactDetail]]:
    selected_candidate = _selected_candidate(candidates)
    ranked_under = f"{draft.incidenceAngleValue}, {draft.polarizationValue}"
    selected_candidate_id = selected_candidate.id if selected_candidate is not None else "n/a"
    selected_source = _candidate_metric(selected_candidate, "Source")
    selected_delta_e = _candidate_metric(selected_candidate, "DeltaE")

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
            description=(
                f"Algorithm result summary and candidate review notes for {selected_candidate_id}, "
                f"ranked under {ranked_under}."
            ),
            metadata=[
                ArtifactMetadataItem(label="run_id", value=run.id),
                ArtifactMetadataItem(label="status", value=run.status),
                ArtifactMetadataItem(label="target_hex", value=draft.targetValue),
                ArtifactMetadataItem(label="ranked_under", value=ranked_under),
                ArtifactMetadataItem(label="incidence_angle", value=draft.incidenceAngleValue),
                ArtifactMetadataItem(label="polarization", value=draft.polarizationValue),
                ArtifactMetadataItem(label="recommended_candidate", value=selected_candidate_id),
                ArtifactMetadataItem(label="recommended_source", value=selected_source),
                ArtifactMetadataItem(label="recommended_delta_e", value=selected_delta_e),
                ArtifactMetadataItem(label="decision_headline", value=decision_support.headline),
                ArtifactMetadataItem(label="decision_confidence", value=decision_support.confidence),
                ArtifactMetadataItem(label="active_model", value=active_model.checkpointFile or active_model.label),
            ],
        ),
        f"{run.id}-export-plan": ArtifactDetail(
            id=f"{run.id}-export-plan",
            runId=run.id,
            name="export-plan.json",
            type="export",
            status="pending",
            description=(
                f"Preview-only export plan generated for {selected_candidate_id}, "
                f"ranked under {ranked_under}."
            ),
            metadata=[
                ArtifactMetadataItem(label="run_id", value=run.id),
                ArtifactMetadataItem(label="status", value="pending approval"),
                ArtifactMetadataItem(label="ranked_under", value=ranked_under),
                ArtifactMetadataItem(label="incidence_angle", value=draft.incidenceAngleValue),
                ArtifactMetadataItem(label="polarization", value=draft.polarizationValue),
                ArtifactMetadataItem(label="selected_candidate", value=selected_candidate_id),
                ArtifactMetadataItem(label="decision_next_action", value=decision_support.nextAction),
                ArtifactMetadataItem(label="active_model", value=active_model.checkpointFile or active_model.label),
                ArtifactMetadataItem(label="delivery_format", value=export_estimate.format),
                ArtifactMetadataItem(label="dimensions", value=export_estimate.dimensions),
            ],
        ),
    }
    return summaries, details


def list_runtime_runs() -> list[RunSummary]:
    return [bundle.workspace.activeRun for bundle in reversed(_runtime_runs.values())]


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
    artifacts, artifact_details = _build_artifacts_for_run(
        active_run,
        draft=response.draft,
        candidates=response.candidates,
        export_estimate=response.exportEstimate,
        active_model=response.activeModel,
        decision_support=response.decisionSupport,
    )
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
        activeModel=response.activeModel,
        agentConfiguration=response.agentConfiguration,
        decisionSupport=response.decisionSupport,
    )
    _runtime_runs[active_run.id] = RuntimeRunBundle(
        workspace=workspace,
        artifacts=artifacts,
        artifact_details=artifact_details,
    )
    _refresh_run_history()
    _persist_runtime_runs()
    refreshed_workspace = _runtime_runs[active_run.id].workspace
    return refreshed_workspace
