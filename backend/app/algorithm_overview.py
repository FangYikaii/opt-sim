from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .config import get_agent_settings
from .schemas import (
    ActiveModelInfo,
    AlgorithmExperiment,
    AlgorithmHeadlineMetric,
    AlgorithmOperationStep,
    AlgorithmOverview,
    AlgorithmTargetComparison,
    AgentConfigurationSummary,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"
GUIDE_DOCUMENT_PATH = ROOT / "docs" / "planning" / "algorithm-operations-guide.md"


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(experiment_dir: Path, candidate_name: Any) -> Path | None:
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        return None
    candidate_path = (experiment_dir / candidate_name).resolve()
    try:
        candidate_path.relative_to(experiment_dir.resolve())
    except ValueError:
        return None
    if candidate_path.exists():
        return candidate_path
    return None


def resolve_selected_checkpoint_path(experiment_id: str) -> Path | None:
    experiment_dir = ARTIFACTS_DIR / experiment_id
    metrics = _safe_load_json(experiment_dir / "metrics.json") or {}
    artifacts = metrics.get("artifacts") or {}

    for candidate_name in (
        artifacts.get("selected_checkpoint"),
        artifacts.get("best_available_checkpoint"),
        "generator_checkpoint_best.pt",
        artifacts.get("final_checkpoint"),
        "generator_checkpoint.pt",
    ):
        resolved = _resolve_artifact_path(experiment_dir, candidate_name)
        if resolved is not None:
            return resolved

    return None


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _stage_for_experiment(
    *,
    mean_best_delta_e: float | None,
    d2_within_5nm_ratio: float | None,
    epochs: int | None,
    samples_per_lab: int | None,
    checkpoint_ready: bool,
) -> tuple[str, str]:
    if not checkpoint_ready:
        return "No checkpoint", "fail"
    if mean_best_delta_e is None or d2_within_5nm_ratio is None:
        return "Metrics incomplete", "warning"
    if mean_best_delta_e <= 1.0 and d2_within_5nm_ratio >= 0.9 and (samples_per_lab or 0) >= 1000:
        return "Near paper target", "pass"
    if (epochs or 0) <= 10 and (samples_per_lab or 0) <= 32:
        return "Smoke run", "warning"
    if mean_best_delta_e <= 5.0:
        return "Promising", "warning"
    return "Needs full training", "fail"


def _runtime_probe() -> tuple[dict[str, Any] | None, str]:
    try:
        from .algorithms.cgan import get_torch_runtime_info

        info = get_torch_runtime_info()
    except Exception as exc:
        return None, (
            "The current interpreter could not import the torch runtime stack. "
            "Activate the `opt_sim` Conda environment before starting the backend. "
            f"Details: {exc}"
        )

    if info.get("device") == "cuda":
        device_name = str(info.get("device_name", "unknown GPU"))
        return info, f"Current backend environment detects CUDA and prefers GPU training on {device_name}."
    return info, "Current backend environment falls back to CPU. Verify the `opt_sim` environment if you expect CUDA."


def _summarize_experiment(experiment: AlgorithmExperiment) -> str:
    if not experiment.checkpointReady:
        return f"{experiment.id} has no generator checkpoint yet."
    if experiment.meanBestDeltaE is None:
        return (
            f"{experiment.id} already wrote a checkpoint, but the paper-style reproduction metrics are still missing."
        )

    paper_target_value = (
        f"{experiment.paperTargetMeanBestDeltaE:.2f}"
        if experiment.paperTargetMeanBestDeltaE is not None
        else "n/a"
    )
    return (
        f"{experiment.id} ran on {experiment.device.upper()}"
        f"{f' ({experiment.deviceName})' if experiment.deviceName else ''}. "
        f"Stage={experiment.stage}. Mean best DeltaE={experiment.meanBestDeltaE:.2f} "
        f"vs paper reference {paper_target_value}. "
        f"d2 within 5 nm={((experiment.d2Within5nmRatio or 0.0) * 100):.1f}%. "
        f"cGAN beat retrieval on {experiment.cganBeatsRetrievalCount}/{experiment.totalTargetsCompared} demo targets. "
        f"Updated {experiment.updatedAt}."
    )


def _build_experiment(path: Path) -> tuple[AlgorithmExperiment, float]:
    metrics_path = path / "metrics.json"
    checkpoint_path = resolve_selected_checkpoint_path(path.name)

    metrics = _safe_load_json(metrics_path) or {}
    runtime = metrics.get("runtime", {})
    training = metrics.get("training", {})
    reproduction = metrics.get("paper_reproduction", {}) or {}
    paper_targets = metrics.get("paper_targets", {}) or {}
    targets = metrics.get("targets", []) or []

    retrieval_scores = [
        float(target["retrieval_best_delta_e"])
        for target in targets
        if target.get("retrieval_best_delta_e") is not None
    ]
    cgan_scores = [
        float(target["cgan_best_delta_e"])
        for target in targets
        if target.get("cgan_best_delta_e") is not None
    ]
    cgan_beats_retrieval = sum(
        1
        for target in targets
        if float(target.get("cgan_best_delta_e", float("inf")))
        < float(target.get("retrieval_best_delta_e", float("inf")))
    )

    updated_timestamp = max(
        candidate.stat().st_mtime
        for candidate in [metrics_path, checkpoint_path]
        if candidate is not None and candidate.exists()
    )
    mean_best_delta_e = reproduction.get("mean_best_delta_e")
    d2_within_5nm_ratio = reproduction.get("d2_ground_truth_within_5nm_ratio")
    samples_per_lab = reproduction.get("samples_per_lab")

    stage, stage_state = _stage_for_experiment(
        mean_best_delta_e=float(mean_best_delta_e) if mean_best_delta_e is not None else None,
        d2_within_5nm_ratio=float(d2_within_5nm_ratio) if d2_within_5nm_ratio is not None else None,
        epochs=int(training["epochs"]) if training.get("epochs") is not None else None,
        samples_per_lab=int(samples_per_lab) if samples_per_lab is not None else None,
        checkpoint_ready=checkpoint_path is not None,
    )

    experiment = AlgorithmExperiment(
        id=path.name,
        updatedAt=_format_timestamp(updated_timestamp),
        stage=stage,
        stageState=stage_state,
        epochs=int(training["epochs"]) if training.get("epochs") is not None else None,
        regressorEpochs=(
            int(training["regressor_epochs"]) if training.get("regressor_epochs") is not None else None
        ),
        batchSize=int(training["batch_size"]) if training.get("batch_size") is not None else None,
        sampleCount=int(samples_per_lab) if samples_per_lab is not None else None,
        device=str(runtime.get("device", training.get("device", "unknown"))),
        deviceName=str(runtime["device_name"]) if runtime.get("device_name") is not None else None,
        checkpointReady=checkpoint_path is not None,
        lossHistoryReady=(path / "loss_history.csv").exists(),
        metricsReady=metrics_path.exists(),
        meanBestDeltaE=float(mean_best_delta_e) if mean_best_delta_e is not None else None,
        paperTargetMeanBestDeltaE=(
            float(paper_targets["generator_mean_best_delta_e_with_1000_z"])
            if paper_targets.get("generator_mean_best_delta_e_with_1000_z") is not None
            else None
        ),
        d2Within5nmRatio=float(d2_within_5nm_ratio) if d2_within_5nm_ratio is not None else None,
        paperTargetD2Within5nmRatio=(
            float(paper_targets["d2_ground_truth_within_5nm_ratio"])
            if paper_targets.get("d2_ground_truth_within_5nm_ratio") is not None
            else None
        ),
        averageRetrievalDeltaE=mean(retrieval_scores) if retrieval_scores else None,
        averageCganDeltaE=mean(cgan_scores) if cgan_scores else None,
        cganBeatsRetrievalCount=cgan_beats_retrieval,
        totalTargetsCompared=len(targets),
        summary="",
    )
    experiment.summary = _summarize_experiment(experiment)
    return experiment, updated_timestamp


def _list_experiments() -> list[tuple[AlgorithmExperiment, float]]:
    experiments: list[tuple[AlgorithmExperiment, float]] = []
    for path in sorted(ARTIFACTS_DIR.glob("cgan_reproduction*")):
        if not path.is_dir():
            continue
        if not (path / "metrics.json").exists():
            continue
        experiments.append(_build_experiment(path))
    return experiments


def _pick_best_experiment(experiments: list[AlgorithmExperiment]) -> AlgorithmExperiment | None:
    if not experiments:
        return None
    with_metrics = [experiment for experiment in experiments if experiment.meanBestDeltaE is not None]
    if with_metrics:
        return min(with_metrics, key=lambda experiment: float(experiment.meanBestDeltaE or float("inf")))
    return experiments[0]


def _headline_metrics(
    *,
    best_experiment: AlgorithmExperiment | None,
    latest_experiment: AlgorithmExperiment | None,
) -> list[AlgorithmHeadlineMetric]:
    if best_experiment is None or latest_experiment is None:
        return []

    metrics: list[AlgorithmHeadlineMetric] = []

    if best_experiment.meanBestDeltaE is not None:
        paper_reference = (
            f"{best_experiment.paperTargetMeanBestDeltaE:.2f}"
            if best_experiment.paperTargetMeanBestDeltaE is not None
            else "n/a"
        )
        metrics.append(
            AlgorithmHeadlineMetric(
                label="Best Reproduction DeltaE",
                value=f"{best_experiment.meanBestDeltaE:.2f}",
                detail=f"Best current experiment is `{best_experiment.id}`. Paper reference is {paper_reference}.",
                state="pass" if best_experiment.meanBestDeltaE <= 1.0 else "fail",
            )
        )

    if best_experiment.d2Within5nmRatio is not None:
        paper_ratio = (
            f"{best_experiment.paperTargetD2Within5nmRatio * 100:.1f}%"
            if best_experiment.paperTargetD2Within5nmRatio is not None
            else "n/a"
        )
        metrics.append(
            AlgorithmHeadlineMetric(
                label="d2 Within 5 nm",
                value=f"{best_experiment.d2Within5nmRatio * 100:.1f}%",
                detail=f"Thickness recovery accuracy. Paper reference is {paper_ratio}.",
                state="pass" if best_experiment.d2Within5nmRatio >= 0.9 else "fail",
            )
        )

    metrics.append(
        AlgorithmHeadlineMetric(
            label="cGAN vs Retrieval",
            value=f"{best_experiment.cganBeatsRetrievalCount}/{best_experiment.totalTargetsCompared}",
            detail="How many demo targets were better with cGAN-generated candidates than nearest retrieval.",
            state="pass" if best_experiment.cganBeatsRetrievalCount > 0 else "warning",
        )
    )

    training_device = (
        latest_experiment.device if latest_experiment.device != "unknown" else best_experiment.device
    )
    training_device_name = latest_experiment.deviceName or best_experiment.deviceName
    metrics.append(
        AlgorithmHeadlineMetric(
            label="Latest Training Device",
            value=training_device.upper(),
            detail=training_device_name or "Device name unavailable.",
            state="pass" if training_device == "cuda" else "warning",
        )
    )

    return metrics


def _target_comparisons(best_experiment: AlgorithmExperiment | None) -> list[AlgorithmTargetComparison]:
    if best_experiment is None:
        return []

    metrics = _safe_load_json(ARTIFACTS_DIR / best_experiment.id / "metrics.json") or {}
    targets = metrics.get("targets", []) or []
    comparisons: list[AlgorithmTargetComparison] = []
    for target in targets:
        retrieval = float(target["retrieval_best_delta_e"])
        cgan = float(target["cgan_best_delta_e"])
        if cgan < retrieval:
            winner = "cgan"
        elif cgan > retrieval:
            winner = "retrieval"
        else:
            winner = "tie"
        comparisons.append(
            AlgorithmTargetComparison(
                targetHex=str(target["target_hex"]),
                retrievalDeltaE=retrieval,
                cganDeltaE=cgan,
                winner=winner,
            )
        )
    return comparisons


def _build_active_model_info(best_experiment: AlgorithmExperiment | None) -> ActiveModelInfo | None:
    if best_experiment is None:
        return None

    checkpoint_path = resolve_selected_checkpoint_path(best_experiment.id)
    if checkpoint_path is None:
        return ActiveModelInfo(
            status="fallback",
            source="runtime-fallback",
            label="Runtime lightweight cGAN",
            experimentId=best_experiment.id,
            summary="No persisted checkpoint was resolved, so inverse design will fall back to runtime lightweight cGAN sampling.",
            meanBestDeltaE=best_experiment.meanBestDeltaE,
            updatedAt=best_experiment.updatedAt,
        )

    metrics = _safe_load_json(ARTIFACTS_DIR / best_experiment.id / "metrics.json") or {}
    artifacts = metrics.get("artifacts") or {}

    return ActiveModelInfo(
        status="ready",
        source="best-experiment",
        label="Best trained paper-reproduction checkpoint",
        experimentId=best_experiment.id,
        checkpointFile=checkpoint_path.name,
        checkpointPath=str(checkpoint_path),
        checkpointMetricName=(
            str(artifacts["selected_checkpoint_metric_name"])
            if artifacts.get("selected_checkpoint_metric_name") is not None
            else None
        ),
        checkpointMetricValue=(
            float(artifacts["selected_checkpoint_metric_value"])
            if artifacts.get("selected_checkpoint_metric_value") is not None
            else None
        ),
        meanBestDeltaE=best_experiment.meanBestDeltaE,
        updatedAt=best_experiment.updatedAt,
        summary=(
            f"Runtime inverse design uses `{checkpoint_path.name}` from `{best_experiment.id}` as the active production checkpoint."
        ),
    )


def _build_agent_configuration_summary() -> AgentConfigurationSummary:
    settings = get_agent_settings()
    configured = bool(settings.api_key)

    if not settings.enabled:
        mode = "disabled"
        summary = "Decision support agent is disabled by configuration."
    elif configured:
        mode = "live"
        summary = (
            f"Decision support agent is enabled and will call {settings.provider_label} using `{settings.model}`."
        )
    else:
        mode = "fallback"
        summary = (
            "Decision support agent is enabled, but no API key is configured, so the backend will use heuristic decision support."
        )

    return AgentConfigurationSummary(
        enabled=settings.enabled,
        configured=configured,
        mode=mode,
        providerLabel=settings.provider_label,
        model=settings.model,
        apiBaseUrl=settings.api_base_url,
        summary=summary,
    )


def get_active_model_info() -> ActiveModelInfo | None:
    experiments = [experiment for experiment, _ in _list_experiments()]
    return _build_active_model_info(_pick_best_experiment(experiments))


def get_agent_configuration_summary() -> AgentConfigurationSummary:
    return _build_agent_configuration_summary()


def _operation_steps() -> list[AlgorithmOperationStep]:
    repo_root = str(ROOT)
    return [
        AlgorithmOperationStep(
            id="step-1",
            title="Activate the project environment",
            description="Use the dedicated Conda environment so FastAPI, PyTorch, NumPy and sklearn are all available.",
            command=f"cd {repo_root}\nconda activate opt_sim",
            expectedResult="`python` can import FastAPI and torch without errors.",
        ),
        AlgorithmOperationStep(
            id="step-2",
            title="Start the backend API",
            description="The backend exposes inverse-design APIs, workspace data, and this algorithm overview.",
            command=f"cd {repo_root}\nconda activate opt_sim\nuvicorn backend.app.main:app --reload --port 8002",
            expectedResult="Open http://127.0.0.1:8002/api/health and receive `{\\\"status\\\":\\\"ok\\\"}`.",
        ),
        AlgorithmOperationStep(
            id="step-3",
            title="Start the frontend workspace",
            description="The Vue app shows algorithm status, candidate ranking, and the operator guide.",
            command=f"cd {repo_root}/frontend\nnpm run dev",
            expectedResult="Open http://127.0.0.1:9002 and see the home page with the algorithm overview panel.",
        ),
        AlgorithmOperationStep(
            id="step-4",
            title="Submit a demo business request",
            description="Send one requirement sentence and one target color to create a reviewable run.",
            command=(
                "curl -X POST http://127.0.0.1:8002/api/agent/design-run \\\n"
                "  -H 'Content-Type: application/json' \\\n"
                "  -d '{\"requirementText\":\"Reproduce a warm copper structural color with the Ag-SiO2-Ag paper route.\","
                "\"targetHex\":\"#bf6f4f\",\"topK\":3}'"
            ),
            expectedResult="The response includes `activeRun`, `candidates`, `constraints`, and `exportEstimate`.",
        ),
        AlgorithmOperationStep(
            id="step-5",
            title="Run a smoke re-training job",
            description="Refresh the lightweight reproduction outputs quickly and update artifacts under `backend/artifacts/`.",
            command=(
                f"cd {repo_root}\nconda activate opt_sim\n"
                "python3 -u backend/scripts/train_cgan_reproduction.py "
                "--dataset-source paper --output-dir backend/artifacts/cgan_reproduction_smoke "
                "--epochs 5 --regressor-epochs 5 --batch-size 512 "
                "--generator-learning-rate 1e-3 --discriminator-learning-rate 2e-4 "
                "--steps-per-batch 1 --retrieval-metric euclidean_lab "
                "--paper-samples-per-lab 16 --device auto"
            ),
            expectedResult=(
                "`metrics.json`, `loss_history.csv`, `candidate_samples.csv`, "
                "`retrieval_metric_comparison.json`, and generator checkpoints are refreshed."
            ),
        ),
        AlgorithmOperationStep(
            id="step-6",
            title="Run a paper-grade training attempt",
            description="Move from smoke validation toward the paper target metrics with many more epochs and samples.",
            command=(
                f"cd {repo_root}\nconda activate opt_sim\n"
                "python3 -u backend/scripts/train_cgan_reproduction.py "
                "--dataset-source paper --paper-samples-per-lab 1000 "
                "--epochs 100000 --regressor-epochs 10000 "
                "--generator-learning-rate 1e-3 --discriminator-learning-rate 2e-4 "
                "--steps-per-batch 1 --retrieval-metric delta_e_2000 --device cuda"
            ),
            expectedResult="Compare the new `paper_reproduction` metrics against `paper_targets` after the run completes.",
        ),
    ]


def get_algorithm_overview() -> AlgorithmOverview:
    raw_experiments = _list_experiments()
    experiments = [experiment for experiment, _ in raw_experiments]
    latest_experiment = max(raw_experiments, key=lambda item: item[1])[0] if raw_experiments else None
    best_experiment = _pick_best_experiment(experiments)
    active_model = _build_active_model_info(best_experiment)
    agent_configuration = _build_agent_configuration_summary()
    _, environment_summary = _runtime_probe()

    if best_experiment is None:
        current_assessment = "No cGAN experiment artifacts were found yet."
        training_conclusion = "No checkpoints are available, so training cannot be considered complete."
        gpu_training_summary = "GPU usage is unknown because no experiment metrics are available."
    else:
        if best_experiment.meanBestDeltaE is not None and best_experiment.meanBestDeltaE <= 1.0:
            current_assessment = (
                "The pipeline is close to the paper target and can be used as a strong inverse-design baseline."
            )
        else:
            current_assessment = (
                "The full pipeline runs end-to-end, but the current cGAN quality is still far from the paper-level target. "
                "It is best understood as a validated prototype rather than a finished production model."
            )

        paper_reference = (
            f"{best_experiment.paperTargetMeanBestDeltaE:.2f}"
            if best_experiment.paperTargetMeanBestDeltaE is not None
            else "n/a"
        )
        best_delta_e = f"{best_experiment.meanBestDeltaE:.2f}" if best_experiment.meanBestDeltaE is not None else "n/a"
        training_conclusion = (
            f"Training has already run and produced checkpoints in {len(experiments)} experiment folders. "
            f"The latest artifact is `{latest_experiment.id if latest_experiment else 'n/a'}`. "
            f"However, the best current paper-style mean DeltaE is {best_delta_e}, while the paper reference is "
            f"{paper_reference}. Treat the model as trained, but not fully finished."
        )

        if latest_experiment is not None and latest_experiment.device == "cuda":
            device_name = latest_experiment.deviceName or "a detected NVIDIA GPU"
            gpu_training_summary = f"Yes. The saved experiment artifacts show CUDA training on {device_name}."
        else:
            gpu_training_summary = "The code supports GPU auto-selection, but the latest saved artifact does not confirm a CUDA run."

    return AlgorithmOverview(
        algorithmName="Ag-SiO2-Ag cGAN plus thin-film inverse design",
        businessGoal=(
            "Given a target structural color, generate several manufacturable Ag-SiO2-Ag thickness triples, "
            "verify them with physics simulation, and surface the most reviewable options to the user."
        ),
        workflowSummary=(
            "The cGAN proposes multiple layer-thickness candidates from a target color. "
            "Thin-film transfer-matrix simulation recalculates color error and process drift, "
            "and a local refinement pass improves both retrieval and cGAN seeds before the UI ranks candidates."
        ),
        plainExplanation=(
            "In plain language: the model first guesses a few film-thickness recipes for the requested color, "
            "then physics simulation checks which guesses are truly close to that color and which ones are more stable "
            "under small fabrication errors, and finally a local search nudges each promising recipe to a better nearby setting."
        ),
        currentAssessment=current_assessment,
        trainingConclusion=training_conclusion,
        gpuTrainingSummary=gpu_training_summary,
        currentEnvironmentSummary=environment_summary,
        artifactRootPath=str(ARTIFACTS_DIR),
        guideDocumentPath=str(GUIDE_DOCUMENT_PATH),
        bestExperimentId=best_experiment.id if best_experiment else None,
        latestExperimentId=latest_experiment.id if latest_experiment else None,
        bestExperiment=best_experiment,
        latestExperiment=latest_experiment,
        activeModel=active_model,
        agentConfiguration=agent_configuration,
        experiments=experiments,
        headlineMetrics=_headline_metrics(
            best_experiment=best_experiment,
            latest_experiment=latest_experiment,
        ),
        targetComparisons=_target_comparisons(best_experiment),
        operationSteps=_operation_steps(),
    )
