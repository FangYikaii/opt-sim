from __future__ import annotations

import asyncio
from datetime import datetime
import json
from typing import Any

import httpx

from .algorithm_overview import get_active_model_info, get_agent_configuration_summary
from .algorithms import run_cgh_design, run_inverse_design
from .config import get_agent_settings
from .models import (
    ActiveModelInfo,
    AgentConfigurationSummary,
    CandidateSolution,
    ConstraintCheck,
    DecisionSupport,
    DesignRequest,
    DesignRunResponse,
    ExportEstimate,
    RunSummary,
    TargetAsset,
    TimelineEvent,
    WorkspaceDraft,
)


def _format_polarization_label(polarization: str) -> str:
    if polarization == "te":
        return "TE"
    if polarization == "tm":
        return "TM"
    return "Unpolarized"


def _format_design_mode_label(design_mode: str) -> str:
    if design_mode == "neural-holography":
        return "Neural holography / CITL"
    return "Structural color / thin film"


def _build_fallback_active_model() -> ActiveModelInfo:
    return ActiveModelInfo(
        status="fallback",
        source="runtime-fallback",
        label="Runtime lightweight cGAN",
        summary="No saved production checkpoint was resolved, so inverse design is using runtime lightweight cGAN sampling.",
    )


def _selected_candidate(candidates: list[CandidateSolution]) -> CandidateSolution | None:
    if not candidates:
        return None
    return next((candidate for candidate in candidates if candidate.selected), candidates[0])


def _metric_value(candidate: CandidateSolution | None, label: str, default: str = "n/a") -> str:
    if candidate is None:
        return default
    metric = next((item for item in candidate.metrics if item.label == label), None)
    return metric.value if metric is not None else default


def _numeric_metric(candidate: CandidateSolution | None, label: str) -> float | None:
    raw = _metric_value(candidate, label, "")
    if not raw:
        return None
    try:
        return float(raw.split()[0])
    except ValueError:
        return None


def _build_run_id() -> str:
    return f"run-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


def _fallback_requirement_summary(request: DesignRequest) -> str:
    if request.designMode == "neural-holography":
        return (
            f"{request.requirementText} Target holographic image color anchor {request.targetHex.upper()} "
            f"for a phase-only SLM workflow at {request.thetaDeg:.1f} deg under "
            f"{_format_polarization_label(request.polarization)} light."
        )
    return (
        f"{request.requirementText} Target {request.targetHex.upper()} at "
        f"{request.thetaDeg:.1f} deg under {_format_polarization_label(request.polarization)} light."
    )


def _agent_timeout_seconds(max_seconds: float) -> float:
    settings = get_agent_settings()
    return max(1.0, min(settings.timeout_seconds, max_seconds))


async def _call_agent_summary(request: DesignRequest) -> str | None:
    settings = get_agent_settings()
    if not settings.enabled or not settings.api_key:
        return None

    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an optics design agent. Summarize the user intent in one concise "
                    "engineering sentence. Distinguish thin-film structural-color runs from "
                    "neural holography camera-in-the-loop runs."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Requirement: {request.requirementText}\n"
                    f"Design mode: {_format_design_mode_label(request.designMode)}\n"
                    f"Target color: {request.targetHex.upper()}\n"
                    f"Incidence angle: {request.thetaDeg:.1f} degrees\n"
                    f"Polarization: {_format_polarization_label(request.polarization)}"
                ),
            },
        ],
        "temperature": settings.temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=_agent_timeout_seconds(6.0)) as client:
            response = await client.post(
                f"{settings.api_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _heuristic_decision_support(
    *,
    request: DesignRequest,
    candidates: list[CandidateSolution],
    constraints: list[ConstraintCheck],
) -> DecisionSupport:
    selected = _selected_candidate(candidates)
    if selected is None:
        return DecisionSupport(
            mode="heuristic",
            confidence="low",
            headline="No fabrication candidate is available yet.",
            summary="The run completed without ranked candidates, so human review is required before continuing.",
            nextAction="Re-run the design request after confirming the target color and process window.",
            rationale=["No candidate was produced by the current inverse-design pass."],
            risks=["The request cannot move into fabrication planning until at least one candidate is ranked."],
        )

    if request.designMode == "neural-holography":
        return _heuristic_holography_decision_support(
            request=request,
            candidates=candidates,
            constraints=constraints,
        )

    delta_e = _numeric_metric(selected, "DeltaE")
    source = _metric_value(selected, "Source")
    manufacturability = _metric_value(selected, "Manufacturability")
    process_drift = _metric_value(selected, "Process drift")
    confidence = "medium"
    if delta_e is not None and delta_e <= 2.0 and manufacturability == "High":
        confidence = "high"
    elif delta_e is not None and delta_e > 5.0:
        confidence = "low"

    warning_constraints = [item.detail for item in constraints if item.state in {"warning", "fail"}]
    risks = [
        f"Process sensitivity remains a watch item: {process_drift}.",
        f"The ranking is only validated at {request.thetaDeg:.1f} deg under {_format_polarization_label(request.polarization)} light.",
    ]
    if warning_constraints:
        risks.append(warning_constraints[0])

    rationale = [
        f"{selected.id} is the current top-ranked candidate with status `{selected.status}`.",
        f"It combines the lowest available color error with a composite ranking produced by {source}.",
        f"Manufacturability is currently assessed as {manufacturability}.",
    ]

    return DecisionSupport(
        mode="heuristic",
        confidence=confidence,
        headline=f"Recommend {selected.id} as the first fabrication trial.",
        summary=(
            f"{selected.id} is the best current option for {request.targetHex.upper()} because it leads the ranked set "
            f"after color-error, process-drift, and local-refinement scoring."
        ),
        recommendedCandidateId=selected.id,
        nextAction="Review the recommended candidate, then confirm whether it should proceed to export and fabrication planning.",
        rationale=rationale,
        risks=risks[:3],
    )


def _heuristic_holography_decision_support(
    *,
    request: DesignRequest,
    candidates: list[CandidateSolution],
    constraints: list[ConstraintCheck],
) -> DecisionSupport:
    selected = _selected_candidate(candidates)
    if selected is None:
        return DecisionSupport(
            mode="heuristic",
            confidence="low",
            headline="No holography candidate is available yet.",
            summary="The neural holography planning pass produced no route, so the optical setup and target image must be confirmed.",
            nextAction="Confirm SLM resolution, wavelength channels, and whether camera-in-the-loop calibration data is available.",
            rationale=["No phase-generation route was produced by the current planning pass."],
            risks=["A holographic run cannot proceed without a target image and calibrated propagation assumptions."],
        )

    warning_constraints = [item.detail for item in constraints if item.state in {"warning", "fail"}]
    risks = [
        "Bench quality depends on SLM phase response, laser nonuniformity, and aberration calibration.",
        "The current implementation is a planning slice; it does not train a production HoloNet checkpoint yet.",
    ]
    if warning_constraints:
        risks.append(warning_constraints[0])

    return DecisionSupport(
        mode="heuristic",
        confidence="medium",
        headline=f"Recommend {selected.id} as the neural holography integration route.",
        summary=(
            "The selected route preserves the existing physics workspace while adding camera-in-the-loop "
            "calibration, phase-only hologram optimization, and HoloNet-style real-time inference gates."
        ),
        recommendedCandidateId=selected.id,
        nextAction="Review the CITL calibration checklist, then decide whether to collect bench captures or continue with simulation-only validation.",
        rationale=[
            f"{selected.id} maps the new paper requirements onto an executable CGH workflow.",
            "It separates iterative SGD/CITL quality optimization from HoloNet-style real-time deployment.",
            "It keeps physical verification explicit before any final phase-map export.",
        ],
        risks=risks[:3],
    )


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


async def _call_agent_decision_support(
    *,
    request: DesignRequest,
    candidates: list[CandidateSolution],
    constraints: list[ConstraintCheck],
) -> DecisionSupport | None:
    settings = get_agent_settings()
    if not settings.enabled or not settings.api_key:
        return None

    candidate_lines = []
    for candidate in candidates[:3]:
        candidate_lines.append(
            {
                "id": candidate.id,
                "status": candidate.status,
                "rationale": candidate.rationale,
                "parameters": {item.label: item.value for item in candidate.parameters},
                "metrics": {item.label: item.value for item in candidate.metrics},
            }
        )

    constraint_lines = [
        {"label": item.label, "state": item.state, "detail": item.detail}
        for item in constraints[:5]
    ]

    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a structural-color decision support agent. Return JSON only with keys: "
                    "headline, summary, nextAction, confidence, recommendedCandidateId, rationale, risks. "
                    "Use confidence values high, medium, or low. rationale and risks must each be arrays with 1 to 3 short strings."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "requirementText": request.requirementText,
                        "designMode": request.designMode,
                        "targetHex": request.targetHex.upper(),
                        "thetaDeg": request.thetaDeg,
                        "polarization": _format_polarization_label(request.polarization),
                        "candidates": candidate_lines,
                        "constraints": constraint_lines,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": settings.temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=_agent_timeout_seconds(8.0)) as client:
            response = await client.post(
                f"{settings.api_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    parsed = _extract_json_object(raw_text)
    if parsed is None:
        return None

    try:
        return DecisionSupport(
            mode="llm",
            confidence=str(parsed.get("confidence", "medium")),
            headline=str(parsed.get("headline", "")),
            summary=str(parsed.get("summary", "")),
            recommendedCandidateId=(
                str(parsed["recommendedCandidateId"])
                if parsed.get("recommendedCandidateId") is not None
                else None
            ),
            nextAction=str(parsed.get("nextAction", "")),
            rationale=[str(item) for item in parsed.get("rationale", [])][:3],
            risks=[str(item) for item in parsed.get("risks", [])][:3],
        )
    except Exception:
        return None


async def _resolve_decision_support(
    *,
    request: DesignRequest,
    candidates: list[CandidateSolution],
    constraints: list[ConstraintCheck],
) -> DecisionSupport:
    llm_decision = await _call_agent_decision_support(
        request=request,
        candidates=candidates,
        constraints=constraints,
    )
    if llm_decision is not None:
        return llm_decision
    return _heuristic_decision_support(
        request=request,
        candidates=candidates,
        constraints=constraints,
    )


def _build_timeline(
    *,
    request: DesignRequest,
    active_model: ActiveModelInfo,
    decision_support: DecisionSupport,
) -> list[TimelineEvent]:
    if request.designMode == "neural-holography":
        return _build_holography_timeline(
            request=request,
            active_model=active_model,
            decision_support=decision_support,
        )

    polarization_label = _format_polarization_label(request.polarization)
    model_label = active_model.checkpointFile or active_model.label
    return [
        TimelineEvent(
            id="evt-1",
            type="agent",
            label="System reasoning",
            title="Paper workflow selected",
            body=(
                f"Parsed a single-target structural color request for {request.targetHex.upper()} "
                f"at {request.thetaDeg:.1f} degree incidence with {polarization_label} illumination and routed it through the "
                "Ag-SiO2-Ag paper-reproduction workflow."
            ),
            meta="paper reproduction + angle-aware TMM",
            status="Validating",
        ),
        TimelineEvent(
            id="evt-2",
            type="tool",
            label="Calculation step",
            title="Best checkpoint activated",
            body=(
                f"Executed inverse design with `{model_label}` as the active generator, followed by angle-dependent thin-film "
                f"simulation at {request.thetaDeg:.1f} degrees under {polarization_label} light and local refinement."
            ),
            meta=active_model.experimentId or active_model.source,
            status="Simulating",
        ),
        TimelineEvent(
            id="evt-3",
            type="result",
            label="Decision support",
            title=decision_support.headline,
            body=decision_support.summary,
            meta=f"{decision_support.mode} · {decision_support.confidence} confidence",
            status="Needs approval",
        ),
        TimelineEvent(
            id="evt-4",
            type="approval",
            label="Business approval",
            title="Please confirm before adaptation/export",
            body=(
                f"The candidate set is ready for review. Next action: {decision_support.nextAction}"
            ),
            meta="awaiting review",
            status="Needs approval",
            actionLabel="Review deliverables",
        ),
    ]


def _build_holography_candidates(request: DesignRequest) -> list[CandidateSolution]:
    target_hex = request.targetHex.upper()
    return [
        CandidateSolution(
            id="holo-citl-sgd",
            rank=1,
            group="Camera-in-the-loop phase optimization",
            selected=True,
            status="Recommended",
            parameters=[
                {"label": "SLM mode", "value": "phase-only 1080p"},
                {"label": "Propagation", "value": "ASM/Fresnel proxy"},
                {"label": "Optimizer", "value": "SGD + physical capture loss"},
                {"label": "Calibration", "value": "per-pixel phase + source + aberration"},
            ],
            metrics=[
                {"label": "Source", "value": "Neural Holography CITL"},
                {"label": "Quality gate", "value": "PSNR/SSIM capture review"},
                {"label": "Runtime", "value": "iterative offline"},
                {"label": "Bench dependency", "value": "Camera required"},
            ],
            targetColorHex=target_hex,
            simulatedColorHex="#d8dce6",
            processPlusColorHex="#cfd8ee",
            processMinusColorHex="#e1d7cb",
            rationale=(
                "Use camera-in-the-loop optimization as the highest-fidelity route for a single target image, "
                "then use the measured loss to update calibration assumptions."
            ),
        ),
        CandidateSolution(
            id="holo-proxy-holonet",
            rank=2,
            group="Calibrated proxy plus neural inference",
            selected=False,
            status="Robust",
            parameters=[
                {"label": "Network", "value": "HoloNet-style encoder-decoder"},
                {"label": "Training", "value": "CITL-calibrated differentiable proxy"},
                {"label": "Output", "value": "phase map"},
                {"label": "Target", "value": "real-time 1080p"},
            ],
            metrics=[
                {"label": "Source", "value": "HoloNet deployment"},
                {"label": "Quality gate", "value": "proxy + bench holdout"},
                {"label": "Runtime", "value": "real-time target"},
                {"label": "Bench dependency", "value": "Calibration dataset"},
            ],
            targetColorHex=target_hex,
            simulatedColorHex="#cfd5df",
            processPlusColorHex="#c9d4e8",
            processMinusColorHex="#ded5cf",
            rationale=(
                "Train a real-time generator only after the interpretable propagation proxy is calibrated against camera captures."
            ),
        ),
        CandidateSolution(
            id="holo-sim-baseline",
            rank=3,
            group="Simulation-only CGH baseline",
            selected=False,
            status="Watch",
            parameters=[
                {"label": "Baseline", "value": "GS / WH / SGD comparison"},
                {"label": "Model", "value": "ideal wave propagation"},
                {"label": "Output", "value": "phase map + replay preview"},
                {"label": "Use", "value": "smoke test"},
            ],
            metrics=[
                {"label": "Source", "value": "CGH simulation"},
                {"label": "Quality gate", "value": "ideal-model PSNR"},
                {"label": "Runtime", "value": "fast smoke"},
                {"label": "Bench dependency", "value": "None"},
            ],
            targetColorHex=target_hex,
            simulatedColorHex="#d4d8dd",
            processPlusColorHex="#ccd5e2",
            processMinusColorHex="#ddd6cf",
            rationale=(
                "Keep a simulation-only path for regression tests and algorithm comparison, while marking model mismatch as a known risk."
            ),
        ),
    ][: request.topK]


def _build_holography_constraints(request: DesignRequest) -> list[ConstraintCheck]:
    polarization_label = _format_polarization_label(request.polarization)
    return [
        ConstraintCheck(
            id="holo-target-image",
            label="Target image requirement",
            detail=(
                f"Current request uses {request.targetHex.upper()} as a color anchor; production holography needs target RGB images "
                "or multi-plane focal targets before final phase export."
            ),
            state="warning",
        ),
        ConstraintCheck(
            id="holo-citl-calibration",
            label="CITL calibration",
            detail="Camera captures must calibrate source intensity, per-pixel phase nonlinearity, and optical aberrations.",
            state="warning",
        ),
        ConstraintCheck(
            id="holo-slm-output",
            label="SLM phase output",
            detail="The export target changes from grayscale relief height to wrapped phase maps and calibration metadata.",
            state="pass",
        ),
        ConstraintCheck(
            id="holo-view-condition",
            label="Viewing condition",
            detail=f"Planning condition is {request.thetaDeg:.1f} deg with {polarization_label}; bench replay must confirm captured quality.",
            state="pass",
        ),
    ]


def _build_holography_export_estimate() -> ExportEstimate:
    return ExportEstimate(
        dimensions="1920 x 1080 phase map set",
        fileSize="16-64 MB per phase batch",
        tilePlan="frame sequence + calibration manifest",
        format="PNG/TIFF phase map + JSON metadata",
        progress=0,
        tileProgress="0 / pending frames",
    )


def _build_holography_timeline(
    *,
    request: DesignRequest,
    active_model: ActiveModelInfo,
    decision_support: DecisionSupport,
) -> list[TimelineEvent]:
    polarization_label = _format_polarization_label(request.polarization)
    model_label = active_model.checkpointFile or active_model.label
    return [
        TimelineEvent(
            id="evt-1",
            type="agent",
            label="System reasoning",
            title="Neural holography workflow selected",
            body=(
                "Routed the request through the new Neural Holography requirement branch from "
                "3414685.3417802.pdf, covering CGH baselines, camera-in-the-loop calibration, and HoloNet-style inference."
            ),
            meta="CGH + CITL + HoloNet",
            status="Validating",
        ),
        TimelineEvent(
            id="evt-2",
            type="tool",
            label="Planning step",
            title="CITL calibration gates prepared",
            body=(
                f"Prepared phase-only SLM planning at {request.thetaDeg:.1f} degrees under {polarization_label} light, "
                f"using `{model_label}` only as existing platform context rather than a holography checkpoint."
            ),
            meta=active_model.experimentId or active_model.source,
            status="Simulating",
        ),
        TimelineEvent(
            id="evt-3",
            type="result",
            label="Decision support",
            title=decision_support.headline,
            body=decision_support.summary,
            meta=f"{decision_support.mode} · {decision_support.confidence} confidence",
            status="Needs approval",
        ),
        TimelineEvent(
            id="evt-4",
            type="approval",
            label="Bench approval",
            title="Confirm before phase-map export",
            body=f"The holography route is ready for review. Next action: {decision_support.nextAction}",
            meta="awaiting CITL decision",
            status="Needs approval",
            actionLabel="Review holography gates",
        ),
    ]


async def run_design_agent(request: DesignRequest) -> DesignRunResponse:
    requirement_summary_task = asyncio.create_task(_call_agent_summary(request))
    if request.designMode == "neural-holography":
        try:
            cgh_result = await asyncio.to_thread(
                run_cgh_design,
                request.targetHex,
                request.topK,
                request.thetaDeg,
                request.polarization,
            )
            candidates = cgh_result.candidates
            constraints = cgh_result.constraints
            export_estimate = cgh_result.export_estimate
        except Exception:
            candidates = _build_holography_candidates(request)
            constraints = _build_holography_constraints(request)
            export_estimate = _build_holography_export_estimate()
    else:
        inverse_result = await asyncio.to_thread(
            run_inverse_design,
            request.targetHex,
            request.topK,
            request.thetaDeg,
            request.polarization,
        )
        candidates = inverse_result.candidates
        constraints = inverse_result.constraints
        export_estimate = inverse_result.export_estimate

    active_model = get_active_model_info() or _build_fallback_active_model()
    agent_configuration = get_agent_configuration_summary()
    decision_support_task = asyncio.create_task(
        _resolve_decision_support(
            request=request,
            candidates=candidates,
            constraints=constraints,
        )
    )
    requirement_summary, decision_support = await asyncio.gather(
        requirement_summary_task,
        decision_support_task,
    )

    polarization_label = _format_polarization_label(request.polarization)
    run_id = _build_run_id()

    if request.designMode == "neural-holography":
        draft = WorkspaceDraft(
            requirementText=requirement_summary or _fallback_requirement_summary(request),
            designMode=request.designMode,
            referenceSource="3414685.3417802.pdf · Neural Holography with Camera-in-the-loop Training",
            outputKind="Phase-only SLM hologram route",
            targetLabel="Target image anchor",
            targetValue=request.targetHex.upper(),
            incidenceAngleLabel="Replay angle",
            incidenceAngleValue=f"{request.thetaDeg:.1f} deg",
            polarizationLabel="Illumination",
            polarizationValue=polarization_label,
            heightWindow="Wrapped phase 0-2pi / calibrated SLM response",
            exportMode="Phase map + calibration manifest",
            calibrationMode="Camera-in-the-loop calibration",
            runtimeTarget="HoloNet-style real-time 1080p inference after proxy training",
        )
        title = f"Neural holography CITL brief · {request.targetHex.upper()}"
        target_detail = (
            f"{request.targetHex.upper()} image anchor · phase-only SLM · "
            f"{request.thetaDeg:.1f} deg · {polarization_label}"
        )
    else:
        draft = WorkspaceDraft(
            requirementText=requirement_summary or _fallback_requirement_summary(request),
            designMode=request.designMode,
            referenceSource="10.1515_nanoph-2022-0095.pdf · cGAN structural-color inverse design",
            outputKind="Ag-SiO2-Ag candidate set",
            targetLabel="Target color",
            targetValue=request.targetHex.upper(),
            incidenceAngleLabel="Incidence angle",
            incidenceAngleValue=f"{request.thetaDeg:.1f} deg",
            polarizationLabel="Polarization",
            polarizationValue=polarization_label,
            heightWindow="Ag 10-30 nm / SiO2 60-180 nm",
            exportMode="Preview-first TIFF planning",
            calibrationMode="Angle-aware TMM forward simulation",
            runtimeTarget="Business preview and fabrication planning",
        )
        title = f"Ag-SiO2-Ag decision brief · {request.targetHex.upper()}"
        target_detail = f"{request.targetHex.upper()} at {request.thetaDeg:.1f} deg · {polarization_label}"

    active_run = RunSummary(
        id=run_id,
        title=title,
        status="Needs approval",
        updatedAt="just now",
        warning=decision_support.confidence == "low",
    )
    targets = [
        TargetAsset(
            id=f"{run_id}-target",
            name="Reference target",
            type="color",
            detail=target_detail,
            swatchHex=request.targetHex,
        )
    ]
    timeline = _build_timeline(
        request=request,
        active_model=active_model,
        decision_support=decision_support,
    )

    return DesignRunResponse(
        activeRun=active_run,
        draft=draft,
        targets=targets,
        timeline=timeline,
        candidates=candidates,
        constraints=constraints,
        exportEstimate=export_estimate,
        activeModel=active_model,
        agentConfiguration=agent_configuration,
        decisionSupport=decision_support,
    )
