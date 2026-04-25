from __future__ import annotations

import asyncio
from datetime import datetime
import json
from typing import Any

import httpx

from .algorithm_overview import get_active_model_info, get_agent_configuration_summary
from .algorithms import run_inverse_design
from .config import get_agent_settings
from .models import (
    ActiveModelInfo,
    AgentConfigurationSummary,
    CandidateSolution,
    ConstraintCheck,
    DecisionSupport,
    DesignRequest,
    DesignRunResponse,
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
                    "You are an optics design agent. Summarize the user intent for a single-target "
                    "Ag-SiO2-Ag structural color run in one concise engineering sentence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Requirement: {request.requirementText}\n"
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


async def run_design_agent(request: DesignRequest) -> DesignRunResponse:
    requirement_summary_task = asyncio.create_task(_call_agent_summary(request))
    inverse_result = await asyncio.to_thread(
        run_inverse_design,
        request.targetHex,
        request.topK,
        request.thetaDeg,
        request.polarization,
    )
    active_model = get_active_model_info() or _build_fallback_active_model()
    agent_configuration = get_agent_configuration_summary()
    decision_support_task = asyncio.create_task(
        _resolve_decision_support(
            request=request,
            candidates=inverse_result.candidates,
            constraints=inverse_result.constraints,
        )
    )
    requirement_summary, decision_support = await asyncio.gather(
        requirement_summary_task,
        decision_support_task,
    )

    polarization_label = _format_polarization_label(request.polarization)
    run_id = _build_run_id()

    draft = WorkspaceDraft(
        requirementText=requirement_summary or _fallback_requirement_summary(request),
        targetLabel="Target color",
        targetValue=request.targetHex.upper(),
        incidenceAngleLabel="Incidence angle",
        incidenceAngleValue=f"{request.thetaDeg:.1f} deg",
        polarizationLabel="Polarization",
        polarizationValue=polarization_label,
        heightWindow="Ag 10-30 nm / SiO2 60-180 nm",
        exportMode="Preview-first TIFF planning",
    )
    active_run = RunSummary(
        id=run_id,
        title=f"Ag-SiO2-Ag decision brief · {request.targetHex.upper()}",
        status="Needs approval",
        updatedAt="just now",
        warning=decision_support.confidence == "low",
    )
    targets = [
        TargetAsset(
            id=f"{run_id}-target",
            name="Reference target",
            type="color",
            detail=f"{request.targetHex.upper()} at {request.thetaDeg:.1f} deg · {polarization_label}",
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
        candidates=inverse_result.candidates,
        constraints=inverse_result.constraints,
        exportEstimate=inverse_result.export_estimate,
        activeModel=active_model,
        agentConfiguration=agent_configuration,
        decisionSupport=decision_support,
    )
