from __future__ import annotations

import os
from typing import Any

import httpx

from .algorithms import run_inverse_design
from .models import (
    CandidateSolution,
    ConstraintCheck,
    DesignRequest,
    DesignRunResponse,
    ExportEstimate,
    RunSummary,
    TargetAsset,
    TimelineEvent,
    WorkspaceDraft,
)
from .runtime_store import store_design_run

CODEX_BASE_URL = "https://code.ppchat.vip/v1"


def _fallback_timeline(requirement_text: str, target_hex: str) -> list[TimelineEvent]:
    return [
        TimelineEvent(
            id="evt-1",
            type="agent",
            label="System reasoning",
            title="Paper workflow selected",
            body=f"Parsed a single-target structural color request for {target_hex.upper()} and routed it through the Ag-SiO2-Ag paper-reproduction workflow.",
            meta="paper reproduction",
            status="Validating",
        ),
        TimelineEvent(
            id="evt-2",
            type="tool",
            label="Calculation step",
            title="run_inverse_design()",
            body="Executed the hybrid inverse-design stack: retrieval seeds, cGAN proposals, thin-film simulation, and local refinement before manufacturability-aware ranking.",
            meta="retrieval + cgan + refinement",
            status="Simulating",
        ),
        TimelineEvent(
            id="evt-3",
            type="approval",
            label="Business approval",
            title="Please confirm before adaptation/export",
            body=f"The reproduction-first candidate set is ready for review before broader problem adaptation. Requirement summary: {requirement_text}",
            meta="awaiting review",
            status="Needs approval",
            actionLabel="Review deliverables",
        ),
    ]


async def _call_codex_summary(requirement_text: str, target_hex: str) -> str | None:
    api_key = os.getenv("CODEX_API_KEY")
    if not api_key:
        return None

    payload: dict[str, Any] = {
        "model": os.getenv("CODEX_MODEL", "gpt-5.3-codex-spark"),
        "messages": [
            {
                "role": "system",
                "content": "You are an optics design agent. Summarize user intent for a single-target Ag-SiO2-Ag structural color design run in one concise engineering sentence.",
            },
            {
                "role": "user",
                "content": f"Requirement: {requirement_text}\nTarget color: {target_hex}",
            },
        ],
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{CODEX_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


async def run_design_agent(request: DesignRequest) -> DesignRunResponse:
    inverse_result = run_inverse_design(request.targetHex, request.topK)
    llm_summary = await _call_codex_summary(request.requirementText, request.targetHex)

    draft = WorkspaceDraft(
        requirementText=llm_summary or request.requirementText,
        targetLabel="Target color",
        targetValue=request.targetHex.upper(),
        heightWindow="Ag 10-30 nm / SiO2 60-180 nm",
        exportMode="Preview-first TIFF planning",
    )
    active_run = RunSummary(
        id="run-demo-live",
        title="Ag-SiO2-Ag reproduction brief",
        status="Needs approval",
        updatedAt="just now",
        warning=False,
    )
    targets = [
        TargetAsset(
            id="target-demo",
            name="Reference target",
            type="color",
            detail=request.targetHex.upper(),
            swatchHex=request.targetHex,
        )
    ]
    timeline = _fallback_timeline(request.requirementText, request.targetHex)

    return DesignRunResponse(
        activeRun=active_run,
        draft=draft,
        targets=targets,
        timeline=timeline,
        candidates=inverse_result.candidates,
        constraints=inverse_result.constraints,
        exportEstimate=inverse_result.export_estimate,
    )
