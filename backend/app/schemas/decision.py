from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


DecisionConfidence = Literal["high", "medium", "low"]
DecisionMode = Literal["heuristic", "llm"]
ModelSelectionMode = Literal["best-experiment", "runtime-fallback"]
AgentMode = Literal["live", "fallback", "disabled"]


class ActiveModelInfo(BaseModel):
    status: Literal["ready", "fallback"]
    source: ModelSelectionMode
    label: str
    experimentId: str | None = None
    checkpointFile: str | None = None
    checkpointPath: str | None = None
    checkpointMetricName: str | None = None
    checkpointMetricValue: float | None = None
    meanBestDeltaE: float | None = None
    updatedAt: str | None = None
    summary: str


class AgentConfigurationSummary(BaseModel):
    enabled: bool
    configured: bool
    mode: AgentMode
    providerLabel: str
    model: str
    apiBaseUrl: str
    summary: str


class DecisionSupport(BaseModel):
    mode: DecisionMode
    confidence: DecisionConfidence
    headline: str
    summary: str
    recommendedCandidateId: str | None = None
    nextAction: str
    rationale: list[str]
    risks: list[str]
