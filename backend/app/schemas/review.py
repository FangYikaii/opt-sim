from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


ConstraintState = Literal["pass", "warning", "fail", "unknown"]
CandidateStatus = Literal["Recommended", "Robust", "Watch", "Blocked"]


class CandidateParameter(BaseModel):
    label: str
    value: str


class CandidateMetric(BaseModel):
    label: str
    value: str


class CandidateSolution(BaseModel):
    id: str
    rank: int
    group: str
    selected: bool = False
    status: CandidateStatus
    parameters: list[CandidateParameter]
    metrics: list[CandidateMetric]
    targetColorHex: str
    simulatedColorHex: str
    processPlusColorHex: str
    processMinusColorHex: str
    rationale: str


class ConstraintCheck(BaseModel):
    id: str
    label: str
    detail: str
    state: ConstraintState


class ExportEstimate(BaseModel):
    dimensions: str
    fileSize: str
    tilePlan: str
    format: str
    progress: int
    tileProgress: str
