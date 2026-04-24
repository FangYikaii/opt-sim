from __future__ import annotations

from pydantic import BaseModel, Field

from .review import CandidateSolution, ConstraintCheck, ExportEstimate
from .runs import RunSummary, TargetAsset, WorkspaceDraft
from .timeline import TimelineEvent


class DesignRequest(BaseModel):
    requirementText: str = Field(min_length=4)
    targetHex: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    topK: int = Field(default=3, ge=1, le=8)


class DesignRunResponse(BaseModel):
    activeRun: RunSummary
    draft: WorkspaceDraft
    targets: list[TargetAsset]
    timeline: list[TimelineEvent]
    candidates: list[CandidateSolution]
    constraints: list[ConstraintCheck]
    exportEstimate: ExportEstimate
