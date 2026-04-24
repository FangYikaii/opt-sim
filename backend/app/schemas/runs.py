from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


RunStatus = Literal[
    "Draft",
    "Validating",
    "Simulating",
    "Ranking",
    "Needs approval",
    "Exporting",
    "Complete",
    "Failed",
]


class RunSummary(BaseModel):
    id: str
    title: str
    status: RunStatus
    updatedAt: str
    warning: bool = False


class WorkspaceDraft(BaseModel):
    requirementText: str
    targetLabel: str
    targetValue: str
    heightWindow: str
    exportMode: str


class TargetAsset(BaseModel):
    id: str
    name: str
    type: Literal["color", "image", "multi-view"]
    detail: str
    swatchHex: str | None = None
