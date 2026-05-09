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
    designMode: str = "structural-color"
    referenceSource: str = "10.1515_nanoph-2022-0095.pdf"
    outputKind: str = "Thin-film candidate set"
    targetLabel: str
    targetValue: str
    incidenceAngleLabel: str
    incidenceAngleValue: str
    polarizationLabel: str
    polarizationValue: str
    heightWindow: str
    exportMode: str
    calibrationMode: str = "Forward simulation"
    runtimeTarget: str = "Preview-first offline design"


class TargetAsset(BaseModel):
    id: str
    name: str
    type: Literal["color", "image", "multi-view"]
    detail: str
    swatchHex: str | None = None
