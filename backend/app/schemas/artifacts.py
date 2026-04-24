from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


ArtifactStatus = Literal["ready", "running", "pending"]
ArtifactType = Literal["report", "preview", "export"]


class ArtifactSummary(BaseModel):
    id: str
    name: str
    type: ArtifactType
    status: ArtifactStatus


class ArtifactMetadataItem(BaseModel):
    label: str
    value: str


class ArtifactDetail(BaseModel):
    id: str
    runId: str
    name: str
    type: ArtifactType
    status: ArtifactStatus
    description: str
    metadata: list[ArtifactMetadataItem]
