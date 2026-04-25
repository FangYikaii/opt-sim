from __future__ import annotations

from pydantic import BaseModel

from .artifacts import ArtifactSummary
from .decision import ActiveModelInfo, AgentConfigurationSummary, DecisionSupport
from .project import WorkspaceProject
from .review import CandidateSolution, ConstraintCheck, ExportEstimate
from .runs import RunSummary, TargetAsset, WorkspaceDraft
from .timeline import TimelineEvent


class WorkspaceDetail(BaseModel):
    project: WorkspaceProject
    activeRun: RunSummary
    draft: WorkspaceDraft
    runs: list[RunSummary]
    targets: list[TargetAsset]
    artifacts: list[ArtifactSummary]
    timeline: list[TimelineEvent]
    candidates: list[CandidateSolution]
    constraints: list[ConstraintCheck]
    exportEstimate: ExportEstimate
    activeModel: ActiveModelInfo
    agentConfiguration: AgentConfigurationSummary
    decisionSupport: DecisionSupport
