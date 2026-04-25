import type { components } from './generated-types'

export type ApiErrorPayload = components['schemas']['ApiErrorPayload']
export type ActiveModelInfo = components['schemas']['ActiveModelInfo']
export type AlgorithmExperiment = components['schemas']['AlgorithmExperiment']
export type AlgorithmHeadlineMetric = components['schemas']['AlgorithmHeadlineMetric']
export type AlgorithmOperationStep = components['schemas']['AlgorithmOperationStep']
export type AlgorithmOverview = components['schemas']['AlgorithmOverview']
export type AlgorithmTargetComparison = components['schemas']['AlgorithmTargetComparison']
export type AgentConfigurationSummary = components['schemas']['AgentConfigurationSummary']
export type DesignRequest = components['schemas']['DesignRequest']
export type DesignRunResponse = components['schemas']['DesignRunResponse']
export type DecisionSupport = components['schemas']['DecisionSupport']
export type RunSummary = components['schemas']['RunSummary']
export type WorkspaceDraft = components['schemas']['WorkspaceDraft']
export type TargetAsset = components['schemas']['TargetAsset']
export type TimelineEvent = components['schemas']['TimelineEvent']
export type CandidateSolution = components['schemas']['CandidateSolution']
export type ConstraintCheck = components['schemas']['ConstraintCheck']
export type ExportEstimate = components['schemas']['ExportEstimate']
export type ArtifactSummary = components['schemas']['ArtifactSummary']
export type ArtifactDetail = components['schemas']['ArtifactDetail']
export type WorkspaceProject = components['schemas']['WorkspaceProject']

export interface WorkspaceDetail {
  project: WorkspaceProject
  activeRun: RunSummary
  draft: WorkspaceDraft
  runs: RunSummary[]
  targets: TargetAsset[]
  artifacts: ArtifactSummary[]
  timeline: TimelineEvent[]
  candidates: CandidateSolution[]
  constraints: ConstraintCheck[]
  exportEstimate: ExportEstimate
  activeModel: ActiveModelInfo
  agentConfiguration: AgentConfigurationSummary
  decisionSupport: DecisionSupport
}

export interface WorkspaceService {
  listRuns(): Promise<RunSummary[]>
  getAlgorithmOverview(): Promise<AlgorithmOverview>
  getWorkspaceDetail(runId?: string): Promise<WorkspaceDetail>
  getArtifactDetail(artifactId: string): Promise<ArtifactDetail>
  createDesignRun(input: DesignRequest): Promise<DesignRunResponse>
}
