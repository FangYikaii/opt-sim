import type {
  ApiErrorPayload,
  AlgorithmOverview,
  ArtifactDetail,
  ArtifactSummary,
  CandidateSolution,
  ConstraintCheck,
  DesignRequest,
  DesignRunResponse,
  ExportEstimate,
  RunSummary,
  TargetAsset,
  TimelineEvent,
  WorkspaceDraft,
  WorkspaceDetail,
  WorkspaceProject,
  WorkspaceService,
} from '../api/contracts'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

class ApiClientError extends Error {
  code: string
  details?: unknown

  constructor(payload: ApiErrorPayload) {
    super(payload.error.message)
    this.name = 'ApiClientError'
    this.code = payload.error.code
    this.details = payload.error.details
  }
}

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    const payload = (await response.json()) as ApiErrorPayload | { detail?: ApiErrorPayload }
    if ('error' in payload) {
      throw new ApiClientError(payload)
    }
    if (payload.detail && 'error' in payload.detail) {
      throw new ApiClientError(payload.detail)
    }
    throw new Error(`Request failed with status ${response.status}`)
  }

  return (await response.json()) as T
}

async function requestJsonWithBody<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const payload = (await response.json()) as ApiErrorPayload | { detail?: ApiErrorPayload }
    if ('error' in payload) {
      throw new ApiClientError(payload)
    }
    if (payload.detail && 'error' in payload.detail) {
      throw new ApiClientError(payload.detail)
    }
    throw new Error(`Request failed with status ${response.status}`)
  }

  return (await response.json()) as T
}

async function getDefaultRunId(): Promise<string> {
  const runs = await requestJson<RunSummary[]>('/api/runs')
  if (runs.length === 0) {
    throw new Error('No runs available')
  }
  return runs[0].id
}

export const workspaceService: WorkspaceService = {
  async listRuns(): Promise<RunSummary[]> {
    return requestJson<RunSummary[]>('/api/runs')
  },

  async getAlgorithmOverview(): Promise<AlgorithmOverview> {
    return requestJson<AlgorithmOverview>('/api/algorithm-overview')
  },

  async getWorkspaceDetail(runId?: string): Promise<WorkspaceDetail> {
    const resolvedRunId = runId ?? (await getDefaultRunId())

    const [
      project,
      runs,
      activeRun,
      draft,
      targets,
      timeline,
      candidates,
      constraints,
      exportEstimate,
      artifacts,
    ] = await Promise.all([
      requestJson<WorkspaceProject>('/api/project'),
      requestJson<RunSummary[]>('/api/runs'),
      requestJson<RunSummary>(`/api/runs/${resolvedRunId}`),
      requestJson<WorkspaceDraft>(`/api/runs/${resolvedRunId}/draft`),
      requestJson<TargetAsset[]>(`/api/runs/${resolvedRunId}/targets`),
      requestJson<TimelineEvent[]>(`/api/runs/${resolvedRunId}/timeline`),
      requestJson<CandidateSolution[]>(`/api/runs/${resolvedRunId}/candidates`),
      requestJson<ConstraintCheck[]>(`/api/runs/${resolvedRunId}/constraints`),
      requestJson<ExportEstimate>(`/api/runs/${resolvedRunId}/export-estimate`),
      requestJson<ArtifactSummary[]>(`/api/runs/${resolvedRunId}/artifacts`),
    ])

    return {
      project,
      activeRun,
      draft,
      runs,
      targets,
      artifacts,
      timeline,
      candidates,
      constraints,
      exportEstimate,
    }
  },

  async getArtifactDetail(artifactId: string): Promise<ArtifactDetail> {
    return requestJson<ArtifactDetail>(`/api/artifacts/${artifactId}`)
  },

  async createDesignRun(input: DesignRequest): Promise<DesignRunResponse> {
    return requestJsonWithBody<DesignRunResponse>('/api/agent/design-run', input)
  },
}
