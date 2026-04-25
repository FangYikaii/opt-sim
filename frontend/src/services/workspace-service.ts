import type {
  ApiErrorPayload,
  AlgorithmOverview,
  ArtifactDetail,
  DesignRequest,
  DesignRunResponse,
  RunSummary,
  WorkspaceDetail,
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
    return requestJson<WorkspaceDetail>(`/api/runs/${resolvedRunId}/workspace`)
  },

  async getArtifactDetail(artifactId: string): Promise<ArtifactDetail> {
    return requestJson<ArtifactDetail>(`/api/artifacts/${artifactId}`)
  },

  async createDesignRun(input: DesignRequest): Promise<DesignRunResponse> {
    return requestJsonWithBody<DesignRunResponse>('/api/agent/design-run', input)
  },
}
