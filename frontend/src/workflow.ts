export type WorkflowStepId = 'brief' | 'generate' | 'select' | 'export'

export const workflowStepOrder: WorkflowStepId[] = ['brief', 'generate', 'select', 'export']

export function getWorkflowStepIndex(step: WorkflowStepId): number {
  return workflowStepOrder.indexOf(step)
}

export function getNextWorkflowStep(step: WorkflowStepId): WorkflowStepId | null {
  const currentIndex = getWorkflowStepIndex(step)
  return workflowStepOrder[currentIndex + 1] ?? null
}

export function resolveWorkflowStep(status: string): WorkflowStepId {
  if (status === 'Draft' || status === 'Validating') {
    return 'brief'
  }
  if (
    status === 'Simulating' ||
    status === 'Ranking' ||
    status === 'Running' ||
    status === 'Failed'
  ) {
    return 'generate'
  }
  if (status === 'Needs approval') {
    return 'select'
  }
  if (status === 'Exporting' || status === 'Complete') {
    return 'export'
  }
  return 'brief'
}
