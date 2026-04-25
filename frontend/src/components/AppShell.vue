<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from '../i18n'
import type {
  AlgorithmOverview,
  ArtifactDetail,
  ArtifactSummary,
  CandidateSolution,
  ConstraintCheck,
  ExportEstimate,
  RunSummary,
  TargetAsset,
  TimelineEvent,
  WorkspaceDraft,
  WorkspaceProject,
} from '../api/contracts'
import {
  getNextWorkflowStep,
  resolveWorkflowStep,
  type WorkflowStepId,
} from '../workflow'
import TopBar from './TopBar.vue'
import LeftRail from './LeftRail.vue'
import AgentTimeline from './AgentTimeline.vue'
import InspectorPanel from './InspectorPanel.vue'
import BottomDrawer from './BottomDrawer.vue'
import ExperimentGuide from './ExperimentGuide.vue'
import ArtifactDetailPanel from './ArtifactDetailPanel.vue'

const props = defineProps<{
  algorithmOverview: AlgorithmOverview | null
  project: WorkspaceProject
  draft: WorkspaceDraft
  activeRun: RunSummary
  runs: RunSummary[]
  targets: TargetAsset[]
  artifacts: ArtifactSummary[]
  activeArtifactId: string | null
  activeArtifactDetail: ArtifactDetail | null
  isArtifactLoading: boolean
  timeline: TimelineEvent[]
  candidates: CandidateSolution[]
  constraints: ConstraintCheck[]
  exportEstimate: ExportEstimate
}>()

const emit = defineEmits<{
  selectArtifact: [artifactId: string]
}>()

const { t } = useI18n()
const activeStep = ref<WorkflowStepId>('brief')

function scrollToExport(): void {
  document.querySelector('#export-panel')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

function scrollToCandidates(): void {
  document.querySelector('.candidate-list')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function focusCurrentStep(): void {
  if (currentStep.value === 'select') {
    scrollToCandidates()
    return
  }
  if (currentStep.value === 'export') {
    scrollToExport()
  }
}

const currentStep = computed(() => resolveWorkflowStep(props.activeRun.status))

watch(
  currentStep,
  (step) => {
    activeStep.value = step
  },
  { immediate: true },
)

const nextStep = computed(() => getNextWorkflowStep(currentStep.value))

const guideSteps = computed(() => [
  {
    id: 'brief' as WorkflowStepId,
    title: t.value('workflow.step.brief.title'),
    detail: t.value('workflow.step.brief.detail'),
    actionLabel: t.value('workflow.step.brief.action'),
  },
  {
    id: 'generate' as WorkflowStepId,
    title: t.value('workflow.step.generate.title'),
    detail: t.value('workflow.step.generate.detail'),
    actionLabel: t.value('workflow.step.generate.action'),
  },
  {
    id: 'select' as WorkflowStepId,
    title: t.value('workflow.step.select.title'),
    detail: t.value('workflow.step.select.detail'),
    actionLabel: t.value('workflow.step.select.action'),
  },
  {
    id: 'export' as WorkflowStepId,
    title: t.value('workflow.step.export.title'),
    detail: t.value('workflow.step.export.detail'),
    actionLabel: t.value('workflow.step.export.action'),
  },
])

const isBriefStep = computed(() => activeStep.value === 'brief')
const isGenerateStep = computed(() => activeStep.value === 'generate')
const isSelectStep = computed(() => activeStep.value === 'select')
const isExportStep = computed(() => activeStep.value === 'export')
</script>

<template>
  <div class="app-shell">
    <TopBar
      :project="project"
      :active-run="activeRun"
      :current-step="currentStep"
      :active-step="activeStep"
      :next-step="nextStep"
      @review-export="focusCurrentStep"
    />
    <div class="workspace-grid">
      <LeftRail
        :runs="runs"
        :targets="targets"
        :artifacts="artifacts"
        :active-artifact-id="activeArtifactId"
        @select-artifact="emit('selectArtifact', $event)"
      />
      <main class="workspace-main" :aria-label="t('shell.workspaceAria')">
        <ExperimentGuide
          class="workspace-guide"
          :eyebrow="t('guide.run.eyebrow')"
          :title="t('guide.run.title')"
          :subtitle="t('guide.run.subtitle')"
          :steps="guideSteps"
          :current-step="currentStep"
          :active-step="activeStep"
          interactive
          @step-change="activeStep = $event"
        />
        <AgentTimeline
          v-if="isBriefStep || isGenerateStep || isSelectStep || isExportStep"
          :draft="draft"
          :active-run="activeRun"
          :events="timeline"
          :step-id="activeStep"
          :candidate-count="candidates.length"
          :constraint-count="constraints.length"
          :export-estimate="exportEstimate"
          @focus-selection="scrollToCandidates"
          @approve-export="focusCurrentStep"
        />
        <ArtifactDetailPanel
          :artifacts="artifacts"
          :active-artifact-id="activeArtifactId"
          :active-artifact-detail="activeArtifactDetail"
          :is-loading="isArtifactLoading"
          @select-artifact="emit('selectArtifact', $event)"
        />
      </main>
      <InspectorPanel
        :active-step="activeStep"
        :algorithm-overview="algorithmOverview"
        :candidate="candidates[0]"
        :candidates="candidates"
        :constraints="constraints"
        :export-estimate="exportEstimate"
      />
    </div>
    <BottomDrawer
      v-if="isGenerateStep || isExportStep"
      :active-run="activeRun"
      :draft="draft"
      :export-estimate="exportEstimate"
    />
  </div>
</template>
