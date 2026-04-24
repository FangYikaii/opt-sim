<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import { useI18n } from '../i18n'
import type { RunSummary, WorkspaceProject } from '../api/contracts'
import type { WorkflowStepId } from '../workflow'

const props = defineProps<{
  project: WorkspaceProject
  activeRun: RunSummary
  currentStep: WorkflowStepId
  activeStep: WorkflowStepId
  nextStep: WorkflowStepId | null
}>()

defineEmits<{
  reviewExport: []
}>()

const { labelRunStatus, localizeCopy, locale, toggleLocale, t } = useI18n()

function labelWorkflowStep(step: WorkflowStepId | null): string {
  if (!step) {
    return t.value('top.action.complete')
  }
  return t.value(`workflow.step.${step}.title`)
}

const nextStepLabel = computed(() => labelWorkflowStep(props.nextStep))
const currentStepLabel = computed(() => labelWorkflowStep(props.currentStep))
const activeStepLabel = computed(() => labelWorkflowStep(props.activeStep))

const primaryActionLabel = computed(() => {
  if (props.currentStep === 'select') {
    return t.value('workflow.step.select.action')
  }
  if (props.currentStep === 'export') {
    return t.value('workflow.step.export.action')
  }
  return t.value('top.action.progress')
})
</script>

<template>
  <header class="top-bar panel">
    <div class="top-bar__identity">
      <p class="eyebrow">{{ t('top.eyebrow') }}</p>
      <div>
        <h1>{{ localizeCopy(project.name) }}</h1>
        <p class="muted">
          {{ t('top.currentRun') }}:
          <span class="mono">{{ localizeCopy(activeRun.title) }}</span>
        </p>
        <p class="muted">
          {{ t('timeline.currentStep') }}:
          <span class="mono">{{ currentStepLabel }}</span>
        </p>
        <p class="muted">
          {{ t('workflow.viewing') }}:
          <span class="mono">{{ activeStepLabel }}</span>
        </p>
        <p class="muted">
          {{ t('top.next') }}:
          <span class="mono">{{ nextStepLabel }}</span>
        </p>
      </div>
    </div>
    <div class="top-bar__actions">
      <div class="status-cluster">
        <span class="status-pill" :data-state="labelRunStatus(activeRun.status).state">
          {{ labelRunStatus(activeRun.status).text }}
        </span>
        <span class="ghost-pill">{{ t('top.runId') }} {{ activeRun.id }}</span>
      </div>
      <div class="button-row">
        <button class="button button--quiet" type="button" @click="toggleLocale">
          {{ t('app.switch') }} · {{ locale === 'zh' ? 'EN' : '中' }}
        </button>
        <RouterLink class="button button--quiet button--link" :to="{ name: 'home' }">
          {{ t('top.newRun') }}
        </RouterLink>
        <button class="button button--primary" type="button" @click="$emit('reviewExport')">
          {{ primaryActionLabel }}
        </button>
      </div>
    </div>
  </header>
</template>
