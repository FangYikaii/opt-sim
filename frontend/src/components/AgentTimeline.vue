<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import { useI18n } from '../i18n'
import type { ExportEstimate, RunSummary, TimelineEvent, WorkspaceDraft } from '../api/contracts'
import {
  getNextWorkflowStep,
  resolveWorkflowStep,
  type WorkflowStepId,
} from '../workflow'

const props = defineProps<{
  draft: WorkspaceDraft
  activeRun: RunSummary
  events: TimelineEvent[]
  stepId: WorkflowStepId
  candidateCount?: number
  constraintCount?: number
  exportEstimate?: ExportEstimate
}>()

defineEmits<{
  focusSelection: []
  approveExport: []
}>()

const { labelRunStatus, labelTimelineType, localizeCopy, t } = useI18n()

const currentRunStep = computed(() => resolveWorkflowStep(props.activeRun.status))

function labelWorkflowStep(step: WorkflowStepId): string {
  return t.value(`workflow.step.${step}.title`)
}

const currentStepTitle = computed(() => labelWorkflowStep(currentRunStep.value))

const viewedStepTitle = computed(() => labelWorkflowStep(props.stepId))

const nextStepTitle = computed(() => {
  const nextStep = getNextWorkflowStep(currentRunStep.value)
  return nextStep ? labelWorkflowStep(nextStep) : t.value('top.action.complete')
})

const compactEvents = computed(() => props.events.slice(0, 4))
const isBriefStep = computed(() => props.stepId === 'brief')
const isGenerateStep = computed(() => props.stepId === 'generate')
const isSelectStep = computed(() => props.stepId === 'select')
const isExportStep = computed(() => props.stepId === 'export')
</script>

<template>
  <section class="timeline-layout">
    <article class="panel panel--soft composer-card">
      <div class="section-header">
        <div>
          <p class="eyebrow">{{ t('timeline.statusEyebrow') }}</p>
          <h2>{{ viewedStepTitle }}</h2>
        </div>
        <span class="ghost-pill">{{ draft.targetValue }}</span>
      </div>
      <template v-if="isBriefStep">
        <p class="composer-card__lead">{{ t('workflow.panel.briefLead') }}</p>
        <div class="composer-grid">
          <label class="field">
            <span class="field__label">{{ t('timeline.targetLabel') }}</span>
            <div class="field__value field__value--swatch">
              <span class="swatch-chip" style="background-color: #bf6f4f" aria-hidden="true" />
              <span class="mono">{{ props.draft.targetValue }}</span>
            </div>
          </label>
          <label class="field">
            <span class="field__label">{{ t('timeline.heightWindow') }}</span>
            <span class="field__value mono">{{ props.draft.heightWindow }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('timeline.exportMode') }}</span>
            <span class="field__value mono">{{ localizeCopy(props.draft.exportMode) }}</span>
          </label>
        </div>
        <label class="field">
          <span class="field__label">{{ t('timeline.requirement') }}</span>
          <p class="field__value">{{ localizeCopy(draft.requirementText) }}</p>
        </label>
        <div class="button-row">
          <RouterLink class="button button--primary button--link" :to="{ name: 'home' }">
            {{ t('timeline.edit') }}
          </RouterLink>
        </div>
      </template>

      <template v-else-if="isGenerateStep">
        <div class="composer-status-grid">
          <label class="field">
            <span class="field__label">{{ t('timeline.currentStep') }}</span>
            <span class="field__value">{{ currentStepTitle }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('timeline.nextStep') }}</span>
            <span class="field__value">{{ nextStepTitle }}</span>
          </label>
        </div>
        <div class="composer-grid">
          <label class="field">
            <span class="field__label">{{ t('timeline.targetLabel') }}</span>
            <div class="field__value field__value--swatch">
              <span class="swatch-chip" style="background-color: #bf6f4f" aria-hidden="true" />
              <span class="mono">{{ props.draft.targetValue }}</span>
            </div>
          </label>
          <label class="field">
            <span class="field__label">{{ t('timeline.heightWindow') }}</span>
            <span class="field__value mono">{{ props.draft.heightWindow }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('timeline.exportMode') }}</span>
            <span class="field__value mono">{{ localizeCopy(props.draft.exportMode) }}</span>
          </label>
        </div>
      </template>

      <template v-else-if="isSelectStep">
        <p class="composer-card__lead">{{ t('workflow.panel.selectLead') }}</p>
        <div class="composer-status-grid">
          <label class="field">
            <span class="field__label">{{ t('timeline.currentStep') }}</span>
            <span class="field__value">{{ currentStepTitle }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('inspector.candidates') }}</span>
            <span class="field__value mono">{{ candidateCount ?? 0 }} {{ t('common.results') }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('inspector.constraints') }}</span>
            <span class="field__value mono">{{ constraintCount ?? 0 }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('timeline.targetLabel') }}</span>
            <span class="field__value mono">{{ draft.targetValue }}</span>
          </label>
        </div>
        <label class="field">
          <span class="field__label">{{ t('timeline.requirement') }}</span>
          <p class="field__value">{{ localizeCopy(draft.requirementText) }}</p>
        </label>
        <div class="button-row">
          <button class="button button--primary" type="button" @click="$emit('focusSelection')">
            {{ t('workflow.step.select.action') }}
          </button>
        </div>
      </template>

      <template v-else-if="isExportStep && exportEstimate">
        <p class="composer-card__lead">{{ t('workflow.panel.exportLead') }}</p>
        <div class="composer-status-grid">
          <label class="field">
            <span class="field__label">{{ t('timeline.currentStep') }}</span>
            <span class="field__value">{{ currentStepTitle }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('timeline.nextStep') }}</span>
            <span class="field__value">{{ nextStepTitle }}</span>
          </label>
        </div>
        <div class="composer-grid">
          <label class="field">
            <span class="field__label">{{ t('inspector.export.dimensions') }}</span>
            <span class="field__value mono">{{ exportEstimate.dimensions }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('inspector.export.tilePlan') }}</span>
            <span class="field__value mono">{{ localizeCopy(exportEstimate.tilePlan) }}</span>
          </label>
          <label class="field">
            <span class="field__label">{{ t('inspector.export.format') }}</span>
            <span class="field__value mono">{{ localizeCopy(exportEstimate.format) }}</span>
          </label>
        </div>
        <div class="button-row">
          <button class="button button--primary" type="button" @click="$emit('approveExport')">
            {{ t('workflow.step.export.action') }}
          </button>
        </div>
      </template>
    </article>

    <article v-if="isGenerateStep" class="panel timeline-card">
      <div class="section-header">
        <div>
          <h2>{{ t('timeline.activity') }}</h2>
          <p class="section-subtitle">{{ t('timeline.activityMeta') }} · {{ t('common.run') }} {{ props.activeRun.id }}</p>
        </div>
        <span class="status-pill" :data-state="labelRunStatus(props.activeRun.status).state">
          {{ labelRunStatus(props.activeRun.status).text }}
        </span>
      </div>

      <ol class="list-reset timeline-list">
        <li v-for="event in compactEvents" :key="event.id" class="timeline-event" :data-type="event.type">
          <div class="timeline-event__marker" aria-hidden="true" />
          <div class="timeline-event__card">
            <div class="timeline-event__topline">
              <span class="ghost-pill">{{ labelTimelineType(event.type, event.label) }}</span>
              <span class="timeline-event__meta">{{ event.meta ? localizeCopy(event.meta) : '' }}</span>
            </div>
            <h3>{{ localizeCopy(event.title) }}</h3>
            <p class="timeline-event__body">{{ localizeCopy(event.body) }}</p>
            <div class="timeline-event__footer">
              <span
                v-if="event.status"
                class="status-pill status-pill--small"
                :data-state="labelRunStatus(event.status).state"
              >
                {{ labelRunStatus(event.status).text }}
              </span>
              <button
                v-if="event.actionLabel"
                class="button button--primary button--small"
                type="button"
                @click="$emit('approveExport')"
              >
                {{ t('timeline.review') }}
              </button>
            </div>
          </div>
        </li>
      </ol>
    </article>
  </section>
</template>
