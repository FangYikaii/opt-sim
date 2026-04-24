<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from '../i18n'
import { getWorkflowStepIndex, type WorkflowStepId } from '../workflow'

interface GuideStep {
  id: WorkflowStepId
  title: string
  detail: string
  actionLabel?: string
}

const props = defineProps<{
  steps: GuideStep[]
  currentStep: WorkflowStepId
  activeStep?: WorkflowStepId
  title: string
  subtitle?: string
  eyebrow?: string
  interactive?: boolean
}>()

const emit = defineEmits<{
  stepChange: [stepId: WorkflowStepId]
}>()

const { t } = useI18n()

const resolvedSteps = computed(() =>
  props.steps.map((step, index) => {
    const currentIndex = getWorkflowStepIndex(props.currentStep)
    const activeIndex = getWorkflowStepIndex(props.activeStep ?? props.currentStep)
    const isComplete = index < currentIndex
    const isCurrent = index === currentIndex
    const isActive = index === activeIndex
    const isAccessible = index <= currentIndex
    const state = isComplete ? 'complete' : isCurrent ? 'current' : 'pending'
    return {
      ...step,
      index: index + 1,
      state,
      isActive,
      isAccessible,
      stateLabel: t.value(`guide.state.${state}`),
    }
  }),
)
</script>

<template>
  <section class="guide-card panel panel--soft">
    <div class="section-header">
      <div>
        <p v-if="eyebrow" class="eyebrow">{{ eyebrow }}</p>
        <h2>{{ title }}</h2>
        <p v-if="subtitle" class="section-subtitle">{{ subtitle }}</p>
      </div>
      <span class="ghost-pill">{{ steps.length }} {{ t('common.steps') }}</span>
    </div>

    <ol class="list-reset guide-list">
      <li
        v-for="step in resolvedSteps"
        :key="step.id"
        class="guide-step"
        :data-state="step.state"
        :data-active="step.isActive"
      >
        <button
          class="guide-step__button"
          type="button"
          :disabled="interactive && !step.isAccessible"
          :aria-current="step.isActive ? 'step' : undefined"
          @click="interactive ? emit('stepChange', step.id) : undefined"
        >
          <div class="guide-step__index mono">{{ step.index }}</div>
          <div class="guide-step__body">
            <div class="guide-step__header">
              <h3>{{ step.title }}</h3>
              <span
                class="status-pill status-pill--small"
                :data-state="
                  step.state === 'complete'
                    ? 'Complete'
                    : step.state === 'current'
                      ? 'Needs approval'
                      : 'pending'
                "
              >
                {{ step.stateLabel }}
              </span>
            </div>
            <p>{{ step.detail }}</p>
            <span v-if="step.actionLabel" class="guide-step__action mono">{{ step.actionLabel }}</span>
          </div>
        </button>
      </li>
    </ol>
  </section>
</template>
