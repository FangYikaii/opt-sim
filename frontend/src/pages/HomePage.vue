<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import AlgorithmOverviewPanel from '../components/AlgorithmOverviewPanel.vue'
import ExperimentGuide from '../components/ExperimentGuide.vue'
import OperationsGuidePanel from '../components/OperationsGuidePanel.vue'
import { useI18n } from '../i18n'
import { workspaceService } from '../services/workspace-service'
import type { AlgorithmOverview, DesignRequest } from '../api/contracts'
import type { WorkflowStepId } from '../workflow'

const router = useRouter()
const isSubmitting = ref(false)
const isLoadingOverview = ref(true)
const errorMessage = ref('')
const overviewError = ref('')
const algorithmOverview = ref<AlgorithmOverview | null>(null)
const { locale, toggleLocale, t } = useI18n()
const form = ref<DesignRequest>({
  requirementText: t.value('home.form.defaultRequirement'),
  targetHex: '#bf6f4f',
  topK: 3,
  thetaDeg: 0,
  polarization: 'unpolarized',
})
const requirementTouched = ref(false)
const guideSteps = ref([
  { id: 'brief' as WorkflowStepId, title: '', detail: '' },
  { id: 'generate' as WorkflowStepId, title: '', detail: '' },
  { id: 'select' as WorkflowStepId, title: '', detail: '' },
  { id: 'export' as WorkflowStepId, title: '', detail: '' },
])

function syncGuideSteps(): void {
  guideSteps.value = [
    {
      id: 'brief',
      title: t.value('workflow.step.brief.title'),
      detail: t.value('workflow.step.brief.detail'),
    },
    {
      id: 'generate',
      title: t.value('workflow.step.generate.title'),
      detail: t.value('workflow.step.generate.detail'),
    },
    {
      id: 'select',
      title: t.value('workflow.step.select.title'),
      detail: t.value('workflow.step.select.detail'),
    },
    {
      id: 'export',
      title: t.value('workflow.step.export.title'),
      detail: t.value('workflow.step.export.detail'),
    },
  ]
}

async function loadOverview(): Promise<void> {
  isLoadingOverview.value = true
  overviewError.value = ''
  try {
    algorithmOverview.value = await workspaceService.getAlgorithmOverview()
  } catch (error) {
    overviewError.value = error instanceof Error ? error.message : t.value('home.error.overview')
  } finally {
    isLoadingOverview.value = false
  }
}

onMounted(async () => {
  syncGuideSteps()
  await loadOverview()
})

watch(locale, () => {
  syncGuideSteps()
  if (!requirementTouched.value) {
    form.value.requirementText = t.value('home.form.defaultRequirement')
  }
})

async function submitDemoRun(): Promise<void> {
  isSubmitting.value = true
  errorMessage.value = ''

  try {
    const response = await workspaceService.createDesignRun(form.value)
    await router.push({ name: 'run-workspace', params: { runId: response.activeRun.id } })
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : t.value('home.error.run')
  } finally {
    isSubmitting.value = false
  }
}
</script>

<template>
  <section class="route-page home-page panel panel--soft">
    <div class="home-toolbar">
      <div>
        <p class="eyebrow">{{ t('home.eyebrow') }}</p>
        <p class="section-subtitle">{{ t('home.toolbarMeta') }}</p>
      </div>
      <button class="button button--quiet" type="button" @click="toggleLocale">
        {{ t('app.switch') }} · {{ locale === 'zh' ? 'EN' : '中' }}
      </button>
    </div>

    <div class="home-hero">
      <h1>{{ t('home.title') }}</h1>
      <p class="home-hero__body">{{ t('home.body') }}</p>
    </div>

    <section class="summary-banner">
      <h2>{{ t('home.summaryTitle') }}</h2>
      <p>{{ t('home.summaryBody') }}</p>
    </section>

    <ExperimentGuide
      :eyebrow="t('guide.home.eyebrow')"
      :title="t('guide.home.title')"
      :subtitle="t('guide.home.subtitle')"
      :steps="guideSteps"
      current-step="brief"
    />

    <p v-if="overviewError" class="error-text">{{ overviewError }}</p>

    <AlgorithmOverviewPanel
      v-if="algorithmOverview"
      :overview="algorithmOverview"
    />

    <section v-else-if="isLoadingOverview" class="chart-card">
      <p class="eyebrow">{{ t('common.loading') }}</p>
      <h2>{{ t('home.loading') }}</h2>
      <p class="chart-note">{{ t('home.loadingNote') }}</p>
    </section>

    <form class="demo-form" @submit.prevent="submitDemoRun">
      <div class="section-header">
        <div>
          <h2>{{ t('home.form.cardTitle') }}</h2>
          <p class="section-subtitle">{{ t('home.form.cardBody') }}</p>
        </div>
      </div>

      <label class="field">
        <span class="field__label">{{ t('home.form.requirement') }}</span>
        <textarea
          v-model="form.requirementText"
          class="input input--area"
          rows="5"
          spellcheck="false"
          :placeholder="t('home.form.requirementPlaceholder')"
          @input="requirementTouched = true"
        />
      </label>

      <div class="demo-form__grid">
        <label class="field">
          <span class="field__label">{{ t('home.form.targetHex') }}</span>
          <input v-model="form.targetHex" class="input mono" />
        </label>
        <label class="field">
          <span class="field__label">{{ t('home.form.thetaDeg') }}</span>
          <input v-model.number="form.thetaDeg" class="input mono" type="number" min="0" max="89" step="0.1" />
        </label>
        <label class="field">
          <span class="field__label">{{ t('home.form.polarization') }}</span>
          <select v-model="form.polarization" class="input">
            <option value="unpolarized">{{ t('polarization.unpolarized') }}</option>
            <option value="te">{{ t('polarization.te') }}</option>
            <option value="tm">{{ t('polarization.tm') }}</option>
          </select>
        </label>
        <label class="field">
          <span class="field__label">{{ t('home.form.topK') }}</span>
          <input v-model.number="form.topK" class="input mono" type="number" min="1" max="8" />
        </label>
      </div>

      <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
      <p class="section-subtitle">{{ t('home.form.hint') }}</p>

      <div class="button-row">
        <button class="button button--primary" type="submit" :disabled="isSubmitting">
          {{ isSubmitting ? t('home.form.submitting') : t('home.form.submit') }}
        </button>
      </div>
    </form>

    <OperationsGuidePanel
      v-if="algorithmOverview"
      :steps="algorithmOverview.operationSteps"
      :title="t('ops.homeTitle')"
    />
  </section>
</template>
