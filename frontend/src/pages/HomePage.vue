<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from '../i18n'
import { workspaceService } from '../services/workspace-service'
import type { AlgorithmOverview, DesignRequest } from '../api/contracts'

const router = useRouter()
const isSubmitting = ref(false)
const isLoadingOverview = ref(true)
const errorMessage = ref('')
const overviewError = ref('')
const algorithmOverview = ref<AlgorithmOverview | null>(null)
const { labelMetric, labelMetricState, labelMetricValue, locale, localizeCopy, toggleLocale, t } = useI18n()
const form = ref<DesignRequest>({
  requirementText: t.value('home.form.defaultRequirement'),
  targetHex: '#bf6f4f',
  topK: 3,
  thetaDeg: 0,
  polarization: 'unpolarized',
  designMode: 'structural-color',
})
const requirementTouched = ref(false)
const submittingSteps = computed(() => [
  t.value('home.form.submittingStep.generate'),
  t.value('home.form.submittingStep.simulate'),
  t.value('home.form.submittingStep.decision'),
])

const overviewCards = computed(() => {
  const overview = algorithmOverview.value
  if (!overview) {
    return []
  }

  return [
    {
      label: '当前生产模型',
      value: overview.activeModel?.checkpointFile ?? overview.activeModel?.label ?? '未检测到',
      detail: overview.activeModel?.summary ?? '尚未识别到可用模型。',
    },
    {
      label: 'AI 决策代理',
      value: `${overview.agentConfiguration?.providerLabel ?? '未配置'} · ${overview.agentConfiguration?.model ?? 'n/a'}`,
      detail: overview.agentConfiguration?.summary ?? '尚未识别到代理配置。',
    },
    {
      label: '最佳复现实验',
      value:
        overview.bestExperiment?.id ??
        overview.latestExperiment?.id ??
        '暂无训练记录',
      detail: overview.currentAssessment,
    },
  ]
})

const headlineMetrics = computed(() => algorithmOverview.value?.headlineMetrics.slice(0, 4) ?? [])

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
  await loadOverview()
})

watch(locale, () => {
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
  <section class="route-page home-flow">
    <header class="flow-hero panel panel--soft">
      <div class="flow-hero__main">
        <p class="eyebrow">{{ t('home.eyebrow') }}</p>
        <h1>{{ t('home.title') }}</h1>
        <p class="flow-hero__body">{{ t('home.body') }}</p>
      </div>
      <div class="button-row">
        <button class="button button--quiet" type="button" @click="toggleLocale">
          {{ t('app.switch') }} · {{ locale === 'zh' ? 'EN' : '中' }}
        </button>
      </div>
    </header>

    <section class="flow-layout">
      <article class="panel flow-section">
        <div class="section-header">
          <div>
            <p class="eyebrow">Step 1</p>
            <h2>确认当前系统状态</h2>
            <p class="section-subtitle">先判断最佳模型、代理配置和当前训练可用度。</p>
          </div>
          <span v-if="algorithmOverview?.activeModel" class="status-pill" :data-state="algorithmOverview.activeModel.status === 'ready' ? 'pass' : 'warning'">
            {{ algorithmOverview.activeModel.status === 'ready' ? 'Ready' : 'Fallback' }}
          </span>
        </div>

        <p v-if="overviewError" class="error-text">{{ overviewError }}</p>

        <div v-if="isLoadingOverview" class="chart-card">
          <p class="eyebrow">{{ t('common.loading') }}</p>
          <p class="chart-note">{{ t('home.loadingNote') }}</p>
        </div>

        <template v-else-if="algorithmOverview">
          <div class="overview-card-grid">
            <article v-for="card in overviewCards" :key="card.label" class="chart-card">
              <span class="field__label">{{ card.label }}</span>
              <h3 class="overview-card__value">{{ card.value }}</h3>
              <p class="chart-note">{{ localizeCopy(card.detail) }}</p>
            </article>
          </div>

          <div class="headline-metric-grid">
            <article v-for="metric in headlineMetrics" :key="metric.label" class="metric-panel">
              <div class="metric-panel__top">
                <span class="field__label">{{ labelMetric(metric.label) }}</span>
                <span class="status-pill status-pill--small" :data-state="labelMetricState(metric.state).state">
                  {{ labelMetricState(metric.state).text }}
                </span>
              </div>
              <strong class="metric-panel__value mono">{{ labelMetricValue(metric.label, metric.value) }}</strong>
              <p class="chart-note">{{ localizeCopy(metric.detail) }}</p>
            </article>
          </div>
        </template>
      </article>

      <article class="panel flow-section">
        <div class="section-header">
          <div>
            <p class="eyebrow">Step 2</p>
            <h2>{{ t('home.form.cardTitle') }}</h2>
            <p class="section-subtitle">{{ t('home.form.cardBody') }}</p>
          </div>
        </div>

        <form class="flow-form" @submit.prevent="submitDemoRun">
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

          <div class="flow-form__grid">
            <label class="field">
              <span class="field__label">设计模式</span>
              <select v-model="form.designMode" class="input">
                <option value="structural-color">结构色薄膜反演</option>
                <option value="neural-holography">Neural Holography / CITL</option>
              </select>
            </label>
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
      </article>
    </section>

    <div
      v-if="isSubmitting"
      class="submission-overlay"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <article class="submission-card panel">
        <div class="submission-card__header">
          <span class="status-pill" data-state="Simulating">{{ t('home.form.submitting') }}</span>
          <div class="submission-spinner" aria-hidden="true" />
        </div>
        <div class="submission-card__body">
          <h2>{{ t('home.form.submittingTitle') }}</h2>
          <p class="chart-note">{{ t('home.form.submittingBody') }}</p>
        </div>
        <ul class="list-reset checklist submission-checklist">
          <li v-for="step in submittingSteps" :key="step" class="checklist-item">
            <div class="submission-step">
              <span class="submission-step__dot" aria-hidden="true" />
              <p>{{ step }}</p>
            </div>
          </li>
        </ul>
        <p class="section-subtitle">{{ t('home.form.submittingHint') }}</p>
      </article>
    </div>
  </section>
</template>
