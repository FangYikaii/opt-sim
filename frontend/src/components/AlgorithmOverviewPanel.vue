<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from '../i18n'
import type { AlgorithmOverview } from '../api/contracts'

const props = defineProps<{
  overview: AlgorithmOverview
  compact?: boolean
}>()

const heroExperiment = computed(() => props.overview.bestExperiment ?? props.overview.latestExperiment)
const { labelMetric, labelMetricState, labelMetricValue, labelStage, labelWinner, localizeCopy, t } =
  useI18n()
</script>

<template>
  <section class="algorithm-overview panel panel--soft">
    <div class="section-header">
      <div>
        <p class="eyebrow">{{ t('algo.eyebrow') }}</p>
        <h2>{{ t('algo.title') }}</h2>
        <p class="section-subtitle">{{ localizeCopy(overview.businessGoal) }}</p>
      </div>
      <span v-if="heroExperiment" class="status-pill" :data-state="labelStage(heroExperiment.stage, heroExperiment.stageState).state">
        {{ labelStage(heroExperiment.stage, heroExperiment.stageState).text }}
      </span>
    </div>

    <div class="algorithm-callout">
      <h3>{{ t('algo.why') }}</h3>
      <p>{{ localizeCopy(overview.businessGoal) }}</p>
    </div>

    <div class="algorithm-summary-grid">
      <article class="summary-card">
        <span class="field__label">{{ t('algo.current') }}</span>
        <p>{{ localizeCopy(overview.currentAssessment) }}</p>
      </article>
      <article class="summary-card">
        <span class="field__label">{{ t('algo.gpu') }}</span>
        <p>{{ localizeCopy(overview.gpuTrainingSummary) }}</p>
      </article>
      <article class="summary-card">
        <span class="field__label">{{ t('algo.training') }}</span>
        <p>{{ localizeCopy(overview.trainingConclusion) }}</p>
      </article>
      <article class="summary-card">
        <span class="field__label">{{ t('algo.environment') }}</span>
        <p>{{ localizeCopy(overview.currentEnvironmentSummary) }}</p>
      </article>
    </div>

    <p class="chart-note">{{ localizeCopy(overview.plainExplanation) }}</p>

    <div class="headline-metric-grid">
      <article v-for="metric in overview.headlineMetrics" :key="metric.label" class="metric-panel">
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

    <div v-if="!compact" class="algorithm-detail-grid">
      <article class="chart-card">
        <div class="chart-card__header">
          <h3>{{ t('algo.targets') }}</h3>
          <span class="ghost-pill ghost-pill--small">{{ overview.targetComparisons.length }} {{ t('common.targets') }}</span>
        </div>
        <div class="comparison-list">
          <div v-for="item in overview.targetComparisons" :key="item.targetHex" class="comparison-row">
            <div class="comparison-row__identity">
              <span class="swatch-chip swatch-chip--large" :style="{ backgroundColor: item.targetHex }" />
              <span class="mono">{{ item.targetHex }}</span>
            </div>
            <div class="comparison-row__metrics">
              <span class="ghost-pill">{{ t('algo.retrieval') }} {{ item.retrievalDeltaE.toFixed(2) }}</span>
              <span class="ghost-pill">{{ t('algo.cgan') }} {{ item.cganDeltaE.toFixed(2) }}</span>
            </div>
            <span class="status-pill status-pill--small" :data-state="item.winner === 'cgan' ? 'pass' : item.winner === 'retrieval' ? 'warning' : 'unknown'">
              {{ labelWinner(item.winner) }}
            </span>
          </div>
        </div>
      </article>

      <article class="chart-card">
        <div class="chart-card__header">
          <h3>{{ t('algo.experiments') }}</h3>
          <span class="ghost-pill ghost-pill--small">{{ overview.experiments.length }} {{ t('common.runs') }}</span>
        </div>
        <div class="experiment-list">
          <div v-for="experiment in overview.experiments" :key="experiment.id" class="experiment-card">
            <div class="section-header">
              <div>
                <h3>{{ experiment.id }}</h3>
                <p class="chart-note">{{ experiment.updatedAt }}</p>
              </div>
              <span class="status-pill status-pill--small" :data-state="labelStage(experiment.stage, experiment.stageState).state">
                {{ labelStage(experiment.stage, experiment.stageState).text }}
              </span>
            </div>
            <div class="candidate-card__params">
              <span class="ghost-pill">{{ t('algo.experiment.epochs') }} {{ experiment.epochs ?? t('common.notAvailable') }}</span>
              <span class="ghost-pill">{{ t('algo.experiment.samples') }} {{ experiment.sampleCount ?? t('common.notAvailable') }}</span>
              <span class="ghost-pill">{{ labelMetricValue('Latest Training Device', experiment.device.toUpperCase()) }}</span>
              <span v-if="experiment.deviceName" class="ghost-pill">{{ experiment.deviceName }}</span>
            </div>
            <p class="chart-note">{{ localizeCopy(experiment.summary) }}</p>
          </div>
        </div>
      </article>
    </div>
  </section>
</template>
