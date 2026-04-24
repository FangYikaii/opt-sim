<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from '../i18n'
import type {
  AlgorithmOverview,
  CandidateSolution,
  ConstraintCheck,
  ExportEstimate,
} from '../api/contracts'
import type { WorkflowStepId } from '../workflow'

const props = defineProps<{
  activeStep: WorkflowStepId
  algorithmOverview: AlgorithmOverview | null
  candidate: CandidateSolution
  candidates: CandidateSolution[]
  constraints: ConstraintCheck[]
  exportEstimate: ExportEstimate
}>()

const selectedCandidateId = ref(props.candidate.id)
const selectedCandidate = computed(
  () => props.candidates.find((item) => item.id === selectedCandidateId.value) ?? props.candidate,
)
const deltaEMetric = computed(
  () => selectedCandidate.value.metrics.find((metric) => metric.label === 'DeltaE')?.value ?? 'n/a',
)
const keyMetrics = computed(() =>
  selectedCandidate.value.metrics.filter((metric) => ['DeltaE', 'Composite score', 'Manufacturability'].includes(metric.label)),
)
const topConstraints = computed(() => props.constraints.slice(0, 3))
const {
  labelCandidateStatus,
  labelConstraint,
  labelConstraintState,
  labelMetric,
  labelMetricValue,
  labelParameter,
  labelStage,
  localizeCopy,
  t,
} = useI18n()

const isBriefStep = computed(() => props.activeStep === 'brief')
const isGenerateStep = computed(() => props.activeStep === 'generate')
const isSelectStep = computed(() => props.activeStep === 'select')
const isExportStep = computed(() => props.activeStep === 'export')
</script>

<template>
  <aside class="inspector">
    <section v-if="isSelectStep" class="panel inspector-section">
      <div class="section-header">
        <div>
          <h2>{{ t('inspector.summary') }}</h2>
          <p class="section-subtitle">{{ t('inspector.summaryMeta') }}</p>
        </div>
        <span class="ghost-pill">{{ selectedCandidate.id }}</span>
      </div>

      <div class="swatch-grid">
        <div class="swatch-card">
          <span class="swatch-card__label">{{ t('inspector.target') }}</span>
          <span class="swatch-card__tone" :style="{ backgroundColor: selectedCandidate.targetColorHex }" />
          <span class="mono">{{ selectedCandidate.targetColorHex }}</span>
        </div>
        <div class="swatch-card">
          <span class="swatch-card__label">{{ t('inspector.simulated') }}</span>
          <span
            class="swatch-card__tone"
            :style="{ backgroundColor: selectedCandidate.simulatedColorHex }"
          />
          <span class="mono">{{ t('inspector.metricDeltaE') }} {{ deltaEMetric }}</span>
        </div>
        <div class="swatch-card">
          <span class="swatch-card__label">{{ t('inspector.processPlus') }}</span>
          <span
            class="swatch-card__tone"
            :style="{ backgroundColor: selectedCandidate.processPlusColorHex }"
          />
          <span class="mono">{{ selectedCandidate.processPlusColorHex }}</span>
        </div>
        <div class="swatch-card">
          <span class="swatch-card__label">{{ t('inspector.processMinus') }}</span>
          <span
            class="swatch-card__tone"
            :style="{ backgroundColor: selectedCandidate.processMinusColorHex }"
          />
          <span class="mono">{{ selectedCandidate.processMinusColorHex }}</span>
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-card__header">
          <h3>{{ t('inspector.keyMetrics') }}</h3>
          <span class="ghost-pill ghost-pill--small">{{ labelCandidateStatus(selectedCandidate.status).text }}</span>
        </div>
        <div class="metric-summary-grid">
          <div v-for="metric in keyMetrics" :key="metric.label" class="metric-summary-card">
            <span class="metric-block__label">{{ labelMetric(metric.label) }}</span>
            <strong class="mono">{{ labelMetricValue(metric.label, metric.value) }}</strong>
          </div>
        </div>
        <p class="chart-note">{{ localizeCopy(selectedCandidate.rationale) }}</p>
      </div>

      <div class="chart-card">
        <div class="chart-card__header">
          <h3>{{ t('inspector.constraints') }}</h3>
          <span class="ghost-pill ghost-pill--small">{{ topConstraints.length }}</span>
        </div>
        <ul class="list-reset checklist checklist--compact">
          <li v-for="constraint in topConstraints" :key="constraint.id" class="checklist-item">
            <div class="checklist-item__header">
              <h3>{{ labelConstraint(constraint.label) }}</h3>
              <span class="status-pill status-pill--small" :data-state="labelConstraintState(constraint.state).state">
                {{ labelConstraintState(constraint.state).text }}
              </span>
            </div>
            <p>{{ localizeCopy(constraint.detail) }}</p>
          </li>
        </ul>
      </div>
    </section>

    <section
      v-if="(isBriefStep || isGenerateStep) && algorithmOverview"
      class="panel inspector-section"
    >
      <div class="chart-card">
        <div class="chart-card__header">
          <div>
            <h3>{{ t('inspector.algorithm') }}</h3>
            <p class="section-subtitle">{{ localizeCopy(algorithmOverview.algorithmName) }}</p>
          </div>
          <span
            v-if="algorithmOverview.bestExperiment"
            class="status-pill status-pill--small"
            :data-state="labelStage(algorithmOverview.bestExperiment.stage, algorithmOverview.bestExperiment.stageState).state"
          >
            {{ labelStage(algorithmOverview.bestExperiment.stage, algorithmOverview.bestExperiment.stageState).text }}
          </span>
        </div>
        <p>{{ localizeCopy(algorithmOverview.currentAssessment) }}</p>
        <div class="candidate-card__params">
          <span
            v-for="metric in algorithmOverview.headlineMetrics"
            :key="metric.label"
            class="ghost-pill"
          >
            {{ labelMetric(metric.label) }} {{ labelMetricValue(metric.label, metric.value) }}
          </span>
        </div>
        <p class="chart-note">{{ localizeCopy(algorithmOverview.trainingConclusion) }}</p>
      </div>
    </section>

    <section v-if="isSelectStep" class="panel inspector-section">
      <div class="section-header">
        <div>
          <h2>{{ t('inspector.candidates') }}</h2>
          <p class="section-subtitle">{{ t('inspector.candidatesMeta') }}</p>
        </div>
        <span class="ghost-pill">{{ candidates.length }} {{ t('common.results') }}</span>
      </div>

      <ul class="list-reset candidate-list">
        <li
          v-for="item in candidates"
          :key="item.id"
          class="candidate-card"
          :class="{ 'candidate-card--selected': selectedCandidateId === item.id }"
        >
          <div class="candidate-card__header">
            <div>
              <p class="candidate-card__eyebrow">
                {{ localizeCopy(item.group) }} · {{ t('common.rankLabel', { rank: item.rank }) }}
              </p>
              <h3>{{ item.id }}</h3>
            </div>
            <span
              class="status-pill status-pill--small"
              :data-state="labelCandidateStatus(item.status).state"
            >
              {{ labelCandidateStatus(item.status).text }}
            </span>
          </div>

          <div class="candidate-card__params">
            <span v-for="parameter in item.parameters" :key="parameter.label" class="ghost-pill">
              {{ labelParameter(parameter.label) }} {{ parameter.value }}
            </span>
          </div>

          <div class="candidate-card__metrics">
            <div v-for="metric in item.metrics" :key="metric.label" class="metric-block">
              <span class="metric-block__label">{{ labelMetric(metric.label) }}</span>
              <strong class="mono">{{ labelMetricValue(metric.label, metric.value) }}</strong>
            </div>
          </div>

          <p class="candidate-card__rationale">{{ localizeCopy(item.rationale) }}</p>

          <div class="button-row">
            <button
              class="button button--primary button--small"
              type="button"
              :disabled="selectedCandidateId === item.id"
              @click="selectedCandidateId = item.id"
            >
              {{ selectedCandidateId === item.id ? t('inspector.selected') : t('inspector.select') }}
            </button>
          </div>
        </li>
      </ul>
    </section>

    <section v-if="isSelectStep || isExportStep" class="panel inspector-section">
      <div class="section-header">
        <div>
          <h2>{{ t('inspector.constraints') }}</h2>
          <p class="section-subtitle">{{ t('inspector.constraintsMeta') }}</p>
        </div>
        <span class="ghost-pill">{{ selectedCandidate.id }}</span>
      </div>

      <ul class="list-reset checklist">
        <li v-for="constraint in constraints" :key="constraint.id" class="checklist-item">
          <div class="checklist-item__header">
            <h3>{{ labelConstraint(constraint.label) }}</h3>
            <span class="status-pill status-pill--small" :data-state="labelConstraintState(constraint.state).state">
              {{ labelConstraintState(constraint.state).text }}
            </span>
          </div>
          <p>{{ localizeCopy(constraint.detail) }}</p>
        </li>
      </ul>

      <div v-if="isExportStep" id="export-panel" class="export-estimate">
        <div class="section-header">
          <div>
            <h2>{{ t('inspector.export') }}</h2>
            <p class="section-subtitle">{{ t('inspector.exportMeta') }}</p>
          </div>
          <span class="status-pill status-pill--small" data-state="Exporting">
            {{ exportEstimate.progress }}%
          </span>
        </div>
        <dl class="estimate-grid">
          <div>
            <dt>{{ t('inspector.export.dimensions') }}</dt>
            <dd>{{ exportEstimate.dimensions }}</dd>
          </div>
          <div>
            <dt>{{ t('inspector.export.fileSize') }}</dt>
            <dd>{{ localizeCopy(exportEstimate.fileSize) }}</dd>
          </div>
          <div>
            <dt>{{ t('inspector.export.tilePlan') }}</dt>
            <dd>{{ localizeCopy(exportEstimate.tilePlan) }}</dd>
          </div>
          <div>
            <dt>{{ t('inspector.export.format') }}</dt>
            <dd>{{ localizeCopy(exportEstimate.format) }}</dd>
          </div>
        </dl>
        <div class="progress-bar" aria-hidden="true">
          <span :style="{ width: `${exportEstimate.progress}%` }" />
        </div>
        <p class="chart-note">{{ localizeCopy(exportEstimate.tileProgress) }}</p>
      </div>
    </section>
  </aside>
</template>
