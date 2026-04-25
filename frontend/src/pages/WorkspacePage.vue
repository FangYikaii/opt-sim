<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useI18n } from '../i18n'
import { workspaceService } from '../services/workspace-service'
import {
  resolveWorkflowStep,
  type WorkflowStepId,
} from '../workflow'
import type { ArtifactDetail, WorkspaceDetail } from '../api/contracts'

const props = defineProps<{
  runId: string
}>()

const workspace = ref<WorkspaceDetail | null>(null)
const activeArtifactId = ref<string | null>(null)
const activeArtifactDetail = ref<ArtifactDetail | null>(null)
const isArtifactLoading = ref(false)
const artifactDetails = ref<Record<string, ArtifactDetail>>({})
const isLoading = ref(true)
const errorMessage = ref('')
const {
  labelCandidateStatus,
  labelConstraintState,
  labelRunStatus,
  localizeCopy,
  locale,
  toggleLocale,
  t,
} = useI18n()

const currentStep = computed<WorkflowStepId>(() =>
  workspace.value ? resolveWorkflowStep(workspace.value.activeRun.status) : 'brief',
)

const selectedCandidate = computed(() =>
  workspace.value?.candidates.find((candidate) => candidate.selected) ?? workspace.value?.candidates[0] ?? null,
)

const alternativeCandidates = computed(() =>
  workspace.value?.candidates.filter((candidate) => candidate.id !== selectedCandidate.value?.id) ?? [],
)

const prioritizedConstraints = computed(() => workspace.value?.constraints.slice(0, 4) ?? [])
const visibleTimeline = computed(() => workspace.value?.timeline.slice(0, 4) ?? [])

async function loadArtifactDetail(artifactId: string): Promise<void> {
  if (artifactDetails.value[artifactId]) {
    activeArtifactDetail.value = artifactDetails.value[artifactId]
    return
  }

  isArtifactLoading.value = true
  try {
    const detail = await workspaceService.getArtifactDetail(artifactId)
    artifactDetails.value = {
      ...artifactDetails.value,
      [artifactId]: detail,
    }
    activeArtifactDetail.value = detail
  } finally {
    isArtifactLoading.value = false
  }
}

async function loadWorkspace(runId: string): Promise<void> {
  isLoading.value = true
  errorMessage.value = ''
  try {
    const workspaceDetail = await workspaceService.getWorkspaceDetail(runId)
    workspace.value = workspaceDetail
    const firstArtifactId = workspaceDetail.artifacts[0]?.id ?? null
    activeArtifactId.value = firstArtifactId
    activeArtifactDetail.value = null
    artifactDetails.value = {}
    if (firstArtifactId) {
      await loadArtifactDetail(firstArtifactId)
    }
  } catch (error) {
    workspace.value = null
    activeArtifactId.value = null
    activeArtifactDetail.value = null
    artifactDetails.value = {}
    errorMessage.value = error instanceof Error ? error.message : t.value('common.unavailable')
  } finally {
    isLoading.value = false
  }
}

async function selectArtifact(artifactId: string): Promise<void> {
  activeArtifactId.value = artifactId
  await loadArtifactDetail(artifactId)
}

onMounted(async () => {
  await loadWorkspace(props.runId)
})

watch(
  () => props.runId,
  async (runId) => {
    await loadWorkspace(runId)
  },
)
</script>

<template>
  <section v-if="workspace" class="route-page workspace-flow">
    <header class="panel workspace-flow__header">
      <div>
        <p class="eyebrow">{{ t('top.eyebrow') }}</p>
        <h1>{{ workspace.activeRun.title }}</h1>
        <p class="muted">方案单 ID: <span class="mono">{{ workspace.activeRun.id }}</span></p>
      </div>
      <div class="button-row">
        <span class="status-pill" :data-state="labelRunStatus(workspace.activeRun.status).state">
          {{ labelRunStatus(workspace.activeRun.status).text }}
        </span>
        <button class="button button--quiet" type="button" @click="toggleLocale">
          {{ t('app.switch') }} · {{ locale === 'zh' ? 'EN' : '中' }}
        </button>
        <RouterLink class="button button--quiet button--link" :to="{ name: 'home' }">
          {{ t('top.newRun') }}
        </RouterLink>
      </div>
    </header>

    <div class="workspace-flow__body">
      <main class="workspace-flow__main">
        <section class="panel flow-section">
          <div class="section-header">
            <div>
              <p class="eyebrow">Step 1</p>
              <h2>运行条件与模型上下文</h2>
              <p class="section-subtitle">先确认本次任务输入，以及当前接入的最佳训练模型和代理配置。</p>
            </div>
            <span class="ghost-pill">{{ currentStep }}</span>
          </div>

          <div class="overview-card-grid">
            <article class="chart-card">
              <span class="field__label">{{ workspace.draft.targetLabel }}</span>
              <h3 class="overview-card__value">{{ workspace.draft.targetValue }}</h3>
              <p class="chart-note">{{ localizeCopy(workspace.draft.requirementText) }}</p>
            </article>
            <article class="chart-card">
              <span class="field__label">当前模型</span>
              <h3 class="overview-card__value">
                {{ workspace.activeModel.checkpointFile ?? workspace.activeModel.label }}
              </h3>
              <p class="chart-note">{{ localizeCopy(workspace.activeModel.summary) }}</p>
            </article>
            <article class="chart-card">
              <span class="field__label">AI 决策代理</span>
              <h3 class="overview-card__value">
                {{ workspace.agentConfiguration.providerLabel }} · {{ workspace.agentConfiguration.model }}
              </h3>
              <p class="chart-note">{{ localizeCopy(workspace.agentConfiguration.summary) }}</p>
            </article>
          </div>

          <div class="flow-meta-grid">
            <div class="field">
              <span class="field__label">{{ workspace.draft.incidenceAngleLabel }}</span>
              <span class="field__value mono">{{ workspace.draft.incidenceAngleValue }}</span>
            </div>
            <div class="field">
              <span class="field__label">{{ workspace.draft.polarizationLabel }}</span>
              <span class="field__value mono">{{ workspace.draft.polarizationValue }}</span>
            </div>
            <div class="field">
              <span class="field__label">厚度窗口</span>
              <span class="field__value mono">{{ workspace.draft.heightWindow }}</span>
            </div>
            <div class="field">
              <span class="field__label">导出模式</span>
              <span class="field__value mono">{{ workspace.draft.exportMode }}</span>
            </div>
          </div>
        </section>

        <section class="panel flow-section">
          <div class="section-header">
            <div>
              <p class="eyebrow">Step 2</p>
              <h2>AI 决策建议</h2>
              <p class="section-subtitle">这一步只保留推荐结论、判断理由和需要你留意的风险。</p>
            </div>
            <span class="status-pill" :data-state="workspace.decisionSupport.confidence === 'high' ? 'pass' : workspace.decisionSupport.confidence === 'medium' ? 'warning' : 'fail'">
              {{ workspace.decisionSupport.confidence }}
            </span>
          </div>

          <article class="decision-hero">
            <div>
              <p class="field__label">Decision headline</p>
              <h3 class="decision-hero__title">{{ workspace.decisionSupport.headline }}</h3>
              <p class="chart-note">{{ localizeCopy(workspace.decisionSupport.summary) }}</p>
            </div>
            <div class="decision-hero__meta">
              <span class="ghost-pill">{{ workspace.decisionSupport.mode }}</span>
              <span class="ghost-pill">{{ workspace.decisionSupport.recommendedCandidateId ?? 'n/a' }}</span>
            </div>
          </article>

          <div class="decision-grid">
            <article class="chart-card">
              <div class="chart-card__header">
                <h3>推荐理由</h3>
                <span class="ghost-pill ghost-pill--small">{{ workspace.decisionSupport.rationale.length }}</span>
              </div>
              <ul class="list-reset checklist">
                <li v-for="item in workspace.decisionSupport.rationale" :key="item" class="checklist-item">
                  <p>{{ localizeCopy(item) }}</p>
                </li>
              </ul>
            </article>
            <article class="chart-card">
              <div class="chart-card__header">
                <h3>决策风险</h3>
                <span class="ghost-pill ghost-pill--small">{{ workspace.decisionSupport.risks.length }}</span>
              </div>
              <ul class="list-reset checklist">
                <li v-for="item in workspace.decisionSupport.risks" :key="item" class="checklist-item">
                  <p>{{ localizeCopy(item) }}</p>
                </li>
              </ul>
            </article>
          </div>

          <article class="chart-card">
            <span class="field__label">下一步</span>
            <p>{{ localizeCopy(workspace.decisionSupport.nextAction) }}</p>
          </article>
        </section>

        <section class="panel flow-section">
          <div class="section-header">
            <div>
              <p class="eyebrow">Step 3</p>
              <h2>候选评审</h2>
              <p class="section-subtitle">先看推荐方案，再决定是否需要切换到备选。</p>
            </div>
            <span class="ghost-pill">{{ workspace.candidates.length }} candidates</span>
          </div>

          <article v-if="selectedCandidate" class="candidate-focus">
            <div class="candidate-focus__top">
              <div>
                <p class="candidate-card__eyebrow">Recommended</p>
                <h3>{{ selectedCandidate.id }}</h3>
                <p class="chart-note">{{ localizeCopy(selectedCandidate.rationale) }}</p>
              </div>
              <span class="status-pill" :data-state="labelCandidateStatus(selectedCandidate.status).state">
                {{ labelCandidateStatus(selectedCandidate.status).text }}
              </span>
            </div>

            <div class="swatch-grid">
              <div class="swatch-card">
                <span class="swatch-card__label">Target</span>
                <span class="swatch-card__tone" :style="{ backgroundColor: selectedCandidate.targetColorHex }" />
                <span class="mono">{{ selectedCandidate.targetColorHex }}</span>
              </div>
              <div class="swatch-card">
                <span class="swatch-card__label">Simulated</span>
                <span class="swatch-card__tone" :style="{ backgroundColor: selectedCandidate.simulatedColorHex }" />
                <span class="mono">{{ selectedCandidate.simulatedColorHex }}</span>
              </div>
              <div class="swatch-card">
                <span class="swatch-card__label">+0.5 nm</span>
                <span class="swatch-card__tone" :style="{ backgroundColor: selectedCandidate.processPlusColorHex }" />
                <span class="mono">{{ selectedCandidate.processPlusColorHex }}</span>
              </div>
              <div class="swatch-card">
                <span class="swatch-card__label">-0.5 nm</span>
                <span class="swatch-card__tone" :style="{ backgroundColor: selectedCandidate.processMinusColorHex }" />
                <span class="mono">{{ selectedCandidate.processMinusColorHex }}</span>
              </div>
            </div>

            <div class="candidate-detail-grid">
              <div class="chart-card">
                <div class="chart-card__header">
                  <h3>结构参数</h3>
                </div>
                <div class="candidate-card__params">
                  <span v-for="parameter in selectedCandidate.parameters" :key="parameter.label" class="ghost-pill">
                    {{ parameter.label }} {{ parameter.value }}
                  </span>
                </div>
              </div>
              <div class="chart-card">
                <div class="chart-card__header">
                  <h3>关键指标</h3>
                </div>
                <div class="metric-summary-grid">
                  <div v-for="metric in selectedCandidate.metrics" :key="metric.label" class="metric-summary-card">
                    <span class="metric-block__label">{{ metric.label }}</span>
                    <strong class="mono">{{ metric.value }}</strong>
                  </div>
                </div>
              </div>
            </div>
          </article>

          <div v-if="alternativeCandidates.length" class="alternative-grid">
            <article v-for="candidate in alternativeCandidates" :key="candidate.id" class="chart-card">
              <div class="candidate-card__header">
                <div>
                  <p class="candidate-card__eyebrow">{{ localizeCopy(candidate.group) }}</p>
                  <h3>{{ candidate.id }}</h3>
                </div>
                <span class="status-pill status-pill--small" :data-state="labelCandidateStatus(candidate.status).state">
                  {{ labelCandidateStatus(candidate.status).text }}
                </span>
              </div>
              <div class="candidate-card__params">
                <span v-for="parameter in candidate.parameters" :key="parameter.label" class="ghost-pill">
                  {{ parameter.value }}
                </span>
              </div>
              <p class="chart-note">{{ localizeCopy(candidate.rationale) }}</p>
            </article>
          </div>
        </section>

        <section class="panel flow-section">
          <div class="section-header">
            <div>
              <p class="eyebrow">Step 4</p>
              <h2>导出准备与过程记录</h2>
              <p class="section-subtitle">最后确认导出规格，并查看本次 run 的关键过程记录。</p>
            </div>
          </div>

          <div class="decision-grid">
            <article class="chart-card">
              <div class="chart-card__header">
                <h3>导出规格</h3>
                <span class="status-pill status-pill--small" data-state="Exporting">
                  {{ workspace.exportEstimate.progress }}%
                </span>
              </div>
              <dl class="estimate-grid">
                <div>
                  <dt>Dimensions</dt>
                  <dd>{{ workspace.exportEstimate.dimensions }}</dd>
                </div>
                <div>
                  <dt>File size</dt>
                  <dd>{{ workspace.exportEstimate.fileSize }}</dd>
                </div>
                <div>
                  <dt>Tile plan</dt>
                  <dd>{{ workspace.exportEstimate.tilePlan }}</dd>
                </div>
                <div>
                  <dt>Format</dt>
                  <dd>{{ workspace.exportEstimate.format }}</dd>
                </div>
              </dl>
            </article>

            <article class="chart-card">
              <div class="chart-card__header">
                <h3>制造提醒</h3>
                <span class="ghost-pill ghost-pill--small">{{ prioritizedConstraints.length }}</span>
              </div>
              <ul class="list-reset checklist">
                <li v-for="constraint in prioritizedConstraints" :key="constraint.id" class="checklist-item">
                  <div class="checklist-item__header">
                    <h3>{{ constraint.label }}</h3>
                    <span class="status-pill status-pill--small" :data-state="labelConstraintState(constraint.state).state">
                      {{ labelConstraintState(constraint.state).text }}
                    </span>
                  </div>
                  <p>{{ localizeCopy(constraint.detail) }}</p>
                </li>
              </ul>
            </article>
          </div>

          <article class="chart-card">
            <div class="chart-card__header">
              <h3>过程记录</h3>
              <span class="ghost-pill ghost-pill--small">{{ visibleTimeline.length }}</span>
            </div>
            <ol class="list-reset timeline-list">
              <li v-for="event in visibleTimeline" :key="event.id" class="timeline-event" :data-type="event.type">
                <div class="timeline-event__marker" aria-hidden="true" />
                <div class="timeline-event__card">
                  <div class="timeline-event__topline">
                    <span class="ghost-pill">{{ event.label }}</span>
                    <span class="timeline-event__meta">{{ event.meta }}</span>
                  </div>
                  <h3>{{ localizeCopy(event.title) }}</h3>
                  <p class="timeline-event__body">{{ localizeCopy(event.body) }}</p>
                </div>
              </li>
            </ol>
          </article>
        </section>
      </main>

      <aside class="workspace-flow__side">
        <section class="panel flow-section">
          <div class="section-header">
            <div>
              <h2>历史记录</h2>
              <p class="section-subtitle">快速切换最近生成的方案单。</p>
            </div>
            <span class="ghost-pill">{{ workspace.runs.length }}</span>
          </div>
          <ul class="list-reset rail-list">
            <li
              v-for="run in workspace.runs"
              :key="run.id"
              class="rail-item"
              :class="{ 'rail-item--active': run.id === workspace.activeRun.id }"
            >
              <RouterLink class="rail-link" :to="{ name: 'run-workspace', params: { runId: run.id } }">
                <div class="rail-item__row">
                  <span class="rail-item__title">{{ run.title }}</span>
                </div>
                <div class="rail-item__row rail-item__meta">
                  <span class="status-pill status-pill--small" :data-state="labelRunStatus(run.status).state">
                    {{ labelRunStatus(run.status).text }}
                  </span>
                  <span>{{ run.updatedAt }}</span>
                </div>
              </RouterLink>
            </li>
          </ul>
        </section>

        <section class="panel flow-section">
          <div class="section-header">
            <div>
              <h2>Artifact</h2>
              <p class="section-subtitle">保留必要的导出与摘要文件。</p>
            </div>
            <span class="ghost-pill">{{ workspace.artifacts.length }}</span>
          </div>

          <div class="artifact-switcher">
            <button
              v-for="artifact in workspace.artifacts"
              :key="artifact.id"
              class="artifact-switcher__button"
              :class="{ 'artifact-switcher__button--active': artifact.id === activeArtifactId }"
              type="button"
              @click="selectArtifact(artifact.id)"
            >
              {{ artifact.name }}
            </button>
          </div>

          <div v-if="isArtifactLoading" class="chart-card">
            <p class="eyebrow">{{ t('common.loading') }}</p>
            <p class="chart-note">{{ t('artifact.loading') }}</p>
          </div>

          <template v-else-if="activeArtifactDetail">
            <article class="chart-card">
              <div class="chart-card__header">
                <h3>{{ activeArtifactDetail.name }}</h3>
                <span class="status-pill status-pill--small" :data-state="activeArtifactDetail.status">
                  {{ localizeCopy(activeArtifactDetail.status) }}
                </span>
              </div>
              <p>{{ localizeCopy(activeArtifactDetail.description) }}</p>
            </article>
            <article class="chart-card">
              <div class="chart-card__header">
                <h3>元数据</h3>
                <span class="ghost-pill ghost-pill--small">{{ activeArtifactDetail.metadata.length }}</span>
              </div>
              <dl class="artifact-detail-grid">
                <div v-for="item in activeArtifactDetail.metadata" :key="item.label" class="artifact-detail-card">
                  <dt>{{ localizeCopy(item.label) }}</dt>
                  <dd class="mono">{{ localizeCopy(item.value) }}</dd>
                </div>
              </dl>
            </article>
          </template>
        </section>
      </aside>
    </div>
  </section>

  <section v-else class="route-page loading-page panel">
    <p class="eyebrow">{{ t('common.loading') }}</p>
    <h1>{{ isLoading ? t('shell.workspaceLabel') : t('common.unavailable') }}</h1>
    <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
  </section>
</template>
