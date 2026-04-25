<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import AppShell from '../components/AppShell.vue'
import { useI18n } from '../i18n'
import { workspaceService } from '../services/workspace-service'
import type { AlgorithmOverview, ArtifactDetail, WorkspaceDetail } from '../api/contracts'

const props = defineProps<{
  runId: string
}>()

const workspace = ref<WorkspaceDetail | null>(null)
const algorithmOverview = ref<AlgorithmOverview | null>(null)
const activeArtifactId = ref<string | null>(null)
const activeArtifactDetail = ref<ArtifactDetail | null>(null)
const isArtifactLoading = ref(false)
const artifactDetails = ref<Record<string, ArtifactDetail>>({})
const isLoading = ref(true)
const { t } = useI18n()

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
  const [workspaceDetail, overview] = await Promise.all([
    workspaceService.getWorkspaceDetail(runId),
    workspaceService.getAlgorithmOverview(),
  ])
  workspace.value = workspaceDetail
  algorithmOverview.value = overview
  const firstArtifactId = workspaceDetail.artifacts[0]?.id ?? null
  activeArtifactId.value = firstArtifactId
  activeArtifactDetail.value = null
  artifactDetails.value = {}
  if (firstArtifactId) {
    await loadArtifactDetail(firstArtifactId)
  }
  isLoading.value = false
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
  <section v-if="workspace" class="route-page workspace-page">
    <AppShell
      :algorithm-overview="algorithmOverview"
      :project="workspace.project"
      :draft="workspace.draft"
      :active-run="workspace.activeRun"
      :runs="workspace.runs"
      :targets="workspace.targets"
      :artifacts="workspace.artifacts"
      :active-artifact-id="activeArtifactId"
      :active-artifact-detail="activeArtifactDetail"
      :is-artifact-loading="isArtifactLoading"
      :timeline="workspace.timeline"
      :candidates="workspace.candidates"
      :constraints="workspace.constraints"
      :export-estimate="workspace.exportEstimate"
      @select-artifact="selectArtifact"
    />
  </section>
  <section v-else class="route-page loading-page panel">
    <p class="eyebrow">{{ t('common.loading') }}</p>
    <h1>{{ isLoading ? t('shell.workspaceLabel') : t('common.unavailable') }}</h1>
  </section>
</template>
