<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import AppShell from '../components/AppShell.vue'
import { useI18n } from '../i18n'
import { workspaceService } from '../services/workspace-service'
import type { AlgorithmOverview, WorkspaceDetail } from '../api/contracts'

const props = defineProps<{
  runId: string
}>()

const workspace = ref<WorkspaceDetail | null>(null)
const algorithmOverview = ref<AlgorithmOverview | null>(null)
const isLoading = ref(true)
const { t } = useI18n()

async function loadWorkspace(runId: string): Promise<void> {
  isLoading.value = true
  const [workspaceDetail, overview] = await Promise.all([
    workspaceService.getWorkspaceDetail(runId),
    workspaceService.getAlgorithmOverview(),
  ])
  workspace.value = workspaceDetail
  algorithmOverview.value = overview
  isLoading.value = false
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
      :timeline="workspace.timeline"
      :candidates="workspace.candidates"
      :constraints="workspace.constraints"
      :export-estimate="workspace.exportEstimate"
    />
  </section>
  <section v-else class="route-page loading-page panel">
    <p class="eyebrow">{{ t('common.loading') }}</p>
    <h1>{{ isLoading ? t('shell.workspaceLabel') : t('common.unavailable') }}</h1>
  </section>
</template>
