<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from '../i18n'
import type { ArtifactDetail, ArtifactSummary } from '../api/contracts'

const props = defineProps<{
  artifacts: ArtifactSummary[]
  activeArtifactId: string | null
  activeArtifactDetail: ArtifactDetail | null
  isLoading: boolean
}>()

defineEmits<{
  selectArtifact: [artifactId: string]
}>()

const { localizeCopy, t } = useI18n()

const activeArtifact = computed(() =>
  props.artifacts.find((artifact) => artifact.id === props.activeArtifactId) ?? props.artifacts[0] ?? null,
)
</script>

<template>
  <section class="panel inspector-section">
    <div class="section-header">
      <div>
        <h2>{{ t('artifact.title') }}</h2>
        <p class="section-subtitle">{{ t('artifact.subtitle') }}</p>
      </div>
      <span v-if="activeArtifact" class="ghost-pill">{{ activeArtifact.name }}</span>
    </div>

    <div v-if="artifacts.length > 0" class="artifact-switcher">
      <button
        v-for="artifact in artifacts"
        :key="artifact.id"
        class="artifact-switcher__button"
        :class="{ 'artifact-switcher__button--active': artifact.id === activeArtifactId }"
        type="button"
        @click="$emit('selectArtifact', artifact.id)"
      >
        {{ artifact.name }}
      </button>
    </div>

    <div v-if="isLoading" class="chart-card">
      <p class="eyebrow">{{ t('common.loading') }}</p>
      <p class="chart-note">{{ t('artifact.loading') }}</p>
    </div>

    <template v-else-if="activeArtifactDetail">
      <div class="chart-card">
        <div class="chart-card__header">
          <h3>{{ t('artifact.description') }}</h3>
          <span class="status-pill status-pill--small" :data-state="activeArtifactDetail.status">
            {{ activeArtifactDetail.status }}
          </span>
        </div>
        <p>{{ localizeCopy(activeArtifactDetail.description) }}</p>
      </div>

      <div class="chart-card">
        <div class="chart-card__header">
          <h3>{{ t('artifact.metadata') }}</h3>
          <span class="ghost-pill ghost-pill--small">{{ activeArtifactDetail.metadata.length }}</span>
        </div>
        <dl class="artifact-detail-grid">
          <div v-for="item in activeArtifactDetail.metadata" :key="item.label" class="artifact-detail-card">
            <dt>{{ localizeCopy(item.label) }}</dt>
            <dd class="mono">{{ localizeCopy(item.value) }}</dd>
          </div>
        </dl>
      </div>
    </template>

    <div v-else class="chart-card">
      <p class="chart-note">{{ t('artifact.empty') }}</p>
    </div>
  </section>
</template>
