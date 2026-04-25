<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { useI18n } from '../i18n'
import type { ArtifactSummary, RunSummary, TargetAsset } from '../api/contracts'

defineProps<{
  runs: RunSummary[]
  targets: TargetAsset[]
  artifacts: ArtifactSummary[]
  activeArtifactId: string | null
}>()

defineEmits<{
  selectArtifact: [artifactId: string]
}>()

const { labelAssetType, labelRunStatus, localizeCopy, t } = useI18n()
</script>

<template>
  <aside class="left-rail panel" :aria-label="t('left.contextLabel')">
    <section class="rail-section">
      <div class="section-header">
        <h2>{{ t('left.targets') }}</h2>
        <span class="section-meta">{{ targets.length }}</span>
      </div>
      <ul class="list-reset rail-list">
        <li v-for="target in targets" :key="target.id" class="rail-item">
          <div class="rail-item__title">
            <span
              v-if="target.swatchHex"
              class="swatch-chip"
              :style="{ backgroundColor: target.swatchHex }"
              aria-hidden="true"
            />
            <span>{{ localizeCopy(target.name) }}</span>
          </div>
          <p class="rail-item__detail">{{ labelAssetType(target.type) }} · {{ target.detail }}</p>
        </li>
      </ul>
    </section>

    <section class="rail-section">
      <div class="section-header">
        <h2>{{ t('left.runs') }}</h2>
        <span class="section-meta">{{ runs.length }}</span>
      </div>
      <ul class="list-reset rail-list">
        <li
          v-for="run in runs"
          :key="run.id"
          class="rail-item"
          :class="{ 'rail-item--active': labelRunStatus(run.status).state === 'Needs approval' }"
        >
          <RouterLink class="rail-link" :to="{ name: 'run-workspace', params: { runId: run.id } }">
            <div class="rail-item__row">
              <span class="rail-item__title">{{ localizeCopy(run.title) }}</span>
              <span v-if="run.warning" class="dot" aria-hidden="true" />
            </div>
          </RouterLink>
          <div class="rail-item__row rail-item__meta">
            <span class="status-pill status-pill--small" :data-state="labelRunStatus(run.status).state">
              {{ labelRunStatus(run.status).text }}
            </span>
            <span>{{ run.updatedAt }}</span>
          </div>
        </li>
      </ul>
    </section>

    <section class="rail-section">
      <div class="section-header">
        <h2>{{ t('left.artifacts') }}</h2>
        <span class="section-meta">{{ artifacts.length }}</span>
      </div>
      <ul class="list-reset rail-list">
        <li
          v-for="artifact in artifacts"
          :key="artifact.id"
          class="rail-item"
          :class="{ 'rail-item--active': artifact.id === activeArtifactId }"
        >
          <button class="rail-button" type="button" @click="$emit('selectArtifact', artifact.id)">
            <div class="rail-item__row">
              <span class="rail-item__title">{{ artifact.name }}</span>
            </div>
            <div class="rail-item__row rail-item__meta">
              <span class="ghost-pill ghost-pill--small">{{ artifact.type }}</span>
              <span>{{ artifact.status }}</span>
            </div>
          </button>
        </li>
      </ul>
    </section>
  </aside>
</template>
