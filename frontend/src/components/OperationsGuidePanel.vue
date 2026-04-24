<script setup lang="ts">
import type { AlgorithmOperationStep } from '../api/contracts'
import { useI18n } from '../i18n'

defineProps<{
  steps: AlgorithmOperationStep[]
  title?: string
}>()

const { localizeCopy, t } = useI18n()
</script>

<template>
  <section class="operations-guide panel">
    <div class="section-header">
      <div>
        <p class="eyebrow">{{ t('ops.eyebrow') }}</p>
        <h2>{{ title ?? t('ops.defaultTitle') }}</h2>
      </div>
      <span class="ghost-pill">{{ steps.length }} {{ t('common.steps') }}</span>
    </div>

    <ol class="list-reset operations-list">
      <li v-for="(step, index) in steps" :key="step.id" class="operation-card">
        <div class="operation-card__header">
          <span class="ghost-pill ghost-pill--small">{{ index + 1 }}</span>
          <h3>{{ localizeCopy(step.title) }}</h3>
        </div>
        <p>{{ localizeCopy(step.description) }}</p>
        <pre v-if="step.command" class="log-block">{{ step.command }}</pre>
        <p v-if="step.expectedResult" class="chart-note">{{ t('ops.expected') }}: {{ localizeCopy(step.expectedResult) }}</p>
      </li>
    </ol>
  </section>
</template>
