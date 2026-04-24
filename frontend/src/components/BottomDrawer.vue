<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from '../i18n'
import type { ExportEstimate, RunSummary } from '../api/contracts'

const props = defineProps<{
  activeRun: RunSummary
  exportEstimate: ExportEstimate
}>()

const activeTab = ref<'logs' | 'json' | 'commands'>('logs')
const { labelRunStatus, t } = useI18n()
const nextAction = computed(() => {
  const status = props.activeRun.status
  if (status === 'Needs approval') {
    return t.value('top.next.review')
  }
  if (status === 'Exporting') {
    return t.value('top.next.export')
  }
  if (status === 'Complete') {
    return t.value('top.next.complete')
  }
  if (status === 'Failed') {
    return t.value('top.next.failed')
  }
  return t.value('top.next.simulation')
})
</script>

<template>
  <section class="bottom-drawer panel" :aria-label="t('drawer.payload')">
    <div class="bottom-drawer__tabs">
      <button class="drawer-tab" :class="{ 'drawer-tab--active': activeTab === 'logs' }" type="button" @click="activeTab = 'logs'">{{ t('drawer.logs') }}</button>
      <button class="drawer-tab" :class="{ 'drawer-tab--active': activeTab === 'json' }" type="button" @click="activeTab = 'json'">{{ t('drawer.summary') }}</button>
      <button class="drawer-tab" :class="{ 'drawer-tab--active': activeTab === 'commands' }" type="button" @click="activeTab = 'commands'">{{ t('drawer.commands') }}</button>
    </div>
    <div class="bottom-drawer__content">
      <div v-if="activeTab === 'logs'" class="drawer-column">
        <p class="eyebrow">{{ t('drawer.latest') }}</p>
        <dl class="drawer-info-grid">
          <div>
            <dt>{{ t('drawer.currentStatus') }}</dt>
            <dd>{{ labelRunStatus(activeRun.status).text }}</dd>
          </div>
          <div>
            <dt>{{ t('drawer.exportProgress') }}</dt>
            <dd>{{ exportEstimate.progress }}%</dd>
          </div>
          <div>
            <dt>{{ t('drawer.tileProgress') }}</dt>
            <dd>{{ exportEstimate.tileProgress }}</dd>
          </div>
          <div>
            <dt>{{ t('drawer.nextAction') }}</dt>
            <dd>{{ nextAction }}</dd>
          </div>
        </dl>
      </div>
      <div v-if="activeTab === 'json'" class="drawer-column">
        <p class="eyebrow">{{ t('drawer.payload') }}</p>
        <pre class="log-block">{
  "run_id": "{{ activeRun.id }}",
  "status": "{{ labelRunStatus(activeRun.status).text }}",
  "format": "{{ exportEstimate.format }}",
  "dimensions": "{{ exportEstimate.dimensions }}"
}</pre>
      </div>
      <div v-if="activeTab === 'commands'" class="drawer-column">
        <p class="eyebrow">{{ t('drawer.notes') }}</p>
        <dl class="drawer-info-grid">
          <div>
            <dt>{{ t('drawer.deliveryFormat') }}</dt>
            <dd>{{ exportEstimate.format }}</dd>
          </div>
          <div>
            <dt>{{ t('drawer.deliverySize') }}</dt>
            <dd>{{ exportEstimate.fileSize }}</dd>
          </div>
          <div>
            <dt>{{ t('inspector.export.dimensions') }}</dt>
            <dd>{{ exportEstimate.dimensions }}</dd>
          </div>
          <div>
            <dt>{{ t('inspector.export.tilePlan') }}</dt>
            <dd>{{ exportEstimate.tilePlan }}</dd>
          </div>
        </dl>
      </div>
    </div>
  </section>
</template>
