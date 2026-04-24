<script setup lang="ts">
import { useI18n } from '../i18n'

const { t } = useI18n()
</script>

<template>
  <section class="route-page panel panel--soft not-found-page">
    <p class="eyebrow">404</p>
    <h1>{{ t('notFound.title') }}</h1>
    <p class="muted">{{ t('notFound.body') }}</p>
  </section>
</template>
