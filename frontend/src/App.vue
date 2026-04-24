<script setup lang="ts">
import { RouterView } from 'vue-router'
import { provideI18n } from './i18n'

provideI18n()
</script>

<template>
  <RouterView />
</template>
