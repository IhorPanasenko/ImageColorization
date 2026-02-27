<template>
  <span :class="['badge', badgeClass]">{{ label }}</span>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { RunStatus } from '@/types'

const props = defineProps<{
  status: RunStatus
}>()

const badgeClass = computed(() => {
  const map: Record<RunStatus, string> = {
    running:  'badge-blue',
    finished: 'badge-green',
    failed:   'badge-red',
    stopped:  'badge-yellow',
  }
  return map[props.status] ?? 'badge-gray'
})

const label = computed(() => {
  const map: Record<RunStatus, string> = {
    running:  'Running',
    finished: 'Finished',
    failed:   'Failed',
    stopped:  'Stopped',
  }
  return map[props.status] ?? props.status
})
</script>
