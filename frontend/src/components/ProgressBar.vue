<template>
  <div>
    <div v-if="label || showPct" class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1.5">
      <span v-if="label">{{ label }}</span>
      <span v-if="showPct" class="tabular-nums">{{ clampedPct }}%</span>
    </div>
    <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full" :style="{ height: `${height}px` }">
      <div
        :class="['rounded-full transition-all duration-500 ease-out', colorClass]"
        :style="{ width: `${clampedPct}%`, height: `${height}px` }"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    pct: number
    label?: string
    showPct?: boolean
    height?: number
    color?: 'brand' | 'green' | 'red' | 'yellow'
  }>(),
  {
    showPct: true,
    height: 8,
    color: 'brand',
  },
)

const clampedPct = computed(() => Math.min(100, Math.max(0, Math.round(props.pct))))

const colorClass = computed(() => {
  const map = {
    brand:  'bg-brand-500',
    green:  'bg-green-500',
    red:    'bg-red-500',
    yellow: 'bg-yellow-400',
  }
  return map[props.color ?? 'brand']
})
</script>
