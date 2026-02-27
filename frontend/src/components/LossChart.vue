<!--
  LossChart — line chart for training loss curves.

  For standard models:  renders a single "Loss" line.
  For GAN / Fusion:     renders dual lines — "Generator Loss" + "Discriminator Loss".

  Powered by vue-chartjs (Chart.js wrapper).
-->
<template>
  <div>
    <p v-if="data.length === 0" class="text-xs text-gray-400 dark:text-gray-500 py-4 text-center">
      No loss data yet — waiting for first epoch…
    </p>
    <Line v-else :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  type TooltipItem,
} from 'chart.js'
import type { LossPoint } from '@/composables/useTraining'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler,
)

const props = withDefaults(
  defineProps<{
    data:   LossPoint[]
    title?: string
    dark?:  boolean
  }>(),
  {
    title: 'Training Loss',
  },
)

/** Detect dark mode from DOM when prop is not explicitly passed */
const isDark = computed(() =>
  props.dark ?? ((typeof document !== 'undefined' && document.documentElement.classList.contains('dark'))
              || (typeof document !== 'undefined' && document.querySelector('.dark') !== null)),
)

const isGan = computed(() => props.data.some((d) => d.lossD !== undefined))

const labels = computed(() => props.data.map((d) => `Ep ${d.epoch}`))

// Colour palette
const BRAND    = '#4f6ef7'
const RED      = '#ef4444'
const GREEN    = '#22c55e'
const GRID     = '#e5e7eb'
const GRID_DARK = '#374151'
const TEXT      = '#6b7280'
const TEXT_DARK = '#9ca3af'

const chartData = computed(() => {
  if (isGan.value) {
    return {
      labels: labels.value,
      datasets: [
        {
          label: 'Generator Loss',
          data:  props.data.map((d) => d.lossG ?? d.loss),
          borderColor: BRAND,
          backgroundColor: BRAND + '20',
          borderWidth: 2,
          pointRadius: 2,
          fill: false,
          tension: 0.3,
        },
        {
          label: 'Discriminator Loss',
          data:  props.data.map((d) => d.lossD ?? null),
          borderColor: RED,
          backgroundColor: RED + '20',
          borderWidth: 2,
          pointRadius: 2,
          fill: false,
          tension: 0.3,
        },
      ],
    }
  }

  return {
    labels: labels.value,
    datasets: [
      {
        label: 'Loss',
        data:  props.data.map((d) => d.loss),
        borderColor: GREEN,
        backgroundColor: GREEN + '20',
        borderWidth: 2.5,
        pointRadius: 2,
        fill: true,
        tension: 0.35,
      },
    ],
  }
})

const gridColor = computed(() => isDark.value ? GRID_DARK : GRID)
const textColor = computed(() => isDark.value ? TEXT_DARK : TEXT)

const chartOptions = computed(() => ({
  responsive:          true,
  maintainAspectRatio: true,
  animation:           { duration: 200 },
  plugins: {
    legend: {
      display: isGan.value,
      labels: {
        color:    textColor.value,
        boxWidth: 12,
        font:     { size: 11 },
      },
    },
    title: {
      display:  !!props.title,
      text:     props.title,
      color:    textColor.value,
      font:     { size: 12 },
      padding:  { bottom: 8 },
    },
    tooltip: {
      callbacks: {
        label: (ctx: TooltipItem<'line'>) =>
          `${ctx.dataset.label ?? 'Loss'}: ${ctx.parsed.y?.toFixed(4) ?? '—'}`,
      },
    },
  },
  scales: {
    x: {
      ticks: { color: textColor.value, maxTicksLimit: 12, font: { size: 10 } },
      grid:  { color: gridColor.value },
    },
    y: {
      ticks: {
        color: textColor.value,
        font: { size: 10 },
        callback: (v: number | string) =>
            typeof v === 'number' ? v.toFixed(3) : v,
      },
      grid: { color: gridColor.value },
    },
  },
}))
</script>
