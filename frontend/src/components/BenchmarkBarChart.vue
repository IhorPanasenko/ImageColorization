<template>
  <div style="height: 180px; position: relative;">
    <Bar :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
} from 'chart.js'
import type { TooltipItem } from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip)

export interface ModelBar {
  label: string
  value: number | null
  color: string
}

const props = defineProps<{
  models: ModelBar[]
  metricName: string
  unit: string
  lowerIsBetter?: boolean
}>()

function hasDarkClass(): boolean {
  return typeof document !== 'undefined'
    && (document.documentElement.classList.contains('dark')
        || document.querySelector('.dark') !== null)
}

/** Returns a background colour for each bar, highlighting best (green) and worst (red). */
const barColors = computed(() => {
  const vals = props.models.map(m => m.value)
  const numericVals = vals.filter((v): v is number => v !== null)
  if (!numericVals.length) return props.models.map(m => `${m.color}cc`)

  const min = Math.min(...numericVals)
  const max = Math.max(...numericVals)

  return vals.map((v, i) => {
    if (v === null) return `${props.models[i].color}55`
    // When all values are equal, just use the model's colour
    if (min === max) return `${props.models[i].color}cc`
    if (props.lowerIsBetter) {
      if (v === min) return 'rgba(34, 197, 94, 0.82)'   // green — best
      if (v === max) return 'rgba(239, 68, 68, 0.78)'   // red   — worst
    } else {
      if (v === max) return 'rgba(34, 197, 94, 0.82)'   // green — best
      if (v === min) return 'rgba(239, 68, 68, 0.78)'   // red   — worst
    }
    return `${props.models[i].color}cc`
  })
})

const chartData = computed(() => ({
  labels: props.models.map(m => m.label),
  datasets: [
    {
      label: props.metricName,
      data: props.models.map(m => m.value),
      backgroundColor: barColors.value,
      borderRadius: 4,
      borderSkipped: false,
    },
  ],
}))

const chartOptions = computed(() => {
  const dark = hasDarkClass()
  const tickColor = dark ? '#d1d5db' : '#9ca3af'
  const gridColor = dark ? 'rgba(156,163,175,0.12)' : 'rgba(156,163,175,0.1)'

  // Determine decimal places for tooltip
  const decimals = props.unit === 'dB' ? 2 : props.unit === 'ms' ? 1 : 4

  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: TooltipItem<'bar'>) => {
            const v = (ctx.parsed as { y: number }).y
            return v !== null && v !== undefined
              ? ` ${Number(v).toFixed(decimals)}${props.unit ? ' ' + props.unit : ''}`
              : ' —'
          },
        },
      },
    },
    scales: {
      x: {
        ticks: {
          font: { size: 10 },
          color: tickColor,
          maxRotation: 30,
        },
        grid: { display: false },
      },
      y: {
        ticks: {
          font: { size: 10 },
          color: tickColor,
        },
        grid: { color: gridColor },
        title: props.unit
          ? {
              display: true,
              text: props.unit,
              color: dark ? '#9ca3af' : '#6b7280',
              font: { size: 10 },
            }
          : { display: false },
      },
    },
  }
})
</script>
