<template>
  <div>
    <Radar :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Radar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'
import type { TooltipItem } from 'chart.js'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

/** Detect dark mode from DOM */
const isDark = computed(() =>
  typeof document !== 'undefined'
    && (document.documentElement.classList.contains('dark')
        || document.querySelector('.dark') !== null),
)

// ── Types ──────────────────────────────────────────────────────────────────────
export interface RadarSlot {
  label:  string
  psnr:   number | null
  ssim:   number | null
  lpips?: number | null   // optional; lower is better; inverted for display
  color:  string
}

// ── Props ──────────────────────────────────────────────────────────────────────
const props = defineProps<{
  slots: RadarSlot[]
  /** Maximum PSNR used to normalise to 0-1 range (default 40 dB) */
  maxPsnr?: number
}>()

// ── Normalisation ──────────────────────────────────────────────────────────────
const MAX_PSNR = computed(() => props.maxPsnr ?? 40)

/** Convert slot to a 0-1 score for each radar axis */
function normalise(slot: RadarSlot): number[] {
  const psnrNorm = slot.psnr !== null
    ? Math.min(slot.psnr / MAX_PSNR.value, 1)
    : 0

  const ssimNorm = slot.ssim !== null
    ? Math.max(0, Math.min(slot.ssim, 1))
    : 0

  // LPIPS: lower is better → invert (+1-lpips, clamped 0-1)
  const lpipsNorm = slot.lpips !== null && slot.lpips !== undefined
    ? Math.max(0, Math.min(1 - slot.lpips, 1))
    : null

  return lpipsNorm !== null
    ? [psnrNorm, ssimNorm, lpipsNorm]
    : [psnrNorm, ssimNorm]
}

const hasLpips = computed(() =>
  props.slots.some(s => s.lpips !== null && s.lpips !== undefined),
)

const radarLabels = computed(() =>
  hasLpips.value
    ? ['PSNR', 'SSIM', 'Perceptual (1-LPIPS)']
    : ['PSNR', 'SSIM'],
)

// ── Chart data ─────────────────────────────────────────────────────────────────
const chartData = computed(() => ({
  labels: radarLabels.value,
  datasets: props.slots.map(s => ({
    label:            s.label,
    data:             normalise(s),
    backgroundColor:  `${s.color}30`,
    borderColor:      s.color,
    borderWidth:      2,
    pointBackgroundColor: s.color,
    pointRadius:      4,
    pointHoverRadius: 6,
    fill:             true,
  })),
}))

const chartOptions = computed(() => ({
  responsive:          true,
  maintainAspectRatio: true,
  scales: {
    r: {
      beginAtZero: true,
      min:         0,
      max:         1,
      ticks: {
        stepSize: 0.25,
        font:     { size: 10 },
        color:    isDark.value ? '#d1d5db' : '#9ca3af',
        showLabelBackdrop: false,
      },
      pointLabels: {
        font:  { size: 12 },
        color: isDark.value ? '#d1d5db' : '#6b7280',
      },
      grid:         { color: isDark.value ? 'rgba(156,163,175,0.15)' : 'rgba(156,163,175,0.2)' },
      angleLines:   { color: isDark.value ? 'rgba(156,163,175,0.15)' : 'rgba(156,163,175,0.2)' },
    },
  },
  plugins: {
    legend: {
      position: 'bottom' as const,
      labels:   { font: { size: 12 }, padding: 16, usePointStyle: true, color: isDark.value ? '#d1d5db' : '#374151' },
    },
    tooltip: {
      callbacks: {
        label: (ctx: TooltipItem<'radar'>) =>
          ` ${ctx.dataset.label ?? ''}: ${((ctx.parsed as { r: number }).r * 100).toFixed(0)}%`,
      },
    },
  },
}))
</script>
