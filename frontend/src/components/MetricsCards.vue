<!--
  MetricsCards — PSNR / SSIM / LPIPS summary cards with quality colour-coding.

  PSNR interpretation (dB):
    < 25  → poor (red)
    25-30 → acceptable (yellow)
    ≥ 30  → good (green)

  SSIM interpretation (0–1):
    < 0.6  → poor
    0.6-0.8 → acceptable
    ≥ 0.8  → good

  LPIPS (lower is better, 0–1):
    > 0.4  → poor
    0.2-0.4 → acceptable
    ≤ 0.2  → good
-->
<template>
  <div :class="['grid gap-4', gridCols]">
    <div
      v-for="card in cards"
      :key="card.key"
      class="card flex items-center gap-4"
    >
      <div :class="['w-2 self-stretch rounded-full', card.barClass]" />
      <div>
        <p class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
          {{ card.label }}
        </p>
        <p :class="['text-2xl font-bold tabular-nums mt-0.5', card.valueClass]">
          {{ card.formatted }}
        </p>
        <p class="text-[11px] text-gray-400 dark:text-gray-500 mt-0.5">{{ card.hint }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    psnr?:    number | null
    ssim?:    number | null
    lpips?:   number | null
    columns?: 1 | 2 | 3
  }>(),
  {
    columns: 3,
  },
)

const gridCols = computed(() => {
  if (props.columns === 1) return 'grid-cols-1'
  if (props.columns === 2) return 'grid-cols-2'
  return 'grid-cols-1 sm:grid-cols-3'
})

type Quality = 'good' | 'ok' | 'poor' | 'neutral'

function classify(key: 'psnr' | 'ssim' | 'lpips', val: number | null | undefined): Quality {
  if (val == null) return 'neutral'
  if (key === 'psnr') {
    if (val >= 30) return 'good'
    if (val >= 25) return 'ok'
    return 'poor'
  }
  if (key === 'ssim') {
    if (val >= 0.8) return 'good'
    if (val >= 0.6) return 'ok'
    return 'poor'
  }
  // lpips — lower is better
  if (val <= 0.2) return 'good'
  if (val <= 0.4) return 'ok'
  return 'poor'
}

function qualityBarClass(q: Quality): string {
  if (q === 'good')    return 'bg-green-500'
  if (q === 'ok')      return 'bg-yellow-400'
  if (q === 'poor')    return 'bg-red-500'
  return 'bg-gray-300 dark:bg-gray-600'
}

function qualityValueClass(q: Quality): string {
  if (q === 'good')    return 'text-green-600 dark:text-green-400'
  if (q === 'ok')      return 'text-yellow-600 dark:text-yellow-400'
  if (q === 'poor')    return 'text-red-600 dark:text-red-400'
  return 'text-gray-700 dark:text-gray-300'
}

interface CardDef { key: string; label: string; formatted: string; hint: string; barClass: string; valueClass: string }

const cards = computed<CardDef[]>(() => {
  const result: CardDef[] = []

  if (props.psnr !== undefined) {
    const q = classify('psnr', props.psnr)
    result.push({
      key: 'psnr',
      label: 'PSNR',
      formatted: props.psnr != null ? `${props.psnr.toFixed(2)} dB` : '—',
      hint: 'higher is better · ≥ 30 dB = good',
      barClass: qualityBarClass(q),
      valueClass: qualityValueClass(q),
    })
  }

  if (props.ssim !== undefined) {
    const q = classify('ssim', props.ssim)
    result.push({
      key: 'ssim',
      label: 'SSIM',
      formatted: props.ssim != null ? props.ssim.toFixed(4) : '—',
      hint: 'higher is better · ≥ 0.8 = good',
      barClass: qualityBarClass(q),
      valueClass: qualityValueClass(q),
    })
  }

  if (props.lpips !== undefined) {
    const q = classify('lpips', props.lpips)
    result.push({
      key: 'lpips',
      label: 'LPIPS',
      formatted: props.lpips != null ? props.lpips.toFixed(4) : '—',
      hint: 'lower is better · ≤ 0.2 = good',
      barClass: qualityBarClass(q),
      valueClass: qualityValueClass(q),
    })
  }

  return result
})
</script>
