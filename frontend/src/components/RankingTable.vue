<template>
  <div class="overflow-x-auto">
    <table class="w-full text-sm">
      <thead>
        <tr class="text-left text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide border-b border-gray-100 dark:border-gray-700">
          <th class="pb-2 pr-3 font-medium w-12">Rank</th>
          <th class="pb-2 pr-3 font-medium">Model</th>
          <th
            class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500 select-none"
            @click="setSort('psnr')"
          >
            PSNR (dB)
            <span v-if="sortCol === 'psnr'">{{ sortDir === 'asc' ? '↑' : '↓' }}</span>
            <span v-else class="text-gray-300 dark:text-gray-600">↕</span>
          </th>
          <th
            class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500 select-none"
            @click="setSort('ssim')"
          >
            SSIM
            <span v-if="sortCol === 'ssim'">{{ sortDir === 'asc' ? '↑' : '↓' }}</span>
            <span v-else class="text-gray-300 dark:text-gray-600">↕</span>
          </th>
          <th
            class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500 select-none"
            @click="setSort('lpips')"
          >
            LPIPS
            <span v-if="sortCol === 'lpips'">{{ sortDir === 'asc' ? '↑' : '↓' }}</span>
            <span v-else class="text-gray-300 dark:text-gray-600">↕</span>
          </th>
          <th
            class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500 select-none"
            @click="setSort('time')"
          >
            Time (ms)
            <span v-if="sortCol === 'time'">{{ sortDir === 'asc' ? '↑' : '↓' }}</span>
            <span v-else class="text-gray-300 dark:text-gray-600">↕</span>
          </th>
          <th
            class="pb-2 font-medium cursor-pointer hover:text-brand-500 select-none"
            @click="setSort('score')"
          >
            Score
            <span v-if="sortCol === 'score'">{{ sortDir === 'asc' ? '↑' : '↓' }}</span>
            <span v-else class="text-gray-300 dark:text-gray-600">↕</span>
          </th>
        </tr>
      </thead>
      <tbody class="divide-y divide-gray-50 dark:divide-gray-800">
        <tr
          v-for="entry in displayRows"
          :key="entry.row.model + entry.row.label"
          class="transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/40"
        >
          <!-- Rank badge -->
          <td class="py-2.5 pr-3">
            <span v-if="entry.rank === 1" class="flex items-center gap-1 text-amber-400">
              <Trophy class="w-4 h-4" />
              <span class="font-bold text-xs">1</span>
            </span>
            <span v-else-if="entry.rank === 2" class="flex items-center gap-1 text-slate-400">
              <Award class="w-4 h-4" />
              <span class="font-bold text-xs">2</span>
            </span>
            <span v-else-if="entry.rank === 3" class="flex items-center gap-1 text-amber-700 dark:text-amber-600">
              <Award class="w-4 h-4" />
              <span class="font-bold text-xs">3</span>
            </span>
            <span v-else class="text-gray-500 dark:text-gray-400 text-xs font-medium pl-1">
              {{ entry.rank }}
            </span>
          </td>

          <!-- Model label + colour dot -->
          <td class="py-2.5 pr-3">
            <div class="flex items-center gap-2">
              <span
                v-if="entry.row.color"
                class="w-2.5 h-2.5 rounded-full flex-shrink-0"
                :style="{ backgroundColor: entry.row.color }"
              />
              <span class="font-medium text-gray-800 dark:text-gray-200">{{ entry.row.label }}</span>
            </div>
          </td>

          <!-- PSNR -->
          <td class="py-2.5 pr-3 tabular-nums">
            <span :class="psnrClass(entry.row.psnr)">
              {{ entry.row.psnr !== null ? entry.row.psnr.toFixed(2) : '—' }}
            </span>
          </td>

          <!-- SSIM -->
          <td class="py-2.5 pr-3 tabular-nums">
            <span :class="ssimClass(entry.row.ssim)">
              {{ entry.row.ssim !== null ? entry.row.ssim.toFixed(4) : '—' }}
            </span>
          </td>

          <!-- LPIPS -->
          <td class="py-2.5 pr-3 tabular-nums">
            <span :class="lpipsClass(entry.row.lpips)">
              {{ entry.row.lpips !== null ? entry.row.lpips.toFixed(4) : '—' }}
            </span>
          </td>

          <!-- Inference time -->
          <td class="py-2.5 pr-3 tabular-nums text-gray-600 dark:text-gray-400">
            {{ entry.row.inferenceTimeMs !== null ? `${entry.row.inferenceTimeMs.toFixed(1)}` : '—' }}
          </td>

          <!-- Composite score -->
          <td class="py-2.5 tabular-nums">
            <span class="font-semibold text-gray-700 dark:text-gray-300">
              {{ entry.score.toFixed(3) }}
            </span>
          </td>
        </tr>
      </tbody>
    </table>

    <!-- Legend note -->
    <p class="mt-3 text-[11px] text-gray-400 dark:text-gray-500">
      Score = 0.20 × PSNR<sub>norm</sub> + 0.35 × SSIM + 0.35 × (1 − LPIPS) + 0.10 × Speed<sub>norm</sub>
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { Trophy, Award } from 'lucide-vue-next'

export interface RankingRow {
  label: string
  model: string
  color?: string
  psnr: number | null
  ssim: number | null
  lpips: number | null
  inferenceTimeMs: number | null
}

const props = defineProps<{
  rows: RankingRow[]
}>()

type SortCol = 'psnr' | 'ssim' | 'lpips' | 'time' | 'score'

const sortCol = ref<SortCol>('score')
const sortDir = ref<'asc' | 'desc'>('desc')

function setSort(col: SortCol) {
  if (sortCol.value === col) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortCol.value = col
    // Lower-is-better metrics default to ascending; higher-is-better + score to descending
    sortDir.value = col === 'lpips' || col === 'time' ? 'asc' : 'desc'
  }
}

function computeScore(row: RankingRow, maxPsnr: number, maxTime: number): number {
  const psnrNorm  = row.psnr  !== null ? Math.min(row.psnr / maxPsnr, 1) : 0
  const ssimNorm  = row.ssim  !== null ? Math.max(0, Math.min(row.ssim, 1)) : 0
  const lpipsNorm = row.lpips !== null ? Math.max(0, 1 - row.lpips) : 0
  const speedNorm = row.inferenceTimeMs !== null && maxTime > 0
    ? Math.max(0, 1 - row.inferenceTimeMs / maxTime)
    : 0
  return 0.20 * psnrNorm + 0.35 * ssimNorm + 0.35 * lpipsNorm + 0.1 * speedNorm
}

/** Entries sorted by descending score (used for rank badges). */
const rankedEntries = computed(() => {
  const maxPsnr = Math.max(...props.rows.map(r => r.psnr ?? 0), 1)
  const maxTime = Math.max(...props.rows.map(r => r.inferenceTimeMs ?? 0), 1)

  return props.rows
    .map(row => ({ row, score: computeScore(row, maxPsnr, maxTime) }))
    .sort((a, b) => b.score - a.score)
    .map((entry, i) => ({ ...entry, rank: i + 1 }))
})

/** Entries re-sorted according to the currently selected column. */
const displayRows = computed(() => {
  const entries = [...rankedEntries.value]

  entries.sort((a, b) => {
    let cmp = 0
    switch (sortCol.value) {
      case 'psnr':  cmp = (a.row.psnr  ?? -Infinity) - (b.row.psnr  ?? -Infinity); break
      case 'ssim':  cmp = (a.row.ssim  ?? -Infinity) - (b.row.ssim  ?? -Infinity); break
      case 'lpips': cmp = (a.row.lpips ?? Infinity)  - (b.row.lpips ?? Infinity);  break
      case 'time':  cmp = (a.row.inferenceTimeMs ?? Infinity) - (b.row.inferenceTimeMs ?? Infinity); break
      case 'score': cmp = a.score - b.score; break
    }
    return sortDir.value === 'asc' ? cmp : -cmp
  })

  return entries
})

// ── Quality colour helpers (mirrors MetricsCards thresholds) ──────────────────
function psnrClass(v: number | null): string {
  if (v === null) return 'text-gray-400 dark:text-gray-500'
  if (v >= 30) return 'font-semibold text-green-600 dark:text-green-400'
  if (v >= 25) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-500 dark:text-red-400'
}

function ssimClass(v: number | null): string {
  if (v === null) return 'text-gray-400 dark:text-gray-500'
  if (v >= 0.8) return 'font-semibold text-green-600 dark:text-green-400'
  if (v >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-500 dark:text-red-400'
}

function lpipsClass(v: number | null): string {
  if (v === null) return 'text-gray-400 dark:text-gray-500'
  if (v <= 0.2) return 'font-semibold text-green-600 dark:text-green-400'
  if (v <= 0.4) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-500 dark:text-red-400'
}
</script>
