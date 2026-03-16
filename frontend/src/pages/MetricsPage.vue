<template>
  <div class="p-6 space-y-6">
    <!-- ── Page header ─────────────────────────────────────────────────────── -->
    <PageHeader
      title="Batch Evaluation"
      description="Run quantitative metrics against the test set for a single model, or benchmark all models at once."
    />

    <!-- ── Tab switcher ───────────────────────────────────────────────────── -->
    <div class="flex gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-xl w-fit">
      <button
        :class="['px-4 py-1.5 rounded-lg text-sm font-medium transition-colors',
          activeTab === 'single'
            ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300']"
        @click="activeTab = 'single'"
      >
        Single Model
      </button>
      <button
        :class="['px-4 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5',
          activeTab === 'benchmark'
            ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300']"
        @click="activeTab = 'benchmark'"
      >
        <FlaskConical class="w-3.5 h-3.5" />
        All Models Benchmark
      </button>
    </div>

    <!-- ══ SINGLE MODEL TAB ═══════════════════════════════════════════════════ -->
    <template v-if="activeTab === 'single'">

      <!-- Config card -->
      <div class="card space-y-4">
        <ModelSelector
          :model="form.model"
          :checkpoint="form.checkpoint"
          :checkpoints="filteredCheckpoints"
          :loading="loadingMeta"
          @update:model="v => { form.model = v; form.checkpoint = '' }"
          @update:checkpoint="v => form.checkpoint = v"
        />
        <div class="flex items-center gap-3">
          <button
            class="btn btn-primary flex items-center gap-2"
            :disabled="!form.checkpoint || loading"
            @click="runEval"
          >
            <Loader2 v-if="loading" class="w-4 h-4 animate-spin" />
            <BarChart2 v-else class="w-4 h-4" />
            {{ loading ? 'Evaluating…' : 'Evaluate Test Set' }}
          </button>
          <StatusBadge v-if="loading" status="running" label="Running evaluation…" />
          <span v-if="loading" class="text-xs text-gray-400 dark:text-gray-500">
            This may take 30-120 seconds depending on test-set size.
          </span>
        </div>
      </div>

      <!-- Results -->
      <transition name="slide">
        <div v-if="result" class="space-y-5">
          <!-- Summary metrics cards -->
          <MetricsCards
            :psnr="result.avg_psnr"
            :ssim="result.avg_ssim"
            :lpips="result.avg_lpips"
            :inference-time-ms="result.avg_inference_time_ms"
            :columns="4"
          />

          <!-- Bar chart of per-image PSNR -->
          <div v-if="chartData" class="card space-y-3">
            <div class="flex items-center justify-between">
              <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Per-Image PSNR (dB)</h2>
              <span class="text-xs text-gray-400 dark:text-gray-500">{{ result.num_images }} images</span>
            </div>
            <div style="max-height: 260px;">
              <Bar :data="chartData" :options="chartOptions" />
            </div>
          </div>

          <!-- Per-image table -->
          <div class="card space-y-3">
            <div class="flex items-center justify-between flex-wrap gap-3">
              <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Per-Image Results</h2>
              <div class="flex items-center gap-2">
                <label class="text-xs text-gray-500 dark:text-gray-400">Sort by</label>
                <select v-model="sortKey" class="select !py-1 !text-xs w-36">
                  <option value="filename">Filename</option>
                  <option value="psnr">PSNR</option>
                  <option value="ssim">SSIM</option>
                  <option value="lpips">LPIPS</option>
                  <option value="time">Inference Time</option>
                </select>
                <button
                  class="btn btn-secondary !py-1 !px-2 text-xs flex items-center gap-1"
                  @click="sortDir = sortDir === 'asc' ? 'desc' : 'asc'"
                >
                  <ArrowUpDown class="w-3 h-3" />
                  {{ sortDir === 'asc' ? 'Asc' : 'Desc' }}
                </button>
              </div>
            </div>

            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="text-left text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide border-b border-gray-100 dark:border-gray-700">
                    <th class="pb-2 pr-4 font-medium">File</th>
                    <th class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500" @click="cycleSortKey('psnr')">
                      PSNR (dB) <span v-if="sortKey==='psnr'">{{ sortDir==='asc'?'↑':'↓' }}</span>
                    </th>
                    <th class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500" @click="cycleSortKey('ssim')">
                      SSIM <span v-if="sortKey==='ssim'">{{ sortDir==='asc'?'↑':'↓' }}</span>
                    </th>
                    <th class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500" @click="cycleSortKey('lpips')">
                      LPIPS <span v-if="sortKey==='lpips'">{{ sortDir==='asc'?'↑':'↓' }}</span>
                    </th>
                    <th class="pb-2 pr-3 font-medium cursor-pointer hover:text-brand-500" @click="cycleSortKey('time')">
                      Time (ms) <span v-if="sortKey==='time'">{{ sortDir==='asc'?'↑':'↓' }}</span>
                    </th>
                    <th class="pb-2 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-gray-50 dark:divide-gray-800">
                  <tr
                    v-for="row in sortedRows"
                    :key="row.filename"
                    :class="row.error ? 'bg-red-50/40 dark:bg-red-900/10' : ''"
                    class="transition-colors"
                  >
                    <td class="py-2 pr-3 font-mono text-xs text-gray-600 dark:text-gray-300 max-w-[160px] truncate"
                        :title="row.filename">
                      {{ row.filename }}
                    </td>
                    <td class="py-2 pr-3 tabular-nums">
                      <span :class="psnrClass(row.psnr)">{{ row.psnr?.toFixed(2) ?? '—' }}</span>
                    </td>
                    <td class="py-2 pr-3 tabular-nums">
                      <span :class="ssimClass(row.ssim)">{{ row.ssim?.toFixed(4) ?? '—' }}</span>
                    </td>
                    <td class="py-2 pr-3 tabular-nums">
                      <span :class="lpipsClass(row.lpips)">{{ row.lpips?.toFixed(4) ?? '—' }}</span>
                    </td>
                    <td class="py-2 pr-3 tabular-nums text-gray-600 dark:text-gray-300">
                      {{ row.inference_time_ms != null ? row.inference_time_ms.toFixed(1) + ' ms' : '—' }}
                    </td>
                    <td class="py-2">
                      <StatusBadge
                        v-if="row.error"
                        status="failed"
                        :label="row.error.slice(0, 40)"
                      />
                      <StatusBadge v-else status="finished" label="OK" />
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </transition>

      <!-- Single model empty prompt -->
      <div
        v-if="!result && !loading"
        class="card flex flex-col items-center justify-center py-16 gap-3"
      >
        <BarChart2 class="w-12 h-12 text-gray-300 dark:text-gray-600" />
        <p class="text-sm text-gray-500 dark:text-gray-400">
          Select a model and checkpoint, then click <strong>Evaluate Test Set</strong>.
        </p>
      </div>

    </template><!-- end single model tab -->

    <!-- ══ BENCHMARK TAB ══════════════════════════════════════════════════════ -->
    <template v-else-if="activeTab === 'benchmark'">

      <!-- Config card -->
      <div class="card space-y-4">
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Model Configuration</h2>

        <!-- 2-column model slot grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div
            v-for="(slot, i) in benchmarkSlots"
            :key="i"
            :class="['rounded-xl border p-3 space-y-2 transition-colors',
              slot.enabled
                ? 'border-brand-200 dark:border-brand-800 bg-brand-50/30 dark:bg-brand-900/10'
                : 'border-gray-200 dark:border-gray-700 opacity-60']"
          >
            <label class="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" v-model="slot.enabled" class="rounded accent-brand-500" />
              <span
                class="w-2.5 h-2.5 rounded-full flex-shrink-0"
                :style="{ backgroundColor: MODEL_COLORS[slot.model] ?? '#6366f1' }"
              />
              <span class="text-sm font-medium text-gray-800 dark:text-gray-200">{{ slot.label }}</span>
              <span class="text-xs text-gray-400">({{ slot.model }})</span>
            </label>

            <!-- Checkpoint selector — only for neural models -->
            <select
              v-if="!isClassical(slot.model)"
              v-model="slot.checkpoint"
              class="select !py-1 !text-xs w-full"
              :disabled="!slot.enabled"
            >
              <option value="">— select checkpoint —</option>
              <option v-for="c in checkpointsForModel(slot.model)" :key="c.path" :value="c.path">
                {{ c.filename }}
              </option>
            </select>
            <p v-else class="text-xs text-gray-400 dark:text-gray-500 italic">
              Uses first test image as colour reference
            </p>
          </div>
        </div>

        <!-- Max images + image dir + run button row -->
        <div class="flex flex-col gap-3 pt-1">
          <div class="flex items-center gap-4 flex-wrap">
            <label class="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2">
              Max images:
              <input
                type="number"
                v-model.number="maxImages"
                min="1"
                max="200"
                class="input !py-0.5 !text-xs w-16"
                placeholder="all"
              />
            </label>
            <label class="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2 flex-1 min-w-0">
              Image folder (server path):
              <input
                type="text"
                v-model="benchImageDir"
                class="input !py-0.5 !text-xs flex-1 min-w-0 font-mono"
                placeholder="leave blank to use default test_samples folder"
              />
            </label>
          </div>
          <div class="flex items-center gap-3 flex-wrap">
            <button
              class="btn btn-primary flex items-center gap-2"
              :disabled="!canBenchmark || benchLoading"
              @click="runBenchmark"
            >
              <Loader2 v-if="benchLoading" class="w-4 h-4 animate-spin" />
              <FlaskConical v-else class="w-4 h-4" />
              {{ benchLoading ? 'Running benchmark…' : 'Run Benchmark' }}
            </button>
            <StatusBadge v-if="benchLoading" status="running" label="Running all models…" />
            <span v-if="benchLoading" class="text-xs text-gray-400 dark:text-gray-500">
              This may take several minutes on CPU.
            </span>
          </div>
        </div>
      </div>

      <!-- Benchmark results -->
      <transition name="slide">
        <div v-if="benchResult" class="space-y-5">

          <!-- 2×2 metric bar charts -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div class="card space-y-2">
              <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
                PSNR — higher is better
              </h3>
              <BenchmarkBarChart :models="barDataFor('avg_psnr')" metric-name="PSNR" unit="dB" />
            </div>
            <div class="card space-y-2">
              <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
                SSIM — higher is better
              </h3>
              <BenchmarkBarChart :models="barDataFor('avg_ssim')" metric-name="SSIM" unit="" />
            </div>
            <div class="card space-y-2">
              <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
                LPIPS — lower is better
              </h3>
              <BenchmarkBarChart :models="barDataFor('avg_lpips')" metric-name="LPIPS" unit="" :lower-is-better="true" />
            </div>
            <div class="card space-y-2">
              <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
                Inference Time — lower is better
              </h3>
              <BenchmarkBarChart
                :models="barDataFor('avg_inference_time_ms')"
                metric-name="Time"
                unit="ms"
                :lower-is-better="true"
              />
            </div>
          </div>

          <!-- Ranking table -->
          <div class="card space-y-3">
            <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Model Rankings</h2>
            <RankingTable :rows="rankingRows" />
          </div>

          <!-- Per-image drill-down (expandable) -->
          <details class="card">
            <summary class="cursor-pointer text-sm font-semibold text-gray-700 dark:text-gray-300 py-1 flex items-center gap-2 list-none select-none">
              <ChevronDown class="w-4 h-4 transition-transform detail-chevron" />
              Per-Image Details
            </summary>

            <div class="mt-4 space-y-4">
              <div class="flex items-center gap-3">
                <label class="text-xs text-gray-500 dark:text-gray-400">Image:</label>
                <select v-model="selectedImage" class="select !py-1 !text-xs">
                  <option v-for="fname in imageFilenames" :key="fname" :value="fname">{{ fname }}</option>
                </select>
              </div>

              <div v-if="selectedImage" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="space-y-1">
                  <h4 class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">PSNR (dB)</h4>
                  <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'psnr')" metric-name="PSNR" unit="dB" />
                </div>
                <div class="space-y-1">
                  <h4 class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">SSIM</h4>
                  <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'ssim')" metric-name="SSIM" unit="" />
                </div>
                <div class="space-y-1">
                  <h4 class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">LPIPS</h4>
                  <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'lpips')" metric-name="LPIPS" unit="" :lower-is-better="true" />
                </div>
                <div class="space-y-1">
                  <h4 class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Inference Time</h4>
                  <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'inference_time_ms')" metric-name="Time" unit="ms" :lower-is-better="true" />
                </div>
              </div>
            </div>
          </details>

        </div>
      </transition>

      <!-- Benchmark empty prompt -->
      <div
        v-if="!benchResult && !benchLoading"
        class="card flex flex-col items-center justify-center py-16 gap-3"
      >
        <FlaskConical class="w-12 h-12 text-gray-300 dark:text-gray-600" />
        <p class="text-sm text-gray-500 dark:text-gray-400 text-center">
          Configure the models above, then click <strong>Run Benchmark</strong>.<br />
          <span class="text-xs text-gray-400 dark:text-gray-500">
            All enabled models will be evaluated on the server-side test set.
          </span>
        </p>
      </div>

    </template><!-- end benchmark tab -->

  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useToast } from 'vue-toastification'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import type { TooltipItem } from 'chart.js'
import { Loader2, BarChart2, ArrowUpDown, FlaskConical, ChevronDown } from 'lucide-vue-next'

import PageHeader        from '@/components/PageHeader.vue'
import MetricsCards      from '@/components/MetricsCards.vue'
import ModelSelector     from '@/components/ModelSelector.vue'
import StatusBadge       from '@/components/StatusBadge.vue'
import BenchmarkBarChart from '@/components/BenchmarkBarChart.vue'
import RankingTable      from '@/components/RankingTable.vue'
import type { RankingRow } from '@/components/RankingTable.vue'

import { metricsApi } from '@/api/metrics'
import { modelsApi }  from '@/api/models'
import type {
  EvalResult, CheckpointInfo, ModelType, ImageMetrics,
  BenchmarkResult, BenchmarkModelResult,
} from '@/types'
import { CLASSICAL_MODEL_IDS } from '@/types'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

function hasDarkClass(): boolean {
  return typeof document !== 'undefined'
    && (document.documentElement.classList.contains('dark')
        || document.querySelector('.dark') !== null)
}

// ── Model colours ──────────────────────────────────────────────────────────────
const MODEL_COLORS: Record<string, string> = {
  baseline:        '#6366f1',
  unet:            '#22c55e',
  gan:             '#f59e0b',
  fusion:          '#3b82f6',
  classical_welsh: '#ec4899',
  classical_levin: '#14b8a6',
}

// ── Shared state ──────────────────────────────────────────────────────────────
const allCheckpoints = ref<CheckpointInfo[]>([])
const loadingMeta    = ref(true)
const toast          = useToast()
const activeTab      = ref<'single' | 'benchmark'>('single')

// ── Single-model state ────────────────────────────────────────────────────────
const loading  = ref(false)
const result   = ref<EvalResult | null>(null)
const sortKey  = ref<'filename' | 'psnr' | 'ssim' | 'lpips' | 'time'>('psnr')
const sortDir  = ref<'asc' | 'desc'>('desc')
const form     = reactive({ model: 'unet' as ModelType, checkpoint: '' })

// ── Benchmark state ───────────────────────────────────────────────────────────
interface BenchmarkSlot {
  model: ModelType
  checkpoint: string
  label: string
  enabled: boolean
}

const benchmarkSlots = reactive<BenchmarkSlot[]>([
  { model: 'baseline',        checkpoint: '', label: 'Baseline CNN', enabled: true },
  { model: 'unet',            checkpoint: '', label: 'U-Net',        enabled: true },
  { model: 'gan',             checkpoint: '', label: 'GAN',          enabled: true },
  { model: 'fusion',          checkpoint: '', label: 'Fusion',       enabled: true },
  { model: 'classical_welsh', checkpoint: '', label: 'Welsh 2002',   enabled: true },
  { model: 'classical_levin', checkpoint: '', label: 'Levin 2004',   enabled: true },
])

const maxImages     = ref<number | undefined>(undefined)
const benchImageDir = ref<string>('')
const benchLoading  = ref(false)
const benchResult   = ref<BenchmarkResult | null>(null)
const selectedImage = ref<string>('')

// ── Derived (single model) ────────────────────────────────────────────────────
const filteredCheckpoints = computed(() =>
  allCheckpoints.value.filter(c => !c.model_hint || c.model_hint === form.model),
)

const sortedRows = computed((): ImageMetrics[] => {
  if (!result.value) return []
  return [...result.value.per_image].sort((a, b) => {
    let cmp = 0
    if (sortKey.value === 'filename') {
      cmp = a.filename.localeCompare(b.filename)
    } else if (sortKey.value === 'psnr') {
      cmp = (a.psnr ?? -1) - (b.psnr ?? -1)
    } else if (sortKey.value === 'ssim') {
      cmp = (a.ssim ?? -1) - (b.ssim ?? -1)
    } else if (sortKey.value === 'lpips') {
      // lower is better — ascending default, so flip for descending sort
      cmp = (a.lpips ?? 999) - (b.lpips ?? 999)
    } else {
      cmp = (a.inference_time_ms ?? 999999) - (b.inference_time_ms ?? 999999)
    }
    return sortDir.value === 'asc' ? cmp : -cmp
  })
})

const chartData = computed(() => {
  if (!result.value?.per_image.length) return null
  const rows = result.value.per_image.filter(r => r.psnr !== null)
  if (!rows.length) return null
  return {
    labels: rows.map(r => r.filename.replace(/\.[^.]+$/, '')),
    datasets: [{
      label: 'PSNR (dB)',
      data:  rows.map(r => r.psnr),
      backgroundColor: 'rgba(79, 110, 247, 0.75)',
      borderRadius: 3,
      borderSkipped: false,
    }],
  }
})

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: { callbacks: { label: (ctx: TooltipItem<'bar'>) => ` ${((ctx.parsed as { y: number }).y ?? 0).toFixed(2)} dB` } },
  },
  scales: {
    x: {
      ticks: {
        maxRotation: 45,
        font: { size: 10 },
        color: hasDarkClass() ? '#d1d5db' : '#9ca3af',
      },
      grid: { display: false },
    },
    y: {
      ticks: { color: hasDarkClass() ? '#d1d5db' : '#9ca3af', font: { size: 11 } },
      grid: { color: hasDarkClass() ? 'rgba(156,163,175,0.12)' : 'rgba(156,163,175,0.1)' },
    },
  },
}))

// ── Derived (benchmark) ───────────────────────────────────────────────────────
function isClassical(model: ModelType): boolean {
  return CLASSICAL_MODEL_IDS.has(model)
}

function checkpointsForModel(model: ModelType): CheckpointInfo[] {
  return allCheckpoints.value.filter(c =>
    c.model_hint === model
    || (!c.model_hint && c.filename.toLowerCase().includes(model as string)),
  )
}

const canBenchmark = computed(() =>
  benchmarkSlots.some(s => {
    if (!s.enabled) return false
    if (isClassical(s.model)) return true
    return !!s.checkpoint
  }),
)

const rankingRows = computed((): RankingRow[] => {
  if (!benchResult.value) return []
  return benchResult.value.results.map(r => ({
    label:           r.label,
    model:           r.model as string,
    color:           MODEL_COLORS[r.model] ?? undefined,
    psnr:            r.avg_psnr,
    ssim:            r.avg_ssim,
    lpips:           r.avg_lpips,
    inferenceTimeMs: r.avg_inference_time_ms,
  }))
})

interface ModelBarItem { label: string; value: number | null; color: string }

function barDataFor(key: 'avg_psnr' | 'avg_ssim' | 'avg_lpips' | 'avg_inference_time_ms'): ModelBarItem[] {
  if (!benchResult.value) return []
  return benchResult.value.results.map(r => ({
    label: r.label,
    value: (r[key as keyof BenchmarkModelResult] as number | null) ?? null,
    color: MODEL_COLORS[r.model] ?? '#6366f1',
  }))
}

function perImageBarDataFor(
  filename: string,
  key: 'psnr' | 'ssim' | 'lpips' | 'inference_time_ms',
): ModelBarItem[] {
  if (!benchResult.value) return []
  return benchResult.value.results.map(r => {
    const img = r.per_image.find(p => p.filename === filename)
    const raw = img && !img.error ? img[key as keyof ImageMetrics] : null
    return {
      label: r.label,
      value: typeof raw === 'number' ? raw : null,
      color: MODEL_COLORS[r.model] ?? '#6366f1',
    }
  })
}

const imageFilenames = computed((): string[] => {
  if (!benchResult.value) return []
  const all = new Set<string>()
  benchResult.value.results.forEach(r =>
    r.per_image.forEach(img => { if (!img.error) all.add(img.filename) }),
  )
  return [...all].sort()
})

// Auto-fill the best checkpoint for each neural slot when checkpoints load
watch(allCheckpoints, (cks) => {
  for (const slot of benchmarkSlots) {
    if (isClassical(slot.model) || slot.checkpoint) continue
    const modelStr = slot.model as string
    const best =
      cks.find(c => c.filename.includes(modelStr) && c.filename.includes('best'))
      ?? cks.find(c => c.model_hint === slot.model)
    if (best) slot.checkpoint = best.path
  }
})

// Auto-select the first image when benchmark results arrive
watch(imageFilenames, (names) => {
  if (names.length && !selectedImage.value) selectedImage.value = names[0]
})

// ── Lifecycle ──────────────────────────────────────────────────────────────────
onMounted(async () => {
  try {
    allCheckpoints.value = await modelsApi.listCheckpoints()
  } finally {
    loadingMeta.value = false
  }
})

// ── Handlers ───────────────────────────────────────────────────────────────────
async function runEval() {
  if (!form.checkpoint) return
  loading.value = true
  result.value  = null

  try {
    result.value = await metricsApi.batchEvaluate(form.model, form.checkpoint)
    toast.success(`Evaluated ${result.value.num_images} images successfully.`)
  } catch (err: unknown) {
    toast.error(err instanceof Error ? err.message : 'Evaluation failed')
  } finally {
    loading.value = false
  }
}

async function runBenchmark() {
  const models = benchmarkSlots
    .filter(s => s.enabled)
    .map(({ model, checkpoint, label }) => ({ model, checkpoint, label }))
  if (!models.length) return

  benchLoading.value  = true
  benchResult.value   = null
  selectedImage.value = ''

  try {
    benchResult.value = await metricsApi.benchmark(
      models,
      maxImages.value,
      benchImageDir.value.trim() || undefined,
    )
    const count = benchResult.value.results.reduce((s, r) => s + r.num_images, 0)
    toast.success(
      `Benchmark complete — ${benchResult.value.results.length} models, ${count} image evaluations.`,
    )
  } catch (err: unknown) {
    toast.error(err instanceof Error ? err.message : 'Benchmark failed')
  } finally {
    benchLoading.value = false
  }
}

function cycleSortKey(key: typeof sortKey.value) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    // LPIPS and time: lower is better → default ascending; others → descending
    sortDir.value = (key === 'lpips' || key === 'time') ? 'asc' : 'desc'
  }
}

// ── Style helpers ──────────────────────────────────────────────────────────────
function psnrClass(v: number | null): string {
  if (v === null) return 'text-gray-400 dark:text-gray-500'
  if (v >= 30)   return 'font-semibold text-green-600 dark:text-green-400'
  if (v >= 20)   return 'text-amber-600 dark:text-amber-400'
  return 'text-red-500 dark:text-red-400'
}
function ssimClass(v: number | null): string {
  if (v === null) return 'text-gray-400 dark:text-gray-500'
  if (v >= 0.85) return 'font-semibold text-green-600 dark:text-green-400'
  if (v >= 0.6)  return 'text-amber-600 dark:text-amber-400'
  return 'text-red-500 dark:text-red-400'
}
function lpipsClass(v: number | null): string {
  if (v === null) return 'text-gray-400 dark:text-gray-500'
  if (v <= 0.2)  return 'font-semibold text-green-600 dark:text-green-400'
  if (v <= 0.4)  return 'text-amber-600 dark:text-amber-400'
  return 'text-red-500 dark:text-red-400'
}
</script>

<style scoped>
.slide-enter-active, .slide-leave-active { transition: opacity 0.25s ease, transform 0.25s ease; }
.slide-enter-from, .slide-leave-to { opacity: 0; transform: translateY(8px); }
details[open] summary .detail-chevron { transform: rotate(180deg); }
</style>

