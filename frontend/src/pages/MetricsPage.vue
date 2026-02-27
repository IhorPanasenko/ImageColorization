<template>
  <div class="p-6 space-y-6">
    <!-- ── Page header ─────────────────────────────────────────────────────── -->
    <PageHeader
      title="Batch Evaluation"
      description="Run quantitative metrics (PSNR / SSIM) against the test set for any model checkpoint."
    />

    <!-- ── Config card ────────────────────────────────────────────────────── -->
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

    <!-- ── Results ───────────────────────────────────────────────────────── -->
    <transition name="slide">
      <div v-if="result" class="space-y-5">
        <!-- Summary metrics cards -->
        <MetricsCards
          :psnr="result.avg_psnr"
          :ssim="result.avg_ssim"
          :extra="[{ label: 'Images evaluated', value: String(result.num_images), unit: '' }]"
          :columns="3"
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
          <!-- Table header with sort controls -->
          <div class="flex items-center justify-between flex-wrap gap-3">
            <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Per-Image Results</h2>
            <div class="flex items-center gap-2">
              <label class="text-xs text-gray-500 dark:text-gray-400">Sort by</label>
              <select v-model="sortKey" class="select !py-1 !text-xs w-32">
                <option value="filename">Filename</option>
                <option value="psnr">PSNR ↓</option>
                <option value="ssim">SSIM ↓</option>
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
                  <th class="pb-2 pr-4 font-medium cursor-pointer hover:text-brand-500" @click="cycleSortKey('psnr')">
                    PSNR (dB) <span v-if="sortKey==='psnr'">{{ sortDir==='asc'?'↑':'↓' }}</span>
                  </th>
                  <th class="pb-2 pr-4 font-medium cursor-pointer hover:text-brand-500" @click="cycleSortKey('ssim')">
                    SSIM <span v-if="sortKey==='ssim'">{{ sortDir==='asc'?'↑':'↓' }}</span>
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
                  <td class="py-2 pr-4 font-mono text-xs text-gray-600 dark:text-gray-300 max-w-[200px] truncate"
                      :title="row.filename">
                    {{ row.filename }}
                  </td>
                  <td class="py-2 pr-4 tabular-nums">
                    <span :class="psnrClass(row.psnr)">
                      {{ row.psnr?.toFixed(2) ?? '—' }}
                    </span>
                  </td>
                  <td class="py-2 pr-4 tabular-nums">
                    <span :class="ssimClass(row.ssim)">
                      {{ row.ssim?.toFixed(4) ?? '—' }}
                    </span>
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

    <!-- ── Empty / run prompt ───────────────────────────────────────────── -->
    <div
      v-if="!result && !loading"
      class="card flex flex-col items-center justify-center py-16 gap-3"
    >
      <BarChart2 class="w-12 h-12 text-gray-300 dark:text-gray-600" />
      <p class="text-sm text-gray-500 dark:text-gray-400">
        Select a model and checkpoint, then click <strong>Evaluate Test Set</strong>.
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useToast } from 'vue-toastification'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import type { TooltipItem } from 'chart.js'
import { Loader2, BarChart2, ArrowUpDown } from 'lucide-vue-next'

import PageHeader    from '@/components/PageHeader.vue'
import MetricsCards  from '@/components/MetricsCards.vue'
import ModelSelector from '@/components/ModelSelector.vue'
import StatusBadge   from '@/components/StatusBadge.vue'

import { metricsApi }   from '@/api/metrics'
import { modelsApi }    from '@/api/models'
import type { EvalResult, CheckpointInfo, ModelType, ImageMetrics } from '@/types'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

/** Detect dark mode from DOM */
function hasDarkClass(): boolean {
  return typeof document !== 'undefined'
    && (document.documentElement.classList.contains('dark')
        || document.querySelector('.dark') !== null)
}

// ── State ──────────────────────────────────────────────────────────────────────
const allCheckpoints = ref<CheckpointInfo[]>([])
const loadingMeta    = ref(true)
const loading        = ref(false)
const result         = ref<EvalResult | null>(null)
const sortKey        = ref<'filename' | 'psnr' | 'ssim'>('psnr')
const sortDir        = ref<'asc' | 'desc'>('desc')
const form           = reactive({ model: 'unet' as ModelType, checkpoint: '' })

const toast = useToast()

// ── Derived ────────────────────────────────────────────────────────────────────
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
    } else {
      cmp = (a.ssim ?? -1) - (b.ssim ?? -1)
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

function cycleSortKey(key: typeof sortKey.value) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    sortDir.value = 'desc'
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
</script>

<style scoped>
.slide-enter-active, .slide-leave-active { transition: opacity 0.25s ease, transform 0.25s ease; }
.slide-enter-from, .slide-leave-to { opacity: 0; transform: translateY(8px); }
</style>
