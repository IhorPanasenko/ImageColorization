<template>
  <div class="p-6 space-y-6">
    <!-- Page header -->
    <PageHeader
      title="Training History"
      description="Browse every past training run. Inspect loss curves, hyperparameters, and full logs."
    />

    <!-- Toolbar -->
    <div class="flex flex-wrap items-center gap-2">
      <select v-model="filterModel" class="select !w-36 !py-1.5 !text-xs">
        <option value="">All models</option>
        <option v-for="m in MODEL_OPTIONS" :key="m.value" :value="m.value">{{ m.label }}</option>
      </select>
      <select v-model="filterStatus" class="select !w-36 !py-1.5 !text-xs">
        <option value="">All statuses</option>
        <option value="running">Running</option>
        <option value="finished">Finished</option>
        <option value="failed">Failed</option>
        <option value="stopped">Stopped</option>
      </select>
      <select v-model="sortKey" class="select !w-40 !py-1.5 !text-xs">
        <option value="started_at">Newest first</option>
        <option value="started_at_asc">Oldest first</option>
        <option value="loss">Best loss</option>
        <option value="model">Model A-Z</option>
      </select>
      <div class="flex-1" />
      <span class="text-xs text-gray-400 dark:text-gray-500 tabular-nums">
        {{ filteredRuns.length }} run{{ filteredRuns.length !== 1 ? 's' : '' }}
      </span>
      <button
        class="btn btn-secondary flex items-center gap-1.5 !py-1.5 !px-3 text-xs"
        :disabled="refreshing"
        @click="loadRuns"
      >
        <RefreshCw class="w-3.5 h-3.5" :class="{ 'animate-spin': refreshing }" />
        Refresh
      </button>
    </div>

    <!-- Loading spinner (initial) -->
    <div v-if="refreshing && !runs.length" class="card flex items-center justify-center py-16 gap-3">
      <Loader2 class="w-6 h-6 animate-spin text-brand-500" />
      <span class="text-sm text-gray-400 dark:text-gray-500">Loading runs...</span>
    </div>

    <!-- Empty state -->
    <div
      v-else-if="!filteredRuns.length"
      class="card flex flex-col items-center justify-center py-16 gap-3"
    >
      <History class="w-12 h-12 text-gray-300 dark:text-gray-600" />
      <p class="text-sm text-gray-500 dark:text-gray-400">
        No training runs found<span v-if="filterModel || filterStatus"> matching the current filter</span>.
        <RouterLink to="/training" class="ml-1 text-brand-500 hover:underline">Start one &rarr;</RouterLink>
      </p>
    </div>

    <!-- Two-column layout -->
    <div v-else class="grid grid-cols-1 xl:grid-cols-5 gap-6 items-start">

      <!-- Run list (left) -->
      <div class="xl:col-span-2 space-y-2">
        <TransitionGroup name="list" tag="div" class="space-y-2">
          <div
            v-for="run in filteredRuns"
            :key="run.run_id"
            class="card !p-4 cursor-pointer border-2 transition-all duration-150"
            :class="selected && selected.run_id === run.run_id
              ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-900/10 dark:border-brand-500'
              : 'border-transparent hover:border-gray-200 dark:hover:border-gray-600'"
            @click="selectRun(run)"
          >
            <div class="flex items-start justify-between gap-2">
              <div class="min-w-0 flex-1">
                <div class="flex items-center flex-wrap gap-2">
                  <span class="font-semibold text-sm text-gray-800 dark:text-gray-200">
                    {{ MODEL_LABELS[run.model] || run.model }}
                  </span>
                  <StatusBadge :status="run.status" />
                  <span class="font-mono text-[10px] text-gray-400 dark:text-gray-600">#{{ run.run_id.slice(0, 8) }}</span>
                </div>
                <div class="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-500 dark:text-gray-400">
                  <span>{{ run.epoch }}/{{ run.total_epochs }} epochs</span>
                  <span v-if="run.loss != null">Loss {{ run.loss.toFixed(4) }}</span>
                  <span v-if="run.params && run.params.lr">LR {{ run.params.lr }}</span>
                </div>
              </div>
              <div class="shrink-0 text-right space-y-0.5">
                <p class="text-[11px] text-gray-400 dark:text-gray-500">{{ formatDate(run.started_at) }}</p>
                <p v-if="run.finished_at" class="text-[11px] text-gray-400 dark:text-gray-500 tabular-nums">
                  {{ formatDuration(run.started_at, run.finished_at) }}
                </p>
              </div>
            </div>
            <div v-if="run.total_epochs > 0" class="mt-2.5">
              <ProgressBar
                :pct="(run.epoch / run.total_epochs) * 100"
                :height="3"
                :show-pct="false"
                :color="pbarColor(run.status)"
              />
            </div>
          </div>
        </TransitionGroup>
      </div>

      <!-- Detail panel (right) -->
      <div class="xl:col-span-3">
        <!-- Placeholder -->
        <div
          v-if="!selected"
          class="card flex flex-col items-center justify-center py-24 gap-3"
        >
          <MousePointerClick class="w-10 h-10 text-gray-300 dark:text-gray-600" />
          <p class="text-sm text-gray-400 dark:text-gray-500">Select a run to inspect its details.</p>
        </div>

        <!-- Run detail -->
        <div v-else class="space-y-4">

          <!-- Header: title + delete -->
          <div class="flex items-center justify-between gap-3">
            <div>
              <h2 class="text-base font-semibold text-gray-800 dark:text-gray-200">
                {{ MODEL_LABELS[selected.model] }}
                <span class="font-mono text-sm font-normal text-gray-400 dark:text-gray-500">#{{ selected.run_id.slice(0, 8) }}</span>
              </h2>
              <p class="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Started {{ formatDate(selected.started_at) }}</p>
            </div>
            <button
              class="btn btn-danger flex items-center gap-1.5 !py-1.5 !px-3 text-xs"
              @click="confirmDelete = true"
            >
              <Trash2 class="w-3.5 h-3.5" />
              Delete
            </button>
          </div>

          <!-- Stat cards -->
          <div class="card !p-0 overflow-hidden">
            <div class="grid grid-cols-2 sm:grid-cols-4 divide-x divide-gray-100 dark:divide-gray-700 divide-y sm:divide-y-0">
              <div class="p-4">
                <p class="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Status</p>
                <div class="mt-1.5"><StatusBadge :status="selected.status" /></div>
              </div>
              <div class="p-4">
                <p class="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Progress</p>
                <p class="mt-1 font-semibold text-sm text-gray-800 dark:text-gray-200 tabular-nums">
                  {{ selected.epoch }} / {{ selected.total_epochs }}
                  <span class="text-xs font-normal text-gray-400 dark:text-gray-500">epochs</span>
                </p>
              </div>
              <div class="p-4">
                <p class="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Final Loss</p>
                <p class="mt-1 font-semibold text-sm text-gray-800 dark:text-gray-200 tabular-nums">
                  {{ selected.loss != null ? selected.loss.toFixed(4) : '—' }}
                </p>
              </div>
              <div class="p-4">
                <p class="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Duration</p>
                <p class="mt-1 font-semibold text-sm text-gray-800 dark:text-gray-200 tabular-nums">
                  {{ selected.finished_at ? formatDuration(selected.started_at, selected.finished_at) : 'In progress' }}
                </p>
              </div>
            </div>
          </div>

          <!-- Hyperparameters -->
          <div v-if="paramEntries.length > 0" class="card space-y-3">
            <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Hyperparameters</h3>
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <div
                v-for="p in paramEntries"
                :key="p.key"
                class="rounded-lg bg-gray-50 dark:bg-gray-800 px-3 py-2"
              >
                <p class="text-[10px] uppercase tracking-wider text-gray-400 dark:text-gray-500">{{ p.key }}</p>
                <p class="text-sm font-mono text-gray-700 dark:text-gray-300 truncate" :title="String(p.val)">{{ p.val }}</p>
              </div>
            </div>
          </div>

          <!-- Loading run data -->
          <div v-if="loadingLogs" class="card flex items-center justify-center py-10 gap-3">
            <Loader2 class="w-5 h-5 animate-spin text-brand-500" />
            <span class="text-sm text-gray-400 dark:text-gray-500">Loading run data...</span>
          </div>

          <template v-else-if="logData">
            <!-- Loss curve -->
            <div class="card space-y-3">
              <div class="flex items-center justify-between">
                <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Loss Curve</h3>
                <span class="text-xs text-gray-400 dark:text-gray-500 tabular-nums">{{ lossPoints.length }} data points</span>
              </div>
              <LossChart :data="lossPoints" title="" />
              <p v-if="!lossPoints.length" class="text-xs text-gray-400 dark:text-gray-500 text-center py-2">
                No loss data in log yet.
              </p>
            </div>

            <!-- Learning Rate Schedule -->
            <div v-if="lrChartLabels.length > 1" class="card space-y-3">
              <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Learning Rate Schedule</h3>
              <div style="max-height: 176px; position: relative;">
                <Line :data="lrChartData" :options="lrChartOptions" />
              </div>
            </div>

            <!-- TensorBoard scalars (optional, per model_type) -->
            <div v-if="tbTags.length > 0" class="card space-y-3">
              <div class="flex items-center justify-between gap-2">
                <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
                  TensorBoard Scalars
                  <span class="text-xs font-normal text-gray-400 dark:text-gray-500 ml-1">(model-level)</span>
                </h3>
                <button class="text-xs text-brand-500 hover:underline" @click="tbExpanded = !tbExpanded">
                  {{ tbExpanded ? 'Collapse' : 'Expand' }}
                </button>
              </div>
              <div v-if="!tbExpanded" class="flex flex-wrap gap-2">
                <span v-for="tag in tbTags" :key="tag" class="badge badge-blue text-[10px]">{{ tag }}</span>
              </div>
              <div v-else class="space-y-4">
                <div v-for="tag in tbTags" :key="tag" class="space-y-1">
                  <p class="text-xs font-mono text-gray-500 dark:text-gray-400">{{ tag }}</p>
                  <div style="max-height: 144px; position: relative;">
                    <Line :data="tbChartData(tag)" :options="tbChartOptions" />
                  </div>
                </div>
              </div>
            </div>

            <!-- Searchable log viewer -->
            <div class="card space-y-3">
              <div class="flex items-center justify-between flex-wrap gap-2">
                <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
                  Training Log
                  <span class="text-xs font-normal text-gray-400 ml-1">
                    ({{ filteredLogLines.length }}/{{ logData.lines.length }} lines)
                  </span>
                </h3>
                <div class="flex items-center gap-2">
                  <input v-model="logSearch" class="input !py-1 !text-xs w-44" placeholder="Filter lines..." />
                  <button v-if="logSearch" class="text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300" @click="logSearch = ''">
                    <X class="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              <LogViewer :lines="filteredLogLines" :max-height="360" :auto-scroll="false" @clear="logSearch = ''" />
            </div>
          </template>

          <!-- Load error -->
          <div
            v-else-if="detailError"
            class="card rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800
                   text-sm text-red-700 dark:text-red-400 px-4 py-3 flex items-center gap-2"
          >
            <AlertCircle class="w-4 h-4 shrink-0" />
            {{ detailError }}
          </div>

        </div>
      </div>
    </div>

    <!-- Confirm delete dialog -->
    <ConfirmDialog
      :open="confirmDelete"
      title="Delete Run"
      :message="`Permanently delete run #${selected ? selected.run_id.slice(0, 8) : ''}? Log files and checkpoints are not removed.`"
      confirm-label="Delete"
      variant="danger"
      @confirm="deleteRun"
      @cancel="confirmDelete = false"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useToast } from 'vue-toastification'
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
import { RefreshCw, History, Loader2, MousePointerClick, Trash2, X, AlertCircle } from 'lucide-vue-next'

import PageHeader    from '@/components/PageHeader.vue'
import StatusBadge   from '@/components/StatusBadge.vue'
import ProgressBar   from '@/components/ProgressBar.vue'
import LossChart     from '@/components/LossChart.vue'
import LogViewer     from '@/components/LogViewer.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'

import { trainingApi } from '@/api/training'
import { historyApi }  from '@/api/history'
import type { ParsedLog, TensorboardData } from '@/api/history'
import type { TrainingRun, ModelType, RunStatus } from '@/types'
import type { LossPoint } from '@/composables/useTraining'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

// ── Constants ──────────────────────────────────────────────────────────────────
const MODEL_OPTIONS = [
  { value: 'baseline', label: 'Baseline CNN' },
  { value: 'unet',     label: 'U-Net'        },
  { value: 'gan',      label: 'Pix2Pix GAN'  },
  { value: 'fusion',   label: 'Fusion GAN'   },
] as const

const MODEL_LABELS: Record<ModelType, string> = {
  baseline: 'Baseline CNN',
  unet:     'U-Net',
  gan:      'Pix2Pix GAN',
  fusion:   'Fusion GAN',
}

const C = {
  brand:  '#4f6ef7',
  yellow: '#f59e0b',
}

/** Detect dark mode from DOM */
function hasDarkClass(): boolean {
  return typeof document !== 'undefined'
    && (document.documentElement.classList.contains('dark')
        || document.querySelector('.dark') !== null)
}
const chartText = computed(() => hasDarkClass() ? '#d1d5db' : '#9ca3af')
const chartGrid = computed(() => hasDarkClass() ? 'rgba(156,163,175,0.12)' : 'rgba(156,163,175,0.15)')

// ── State ──────────────────────────────────────────────────────────────────────
const runs          = ref<TrainingRun[]>([])
const selected      = ref<TrainingRun | null>(null)
const logData       = ref<ParsedLog | null>(null)
const tbData        = ref<TensorboardData | null>(null)
const loadingLogs   = ref(false)
const detailError   = ref<string | null>(null)
const refreshing    = ref(false)
const confirmDelete = ref(false)
const logSearch     = ref('')
const tbExpanded    = ref(false)
const filterModel   = ref<ModelType | ''>('')
const filterStatus  = ref<RunStatus | ''>('')
const sortKey       = ref<'started_at' | 'started_at_asc' | 'loss' | 'model'>('started_at')

const toast = useToast()

// ── Filtered + sorted runs ─────────────────────────────────────────────────────
const filteredRuns = computed((): TrainingRun[] => {
  let list = [...runs.value]
  if (filterModel.value)  list = list.filter(r => r.model === filterModel.value)
  if (filterStatus.value) list = list.filter(r => r.status === filterStatus.value)
  list.sort((a, b) => {
    if (sortKey.value === 'started_at')     return b.started_at - a.started_at
    if (sortKey.value === 'started_at_asc') return a.started_at - b.started_at
    if (sortKey.value === 'loss')           return (a.loss ?? Infinity) - (b.loss ?? Infinity)
    return a.model.localeCompare(b.model)
  })
  return list
})

// ── Hyperparameter entries ────────────────────────────────────────────────────
const paramEntries = computed(() => {
  if (!selected.value?.params) return []
  return Object.entries(selected.value.params)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([key, val]) => ({ key, val }))
})

// ── Loss chart: ParsedLog → LossPoint[] ──────────────────────────────────────
const lossPoints = computed((): LossPoint[] => {
  if (!logData.value) return []
  const { epochs, losses } = logData.value
  const isGan = selected.value?.model === 'gan' || selected.value?.model === 'fusion'

  // For GAN models, try to extract dual D/G loss from TensorBoard tags
  if (isGan && tbData.value) {
    const tags = tbData.value.tags
    const gKey = Object.keys(tags).find(k => /generator|loss_g|loss\/g/i.test(k))
    const dKey = Object.keys(tags).find(k => /discriminator|loss_d|loss\/d/i.test(k))
    if (gKey && dKey) {
      const gS = tags[gKey]
      const dS = tags[dKey]
      const len = Math.min(gS.length, dS.length)
      return Array.from({ length: len }, (_, i) => ({
        epoch: i + 1,
        loss:  gS[i].value,
        lossG: gS[i].value,
        lossD: dS[i].value,
      }))
    }
  }

  return epochs.slice(0, losses.length).map((ep, i) => ({
    epoch: ep,
    loss:  losses[i],
  }))
})

// ── LR schedule chart ─────────────────────────────────────────────────────────
const lrChartLabels = computed((): string[] => {
  if (!logData.value?.lrs.length) return []
  const { epochs, lrs } = logData.value
  const len = Math.min(epochs.length || lrs.length, lrs.length)
  if (!len) return []
  return epochs.slice(0, len).map(e => `Ep ${e}`)
})

const lrChartData = computed(() => ({
  labels: lrChartLabels.value,
  datasets: [{
    label: 'Learning Rate',
    data: logData.value?.lrs.slice(0, lrChartLabels.value.length) ?? [],
    borderColor:     C.yellow,
    backgroundColor: C.yellow + '20',
    borderWidth: 2,
    pointRadius: 2,
    fill: false,
    tension: 0.2,
  }],
}))

const lrChartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 },
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (ctx: TooltipItem<'line'>) => ` LR: ${(ctx.parsed as { y: number }).y.toExponential(3)}`,
      },
    },
  },
  scales: {
    x: { ticks: { color: chartText.value, maxTicksLimit: 10, font: { size: 10 } }, grid: { color: chartGrid.value } },
    y: {
      ticks: {
        color: chartText.value, font: { size: 10 },
        callback: (v: number | string) => typeof v === 'number' ? v.toExponential(1) : v,
      },
      grid: { color: chartGrid.value },
    },
  },
}))

// ── TensorBoard scalars ────────────────────────────────────────────────────────
const tbTags = computed((): string[] =>
  tbData.value ? Object.keys(tbData.value.tags) : [],
)

const tbChartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 },
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (ctx: TooltipItem<'line'>) => ` ${(ctx.parsed as { y: number }).y.toFixed(4)}`,
      },
    },
  },
  scales: {
    x: { ticks: { color: chartText.value, maxTicksLimit: 8, font: { size: 9 } }, grid: { color: chartGrid.value } },
    y: { ticks: { color: chartText.value, font: { size: 9 } }, grid: { color: chartGrid.value } },
  },
}))

function tbChartData(tag: string) {
  if (!tbData.value?.tags[tag]) return { labels: [], datasets: [] }
  const series = tbData.value.tags[tag]
  return {
    labels: series.map(p => String(p.step)),
    datasets: [{
      label: tag,
      data: series.map(p => p.value),
      borderColor: C.brand,
      backgroundColor: C.brand + '20',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      tension: 0.2,
    }],
  }
}

// ── Log search ────────────────────────────────────────────────────────────────
const filteredLogLines = computed((): string[] => {
  if (!logData.value?.lines) return []
  const q = logSearch.value.trim().toLowerCase()
  if (!q) return logData.value.lines
  return logData.value.lines.filter(l => l.toLowerCase().includes(q))
})

// ── Lifecycle ──────────────────────────────────────────────────────────────────
onMounted(loadRuns)

watch(filteredRuns, list => {
  if (selected.value && !list.some(r => r.run_id === selected.value!.run_id)) {
    selected.value = null
    logData.value  = null
    tbData.value   = null
  }
})

// ── Actions ────────────────────────────────────────────────────────────────────
async function loadRuns() {
  refreshing.value = true
  try {
    runs.value = await trainingApi.listRuns()
  } catch {
    toast.error('Failed to load training runs.')
  } finally {
    refreshing.value = false
  }
}

async function selectRun(run: TrainingRun) {
  if (selected.value?.run_id === run.run_id) {
    selected.value = null
    logData.value  = null
    tbData.value   = null
    return
  }
  selected.value    = run
  logData.value     = null
  tbData.value      = null
  detailError.value = null
  logSearch.value   = ''
  tbExpanded.value  = false
  loadingLogs.value = true

  try {
    const [logs, tb] = await Promise.allSettled([
      historyApi.getLogs(run.run_id),
      historyApi.getTensorboardData(run.model),
    ])
    logData.value = logs.status === 'fulfilled' ? logs.value : null
    tbData.value  = tb.status  === 'fulfilled' ? tb.value  : null
    if (logs.status === 'rejected') {
      detailError.value = `Could not load logs: ${(logs.reason as Error).message}`
    }
  } finally {
    loadingLogs.value = false
  }
}

async function deleteRun() {
  if (!selected.value) return
  const runId = selected.value.run_id
  confirmDelete.value = false
  try {
    await historyApi.deleteRun(runId)
    runs.value     = runs.value.filter(r => r.run_id !== runId)
    selected.value = null
    logData.value  = null
    tbData.value   = null
    toast.success('Run deleted.')
  } catch {
    toast.error('Failed to delete run.')
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

function formatDuration(start: number, end: number): string {
  const s = Math.round(end - start)
  if (s < 60)   return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

function pbarColor(status: RunStatus): 'brand' | 'green' | 'red' | 'yellow' {
  const map: Record<RunStatus, 'brand' | 'green' | 'red' | 'yellow'> = {
    running: 'brand', finished: 'green', failed: 'red', stopped: 'yellow',
  }
  return map[status]
}
</script>

<style scoped>
.list-move,
.list-enter-active,
.list-leave-active { transition: all 0.2s ease; }
.list-enter-from,
.list-leave-to { opacity: 0; transform: translateX(-8px); }
.list-leave-active { position: absolute; width: 100%; }
</style>

