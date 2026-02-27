<template>
  <div class="p-6 space-y-6">
    <!-- ── Page header ──────────────────────────────────────────────────────── -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-50">Dashboard</h1>
      <p class="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
        Overview of your Image Colorization Ensemble
      </p>
    </div>

    <!-- ── Stat cards ───────────────────────────────────────────────────────── -->
    <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
      <div
        v-for="stat in statCards"
        :key="stat.label"
        class="card flex items-center justify-between"
      >
        <div>
          <p class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
            {{ stat.label }}
          </p>
          <p class="text-3xl font-bold text-gray-900 dark:text-gray-50 mt-1">
            <span v-if="loading">—</span>
            <span v-else>{{ stat.value }}</span>
          </p>
        </div>
        <div :class="['p-3 rounded-xl shrink-0', stat.bgClass]">
          <component :is="stat.icon" :class="['w-5 h-5', stat.iconClass]" />
        </div>
      </div>
    </div>

    <!-- ── Quick actions ────────────────────────────────────────────────────── -->
    <section>
      <h2 class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-3">
        Quick Actions
      </h2>
      <div class="flex flex-wrap gap-3">
        <RouterLink to="/training" class="btn btn-primary flex items-center gap-2">
          <Play class="w-4 h-4" />
          Start Training
        </RouterLink>
        <RouterLink to="/colorize" class="btn btn-secondary flex items-center gap-2">
          <Wand2 class="w-4 h-4" />
          Colorize Image
        </RouterLink>
        <RouterLink to="/batch" class="btn btn-secondary flex items-center gap-2">
          <Layers class="w-4 h-4" />
          Batch Colorize
        </RouterLink>
        <RouterLink to="/metrics" class="btn btn-secondary flex items-center gap-2">
          <BarChart2 class="w-4 h-4" />
          Evaluate Models
        </RouterLink>
      </div>
    </section>

    <!-- ── Two-column widgets ────────────────────────────────────────────────── -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

      <!-- Recent Runs -->
      <div class="card">
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">
          Recent Training Runs
        </h2>

        <!-- Loading -->
        <div v-if="loadingRuns" class="space-y-2">
          <div v-for="i in 4" :key="i" class="h-7 bg-gray-100 dark:bg-gray-700 rounded animate-pulse" />
        </div>

        <!-- Empty -->
        <p v-else-if="recentRuns.length === 0" class="text-sm text-gray-400 dark:text-gray-500 py-2">
          No training runs yet.
          <RouterLink to="/training" class="text-brand-500 hover:underline ml-1">Start one →</RouterLink>
        </p>

        <!-- List -->
        <ul v-else class="divide-y divide-gray-100 dark:divide-gray-700">
          <li
            v-for="run in recentRuns"
            :key="run.run_id"
            class="flex items-center justify-between py-2.5 gap-2"
          >
            <div class="flex items-center gap-2 min-w-0">
              <span class="font-medium text-gray-800 dark:text-gray-200 text-sm capitalize">
                {{ modelDisplayName(run.model) }}
              </span>
              <span class="text-xs text-gray-400 dark:text-gray-500 font-mono">{{ run.run_id.slice(0, 8) }}</span>
            </div>
            <div class="flex items-center gap-2 shrink-0">
              <span v-if="run.epoch != null" class="text-xs text-gray-500 dark:text-gray-400">
                ep {{ run.epoch }}/{{ run.total_epochs }}
              </span>
              <span :class="['badge', statusBadgeClass(run.status)]">{{ run.status }}</span>
            </div>
          </li>
        </ul>

        <RouterLink
          to="/history"
          class="mt-4 flex items-center gap-1 text-xs text-brand-500 hover:text-brand-600 transition-colors"
        >
          View all runs <ChevronRight class="w-3.5 h-3.5" />
        </RouterLink>
      </div>

      <!-- Checkpoints by model -->
      <div class="card">
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">
          Checkpoints by Model
        </h2>

        <!-- Loading -->
        <div v-if="loadingCkpts" class="space-y-2">
          <div v-for="i in 4" :key="i" class="h-7 bg-gray-100 dark:bg-gray-700 rounded animate-pulse" />
        </div>

        <!-- Empty -->
        <p v-else-if="checkpointsByModel.length === 0" class="text-sm text-gray-400 dark:text-gray-500 py-2">
          No checkpoints found in
          <code class="text-xs bg-gray-100 dark:bg-gray-700 px-1 rounded">outputs/checkpoints/</code>.
        </p>

        <!-- List -->
        <ul v-else class="divide-y divide-gray-100 dark:divide-gray-700">
          <li
            v-for="group in checkpointsByModel"
            :key="group.model"
            class="flex items-center justify-between py-2.5 gap-2"
          >
            <div class="flex items-center gap-2 min-w-0">
              <HardDrive class="w-4 h-4 text-gray-400 dark:text-gray-500 shrink-0" />
              <span class="font-medium text-gray-800 dark:text-gray-200 text-sm">{{ group.model }}</span>
            </div>
            <div class="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400 shrink-0">
              <span>{{ group.count }} file{{ group.count !== 1 ? 's' : '' }}</span>
              <span>{{ group.totalMb.toFixed(0) }} MB</span>
            </div>
          </li>
        </ul>

        <RouterLink
          to="/colorize"
          class="mt-4 flex items-center gap-1 text-xs text-brand-500 hover:text-brand-600 transition-colors"
        >
          Use a checkpoint <ChevronRight class="w-3.5 h-3.5" />
        </RouterLink>
      </div>
    </div>

    <!-- ── Error banner ──────────────────────────────────────────────────────── -->
    <div
      v-if="error"
      class="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800
             px-4 py-3 text-sm text-red-700 dark:text-red-300 flex items-center gap-2"
    >
      <AlertCircle class="w-4 h-4 shrink-0" />
      {{ error }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import {
  Play,
  Wand2,
  Layers,
  BarChart2,
  Database,
  Activity,
  CheckSquare,
  Box,
  HardDrive,
  ChevronRight,
  AlertCircle,
} from 'lucide-vue-next'
import { modelsApi } from '@/api/models'
import { trainingApi } from '@/api/training'
import type { CheckpointInfo, TrainingRun, RunStatus, ModelType } from '@/types'

// ── Data ───────────────────────────────────────────────────────────────────────
const checkpoints = ref<CheckpointInfo[]>([])
const runs        = ref<TrainingRun[]>([])
const loadingCkpts = ref(true)
const loadingRuns  = ref(true)
const error        = ref<string | null>(null)

const loading = computed(() => loadingCkpts.value || loadingRuns.value)

onMounted(async () => {
  await Promise.allSettled([
    modelsApi.listCheckpoints()
      .then((data) => { checkpoints.value = data })
      .catch(() => { error.value = 'Failed to load checkpoints.' })
      .finally(() => { loadingCkpts.value = false }),

    trainingApi.listRuns()
      .then((data) => { runs.value = data })
      .catch(() => { /* non-critical, silently skip */ })
      .finally(() => { loadingRuns.value = false }),
  ])
})

// ── Derived data ───────────────────────────────────────────────────────────────
const activeRuns   = computed(() => runs.value.filter((r) => r.status === 'running'))
const finishedRuns = computed(() => runs.value.filter((r) => r.status === 'finished'))

/** Most recent 5 runs (sorted by start time descending) */
const recentRuns = computed(() =>
  [...runs.value]
    .sort((a, b) => b.started_at - a.started_at)
    .slice(0, 5),
)

/** Checkpoint counts grouped by inferred model type */
const checkpointsByModel = computed(() => {
  const map = new Map<string, { count: number; totalMb: number }>()
  for (const ck of checkpoints.value) {
    const model = inferModel(ck.filename)
    const existing = map.get(model) ?? { count: 0, totalMb: 0 }
    map.set(model, { count: existing.count + 1, totalMb: existing.totalMb + (ck.size_mb ?? 0) })
  }
  return [...map.entries()]
    .map(([model, data]) => ({ model, ...data }))
    .sort((a, b) => b.count - a.count)
})

// ── Stat cards ─────────────────────────────────────────────────────────────────
const statCards = computed(() => [
  {
    label: 'Total Checkpoints',
    value: checkpoints.value.length,
    icon: Database,
    bgClass: 'bg-blue-50 dark:bg-blue-900/20',
    iconClass: 'text-blue-500',
  },
  {
    label: 'Active Runs',
    value: activeRuns.value.length,
    icon: Activity,
    bgClass: 'bg-green-50 dark:bg-green-900/20',
    iconClass: 'text-green-500',
  },
  {
    label: 'Finished Runs',
    value: finishedRuns.value.length,
    icon: CheckSquare,
    bgClass: 'bg-brand-50 dark:bg-brand-900/20',
    iconClass: 'text-brand-500',
  },
  {
    label: 'Available Models',
    value: 4,
    icon: Box,
    bgClass: 'bg-purple-50 dark:bg-purple-900/20',
    iconClass: 'text-purple-500',
  },
])

// ── Helpers ────────────────────────────────────────────────────────────────────
function inferModel(filename: string): string {
  const f = filename.toLowerCase()
  if (f.startsWith('fusion'))                    return 'Fusion GAN'
  if (f.startsWith('unet') || f.startsWith('u_net')) return 'U-Net'
  if (f.startsWith('gan') || f.startsWith('disc') || f.startsWith('pix2pix')) return 'Pix2Pix GAN'
  if (f.startsWith('baseline'))                  return 'Baseline CNN'
  return 'Other'
}

function modelDisplayName(model: ModelType): string {
  const map: Record<ModelType, string> = {
    baseline: 'Baseline CNN',
    unet:     'U-Net',
    gan:      'Pix2Pix GAN',
    fusion:   'Fusion GAN',
  }
  return map[model] ?? model
}

function statusBadgeClass(status: RunStatus): string {
  const map: Record<RunStatus, string> = {
    running:  'badge-blue',
    finished: 'badge-green',
    failed:   'badge-red',
    stopped:  'badge-yellow',
  }
  return map[status] ?? 'badge-gray'
}
</script>
