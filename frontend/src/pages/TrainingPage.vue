<template>
  <div class="p-6 space-y-6">
    <!-- ── Page header ──────────────────────────────────────────────────────── -->
    <PageHeader
      title="Training"
      description="Configure and launch a training run. Watch live progress via Server-Sent Events."
    />

    <!-- ── Configuration card ───────────────────────────────────────────────── -->
    <div class="card space-y-5">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Configuration</h2>
        <span v-if="isRunning" class="badge badge-blue flex items-center gap-1.5">
          <span class="block w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          Training in progress
        </span>
      </div>

      <!-- Model selector -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label class="label">Model</label>
          <select v-model="params.model" class="select" :disabled="isRunning">
            <option value="baseline">Baseline CNN</option>
            <option value="unet">U-Net</option>
            <option value="gan">Pix2Pix GAN</option>
            <option value="fusion">Fusion GAN</option>
          </select>
        </div>
        <div>
          <label class="label">Data Path</label>
          <input
            v-model="params.data_path"
            class="input"
            placeholder="data/coco/val2017"
            :disabled="isRunning"
          />
        </div>
      </div>

      <!-- Common params -->
      <div class="grid grid-cols-2 sm:grid-cols-3 gap-4">
        <div>
          <label class="label">Epochs</label>
          <input
            v-model.number="params.epochs"
            type="number" min="1" max="1000"
            class="input" :disabled="isRunning"
          />
        </div>
        <div>
          <label class="label">Batch Size</label>
          <input
            v-model.number="params.batch_size"
            type="number" min="1" max="256"
            class="input" :disabled="isRunning"
          />
        </div>
        <div>
          <label class="label">Learning Rate</label>
          <input
            v-model.number="params.lr"
            type="number" step="0.00001" min="0.000001"
            class="input" :disabled="isRunning"
          />
        </div>
      </div>

      <!-- GAN / Fusion extras -->
      <Transition name="slide">
        <div v-if="isGanModel" class="space-y-4 pt-1">
          <div class="border-t border-gray-100 dark:border-gray-700 pt-4">
            <p class="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">
              GAN / Fusion Options
            </p>
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <!-- Lambda L1 -->
              <div>
                <label class="label">Lambda L1</label>
                <input
                  v-model.number="params.lambda_l1"
                  type="number" min="0"
                  class="input" :disabled="isRunning"
                />
              </div>

              <!-- Resume Generator -->
              <div>
                <label class="label">Resume Generator</label>
                <select v-model="params.resume_g" class="select" :disabled="isRunning">
                  <option value="">— start fresh —</option>
                  <option
                    v-for="ck in ganCheckpoints"
                    :key="ck.path"
                    :value="ck.path"
                  >
                    {{ ck.filename }} ({{ ck.size_mb }} MB)
                  </option>
                </select>
              </div>

              <!-- Resume Discriminator -->
              <div>
                <label class="label">Resume Discriminator</label>
                <select v-model="params.resume_d" class="select" :disabled="isRunning">
                  <option value="">— start fresh —</option>
                  <option
                    v-for="ck in discCheckpoints"
                    :key="ck.path"
                    :value="ck.path"
                  >
                    {{ ck.filename }} ({{ ck.size_mb }} MB)
                  </option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </Transition>

      <!-- Action row -->
      <div class="flex items-center gap-3 pt-1">
        <button
          class="btn btn-primary flex items-center gap-2"
          :disabled="isRunning || starting"
          @click="handleStart"
        >
          <Loader2 v-if="starting" class="w-4 h-4 animate-spin" />
          <Play v-else class="w-4 h-4" />
          {{ starting ? 'Starting…' : 'Start Training' }}
        </button>

        <button
          v-if="isRunning"
          class="btn btn-danger flex items-center gap-2"
          :disabled="stopping"
          @click="confirmStop = true"
        >
          <Square class="w-4 h-4" />
          Stop Training
        </button>

        <span v-if="startError" class="text-xs text-red-500 dark:text-red-400 flex items-center gap-1">
          <AlertCircle class="w-3.5 h-3.5" />
          {{ startError }}
        </span>
      </div>
    </div>

    <!-- ── Live progress panel ───────────────────────────────────────────────── -->
    <Transition name="slide">
      <div v-if="status" class="space-y-4">

        <!-- Run summary header -->
        <div class="card">
          <div class="flex items-center justify-between gap-4 mb-4 flex-wrap">
            <div class="flex items-center gap-3">
              <StatusBadge :status="status.status" />
              <span class="text-xs text-gray-400 dark:text-gray-500 font-mono">
                run {{ runId?.slice(0, 8) }}…
              </span>
              <span
                v-if="sseConnected"
                class="flex items-center gap-1 text-[11px] text-green-500 dark:text-green-400"
                title="SSE stream connected"
              >
                <Wifi class="w-3 h-3" /> live
              </span>
              <span
                v-else
                class="flex items-center gap-1 text-[11px] text-gray-400 dark:text-gray-500"
                title="SSE disconnected — polling"
              >
                <WifiOff class="w-3 h-3" /> reconnecting…
              </span>
            </div>
          </div>

          <!-- Progress bar -->
          <ProgressBar
            :pct="progressPct"
            :label="`Epoch ${status.epoch} / ${status.total_epochs}`"
            class="mb-4"
          />

          <!-- Loss values -->
          <div class="flex flex-wrap gap-6 text-sm">
            <template v-if="isGan">
              <div>
                <span class="text-xs text-gray-400 dark:text-gray-500 block">Generator Loss</span>
                <span class="font-mono font-semibold text-brand-500 dark:text-brand-400">
                  {{ status.loss != null ? status.loss.toFixed(5) : '—' }}
                </span>
              </div>
              <div v-if="latestLossD != null">
                <span class="text-xs text-gray-400 dark:text-gray-500 block">Discriminator Loss</span>
                <span class="font-mono font-semibold text-red-500 dark:text-red-400">
                  {{ latestLossD.toFixed(5) }}
                </span>
              </div>
            </template>
            <template v-else>
              <div>
                <span class="text-xs text-gray-400 dark:text-gray-500 block">Loss</span>
                <span class="font-mono font-semibold text-green-600 dark:text-green-400">
                  {{ status.loss != null ? status.loss.toFixed(5) : '—' }}
                </span>
              </div>
            </template>
          </div>
        </div>

        <!-- Loss chart -->
        <div class="card">
          <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Loss Curve
          </h3>
          <LossChart :data="lossHistory" />
        </div>

        <!-- Log viewer -->
        <div class="card p-0 overflow-hidden">
          <LogViewer
            :lines="logLines"
            :max-height="280"
            @clear="logLines.splice(0)"
          />
        </div>
      </div>
    </Transition>

    <!-- ── Confirm stop dialog ───────────────────────────────────────────────── -->
    <ConfirmDialog
      :open="confirmStop"
      title="Stop Training?"
      message="The current run will be terminated. Checkpoints already saved to disk will not be lost."
      confirm-label="Stop"
      variant="danger"
      @confirm="handleStop"
      @cancel="confirmStop = false"
    >
      <template #icon>
        <div class="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
          <OctagonX class="w-6 h-6 text-red-500" />
        </div>
      </template>
    </ConfirmDialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useToast } from 'vue-toastification'
import {
  Play, Square, Loader2,
  Wifi, WifiOff, AlertCircle, OctagonX,
} from 'lucide-vue-next'

import StatusBadge    from '@/components/StatusBadge.vue'
import PageHeader     from '@/components/PageHeader.vue'
import ProgressBar    from '@/components/ProgressBar.vue'
import LogViewer      from '@/components/LogViewer.vue'
import LossChart      from '@/components/LossChart.vue'
import ConfirmDialog  from '@/components/ConfirmDialog.vue'

import { useTraining }  from '@/composables/useTraining'
import { modelsApi }    from '@/api/models'
import type { TrainingParams, ModelType, CheckpointInfo } from '@/types'

const toast = useToast()

// ── Training composable ────────────────────────────────────────────────────────
const {
  runId, status, lossHistory, logLines,
  starting, stopping, isRunning, progressPct, isGan,
  sseConnected,
  start, stop,
} = useTraining()

// ── Form state ─────────────────────────────────────────────────────────────────
const params = reactive<TrainingParams>({
  model:      'unet',
  epochs:     20,
  batch_size: 16,
  lr:         0.0002,
  lambda_l1:  100,
  data_path:  'data/coco/val2017',
  resume_g:   '',
  resume_d:   '',
})

const isGanModel = computed(() =>
  params.model === 'gan' || params.model === 'fusion',
)

// ── Checkpoint lists (for resume dropdowns) ──────────────────────────────────
const allCheckpoints = ref<CheckpointInfo[]>([])

const ganCheckpoints = computed(() =>
  allCheckpoints.value.filter((ck) => {
    const f = ck.filename.toLowerCase()
    return params.model === 'fusion'
      ? f.startsWith('fusion')
      : f.startsWith('gan') || f.startsWith('pix2pix') || f.includes('generator')
  }),
)

const discCheckpoints = computed(() =>
  allCheckpoints.value.filter((ck) => {
    const f = ck.filename.toLowerCase()
    return f.startsWith('disc') || f.includes('discriminator') || f.startsWith('fusion')
  }),
)

// ── Latest discriminator loss (from most recent loss history entry) ───────────
const latestLossD = computed<number | null>(() => {
  const last = lossHistory.value[lossHistory.value.length - 1]
  return last?.lossD ?? null
})

// ── UI flags ───────────────────────────────────────────────────────────────────
const confirmStop = ref(false)
const startError  = ref<string | null>(null)

// ── Lifecycle ──────────────────────────────────────────────────────────────────
onMounted(() => {
  modelsApi.listCheckpoints()
    .then((list) => { allCheckpoints.value = list })
    .catch(() => { /* non-critical */ })
})

// ── Default batch size by model ────────────────────────────────────────────────
// Automatically adjust batch size hint when model changes (user can override)
function applyModelDefaults(model: ModelType) {
  if (model === 'baseline' || model === 'unet') {
    if (params.batch_size === 8)  params.batch_size = 16
  } else {
    if (params.batch_size === 16) params.batch_size = 8
  }
}

watch(() => params.model, applyModelDefaults)

// ── Actions ────────────────────────────────────────────────────────────────────
async function handleStart() {
  startError.value = null

  // Strip empty resume fields for non-GAN models
  const payload: TrainingParams = { ...params }
  if (!isGanModel.value) {
    delete payload.lambda_l1
    delete payload.resume_g
    delete payload.resume_d
  } else {
    if (!payload.resume_g) delete payload.resume_g
    if (!payload.resume_d) delete payload.resume_d
  }

  try {
    await start(payload)
    toast.success('Training run started!')
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Failed to start training.'
    startError.value = msg
    toast.error(msg)
  }
}

async function handleStop() {
  confirmStop.value = false
  try {
    await stop()
    toast.info('Training run stopped.')
  } catch (err: unknown) {
    toast.error(err instanceof Error ? err.message : 'Failed to stop run.')
  }
}
</script>

<style scoped>
.slide-enter-active,
.slide-leave-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}
.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  transform: translateY(8px);
}
</style>
