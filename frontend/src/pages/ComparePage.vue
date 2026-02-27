<template>
  <div class="p-6 space-y-6">
    <!-- ── Page header ──────────────────────────────────────────────────────── -->
    <PageHeader
      title="Model Comparison"
      description="Upload a single image and compare colorization results side-by-side across multiple models."
    />

    <!-- Setup card -->
    <div class="card space-y-5">
      <div>
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Reference Image</h2>
        <ImageDropzone
          :file="sourceFile"
          :preview-max-height="200"
          @file-selected="onFileSelected"
          @file-cleared="onFileCleared"
        />
      </div>

      <div>
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
            Models
            <span class="text-xs font-normal text-gray-400 dark:text-gray-500 ml-1">({{ slots.length }}/{{ MAX_SLOTS }})</span>
          </h2>
          <button
            v-if="slots.length < MAX_SLOTS"
            class="btn btn-secondary flex items-center gap-1 !py-1 !px-3 text-xs"
            @click="addSlot"
          >
            <Plus class="w-3.5 h-3.5" />
            Add model
          </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div
            v-for="(slot, i) in slots"
            :key="i"
            class="rounded-xl border border-gray-200 dark:border-gray-700 p-4 space-y-3"
            :style="{ borderLeftColor: SLOT_COLORS[i], borderLeftWidth: '3px' }"
          >
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <span class="w-2.5 h-2.5 rounded-full" :style="{ background: SLOT_COLORS[i] }" />
                <input
                  v-model="slot.label"
                  class="input !py-0.5 !text-xs font-medium w-28"
                  :placeholder="`Model ${i + 1}`"
                />
              </div>
              <button
                v-if="slots.length > MIN_SLOTS"
                class="text-gray-400 dark:text-gray-500 hover:text-red-500 transition-colors"
                @click="removeSlot(i)"
              >
                <X class="w-4 h-4" />
              </button>
            </div>
            <ModelSelector
              :model="slot.model"
              :checkpoint="slot.checkpoint"
              :checkpoints="checkpointsForModel(slot.model)"
              :loading="loadingMeta"
              @update:model="v => { slot.model = v; slot.checkpoint = '' }"
              @update:checkpoint="v => slot.checkpoint = v"
            />
          </div>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <button
          class="btn btn-primary flex items-center gap-2"
          :disabled="!canCompare || loading"
          @click="runComparison"
        >
          <Loader2 v-if="loading" class="w-4 h-4 animate-spin" />
          <GitCompare v-else class="w-4 h-4" />
          {{ loading ? 'Comparing…' : 'Run Comparison' }}
        </button>
        <StatusBadge v-if="loading" status="running" :label="`${doneCount}/${slots.length} done`" />
      </div>
    </div>

    <!-- Error banner -->
    <transition name="fade">
      <div
        v-if="error"
        class="rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800
               px-4 py-3 text-sm text-red-700 dark:text-red-400 flex items-center gap-2"
      >
        <AlertCircle class="w-4 h-4 shrink-0" /> {{ error }}
      </div>
    </transition>

    <!-- Results -->
    <transition name="slide">
      <div v-if="results.length" class="space-y-6">
        <div class="card space-y-3">
          <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Metric Radar</h2>
          <div class="max-w-sm mx-auto">
            <RadarChart :slots="radarSlots" />
          </div>
        </div>

        <div class="grid gap-5" :class="gridClass">
          <div v-for="(r, i) in results" :key="i" class="card space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full" :style="{ background: SLOT_COLORS[i] }" />
                <span class="font-semibold text-sm text-gray-800 dark:text-gray-200">{{ r.label }}</span>
                <span class="text-xs text-gray-400 dark:text-gray-500">{{ r.model }}</span>
              </div>
              <div
                v-if="r.isWinner"
                class="flex items-center gap-1 px-2 py-0.5 rounded-full
                       bg-amber-50 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-700"
              >
                <Trophy class="w-3.5 h-3.5 text-amber-500" />
                <span class="text-xs font-semibold text-amber-700 dark:text-amber-400">Best</span>
              </div>
            </div>

            <ImageCompare
              :panels="[
                { label: 'Input', src: `data:image/png;base64,${r.result.grayscale}` },
                { label: 'Colourized', src: `data:image/png;base64,${r.result.colorized}` },
              ]"
              :max-height="260"
            />

            <MetricsCards
              :psnr="r.result.metrics.psnr"
              :ssim="r.result.metrics.ssim"
              :columns="2"
            />
          </div>
        </div>
      </div>
    </transition>

    <!-- Empty state -->
    <div
      v-if="!results.length && !loading"
      class="card flex flex-col items-center justify-center py-16 gap-3"
    >
      <GitCompare class="w-12 h-12 text-gray-300 dark:text-gray-600" />
      <p class="text-sm text-gray-500 dark:text-gray-400">
        Select an image, configure model slots, then click <strong>Run Comparison</strong>.
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useToast } from 'vue-toastification'
import { Plus, X, Loader2, GitCompare, Trophy, AlertCircle } from 'lucide-vue-next'

import PageHeader    from '@/components/PageHeader.vue'
import ImageDropzone from '@/components/ImageDropzone.vue'
import ImageCompare  from '@/components/ImageCompare.vue'
import MetricsCards  from '@/components/MetricsCards.vue'
import ModelSelector from '@/components/ModelSelector.vue'
import StatusBadge   from '@/components/StatusBadge.vue'
import RadarChart    from '@/components/RadarChart.vue'
import type { RadarSlot } from '@/components/RadarChart.vue'

import { inferenceApi } from '@/api/inference'
import { modelsApi }    from '@/api/models'
import type { ColorizeResult, CheckpointInfo, ModelType } from '@/types'

const MIN_SLOTS   = 1
const MAX_SLOTS   = 4
const SLOT_COLORS = ['#4f6ef7', '#10b981', '#f59e0b', '#ef4444']

interface Slot {
  label:      string
  model:      ModelType
  checkpoint: string
}

interface CompareResult {
  label:    string
  model:    ModelType
  result:   ColorizeResult
  isWinner: boolean
}

const allCheckpoints = ref<CheckpointInfo[]>([])
const loadingMeta    = ref(true)
const loading        = ref(false)
const doneCount      = ref(0)
const error          = ref<string | null>(null)
const sourceFile     = ref<File | null>(null)
const results        = ref<CompareResult[]>([])

const slots = reactive<Slot[]>([
  { label: 'Baseline', model: 'baseline', checkpoint: '' },
  { label: 'U-Net',    model: 'unet',     checkpoint: '' },
])

const toast = useToast()

const canCompare = computed(
  () => sourceFile.value !== null && slots.every(s => s.checkpoint !== ''),
)

const gridClass = computed(() => {
  const n = results.value.length
  if (n === 1) return 'grid-cols-1 max-w-lg'
  if (n === 2) return 'grid-cols-1 lg:grid-cols-2'
  if (n === 3) return 'grid-cols-1 lg:grid-cols-3'
  return 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-4'
})

const radarSlots = computed((): RadarSlot[] =>
  results.value.map((r, i) => ({
    label: r.label,
    psnr:  r.result.metrics.psnr,
    ssim:  r.result.metrics.ssim,
    color: SLOT_COLORS[i],
  })),
)

onMounted(async () => {
  try {
    allCheckpoints.value = await modelsApi.listCheckpoints()
  } finally {
    loadingMeta.value = false
  }
})

function onFileSelected(file: File) {
  sourceFile.value = file
  results.value    = []
  error.value      = null
}

function onFileCleared() {
  sourceFile.value = null
  results.value    = []
  error.value      = null
}

function addSlot() {
  if (slots.length >= MAX_SLOTS) return
  slots.push({ label: `Model ${slots.length + 1}`, model: 'unet', checkpoint: '' })
}

function removeSlot(i: number) {
  if (slots.length <= MIN_SLOTS) return
  slots.splice(i, 1)
}

function checkpointsForModel(model: ModelType): CheckpointInfo[] {
  return allCheckpoints.value.filter(c => !c.model_hint || c.model_hint === model)
}

async function runComparison() {
  if (!sourceFile.value) return
  loading.value   = true
  doneCount.value = 0
  error.value     = null
  results.value   = []

  try {
    const settled = await Promise.allSettled(
      slots.map(async (s): Promise<CompareResult> => {
        const r = await inferenceApi.colorize(
          sourceFile.value!,
          s.model,
          s.checkpoint,
          'grayscale',
        )
        doneCount.value++
        return { label: s.label, model: s.model, result: r, isWinner: false }
      })
    )

    const fulfilled: CompareResult[] = []
    const errors: string[] = []

    settled.forEach((res, i) => {
      if (res.status === 'fulfilled') {
        fulfilled.push(res.value)
      } else {
        errors.push(`${slots[i].label}: ${(res.reason as Error).message}`)
      }
    })

    if (errors.length) toast.error(`Some models failed:\n${errors.join('\n')}`)

    if (fulfilled.length > 1) {
      let bestScore = -Infinity
      let bestIdx   = 0
      fulfilled.forEach((r, i) => {
        const score = 0.5 * ((r.result.metrics.psnr ?? 0) / 40) + 0.5 * (r.result.metrics.ssim ?? 0)
        if (score > bestScore) { bestScore = score; bestIdx = i }
      })
      fulfilled[bestIdx].isWinner = true
    }

    results.value = fulfilled
    if (fulfilled.length) toast.success(`Compared ${fulfilled.length} model${fulfilled.length > 1 ? 's' : ''}.`)
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Comparison failed.'
    error.value = msg
    toast.error(msg)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.fade-enter-active, .fade-leave-active  { transition: opacity .2s ease; }
.fade-enter-from,  .fade-leave-to       { opacity: 0; }

.slide-enter-active, .slide-leave-active { transition: opacity .25s ease, transform .25s ease; }
.slide-enter-from,   .slide-leave-to     { opacity: 0; transform: translateY(8px); }
</style>
