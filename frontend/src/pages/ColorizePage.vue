<template>
  <div class="p-6 space-y-6">
    <!-- ── Page header ──────────────────────────────────────────────────────── -->
    <PageHeader
      title="Colorize Image"
      description="Upload a grayscale or colour photo and let the selected model restore its colours."
    />

    <!-- ── Configuration card ───────────────────────────────────────────────── -->
    <div class="card space-y-5">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Settings</h2>
      </div>

      <!-- Mode toggle -->
      <div>
        <label class="label">Input Mode</label>
        <div class="flex gap-3 mt-1">
          <label
            v-for="opt in modeOptions"
            :key="opt.value"
            :class="[
              'flex-1 flex items-start gap-3 px-4 py-3 rounded-xl border-2 cursor-pointer transition-colors',
              config.mode === opt.value
                ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-brand-300',
            ]"
          >
            <input
              type="radio"
              :value="opt.value"
              v-model="config.mode"
              class="mt-0.5 accent-brand-500"
              :disabled="loading"
            />
            <div>
              <p class="text-sm font-medium text-gray-800 dark:text-gray-200">{{ opt.label }}</p>
              <p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{{ opt.hint }}</p>
            </div>
          </label>
        </div>
      </div>

      <!-- Model + Checkpoint selector -->
      <ModelSelector
        :model="config.model"
        :checkpoint="config.checkpoint"
        :checkpoints="checkpoints"
        :loading="loadingMeta"
        @update:model="v => { config.model = v; config.checkpoint = '' }"
        @update:checkpoint="v => config.checkpoint = v"
      />
    </div>

    <!-- ── Upload + Action ───────────────────────────────────────────────────── -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Drop zone -->
      <div class="card space-y-4">
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Image</h2>
        <ImageDropzone
          :file="selectedFile"
          :preview-max-height="280"
          @file-selected="onFileSelected"
          @file-cleared="onFileCleared"
        />
      </div>

      <!-- Action + feedback -->
      <div class="card flex flex-col justify-between space-y-4">
        <div class="space-y-3">
          <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Action</h2>

          <!-- File info -->
          <div v-if="selectedFile" class="rounded-lg bg-gray-50 dark:bg-gray-800 px-4 py-3 space-y-1">
            <div class="flex justify-between text-xs">
              <span class="text-gray-500 dark:text-gray-400">File</span>
              <span class="font-medium text-gray-700 dark:text-gray-300 truncate max-w-[60%]">
                {{ selectedFile.name }}
              </span>
            </div>
            <div class="flex justify-between text-xs">
              <span class="text-gray-500 dark:text-gray-400">Size</span>
              <span class="text-gray-700 dark:text-gray-300">{{ formatSize(selectedFile.size) }}</span>
            </div>
            <div class="flex justify-between text-xs">
              <span class="text-gray-500 dark:text-gray-400">Mode</span>
              <span class="text-gray-700 dark:text-gray-300">
                {{ config.mode === 'grayscale' ? 'Grayscale → Colour' : 'Colour → Re-colour' }}
              </span>
            </div>
          </div>
          <div v-else class="rounded-lg bg-gray-50 dark:bg-gray-800 px-4 py-6 text-center">
            <p class="text-sm text-gray-400 dark:text-gray-500">No image selected yet.</p>
          </div>

          <!-- Validation hints -->
          <p v-if="!config.checkpoint && !loadingMeta" class="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
            <AlertCircle class="w-3.5 h-3.5" />
            Select a checkpoint first.
          </p>
        </div>

        <!-- Colorize button -->
        <button
          class="btn btn-primary w-full"
          :disabled="!canColorize || loading"
          @click="runColorize"
        >
          <Loader2 v-if="loading" class="w-4 h-4 animate-spin" />
          <Wand2 v-else class="w-4 h-4" />
          {{ loading ? 'Colorizing…' : 'Colorize' }}
        </button>
      </div>
    </div>

    <!-- ── Error banner ──────────────────────────────────────────────────────── -->
    <transition name="fade">
      <div
        v-if="error"
        class="rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800
               px-4 py-3 text-sm text-red-700 dark:text-red-400 flex items-center gap-2"
      >
        <AlertCircle class="w-4 h-4 shrink-0" />
        {{ error }}
      </div>
    </transition>

    <!-- ── Results section ───────────────────────────────────────────────────── -->
    <transition name="slide">
      <div v-if="result" class="space-y-5">
        <div class="flex items-center justify-between">
          <h2 class="text-base font-semibold text-gray-800 dark:text-gray-200">Result</h2>
          <button
            class="btn btn-secondary flex items-center gap-2"
            @click="downloadResult"
          >
            <Download class="w-4 h-4" />
            Download
          </button>
        </div>

        <!-- Image comparison -->
        <div class="card">
          <ImageCompare
            :panels="resultPanels"
            :psnr="result.metrics.psnr ?? undefined"
            :ssim="result.metrics.ssim ?? undefined"
            :max-height="360"
          />
        </div>

        <!-- Metrics cards (only when comparing against ground truth in colour mode) -->
        <MetricsCards
          v-if="result.metrics.psnr !== null || result.metrics.ssim !== null"
          :psnr="result.metrics.psnr"
          :ssim="result.metrics.ssim"
          :columns="2"
        />

        <!-- How the pipeline works callout -->
        <div class="rounded-xl border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 px-4 py-3">
          <p class="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
            <strong>Pipeline:</strong>
            Input → L channel extracted → normalised to [0, 1] →
            model predicts <em>ab</em> channels → denormalised →
            L + ab concatenated → CIE L*a*b* → sRGB conversion.
          </p>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useToast } from 'vue-toastification'
import { Wand2, Download, Loader2, AlertCircle } from 'lucide-vue-next'

import PageHeader    from '@/components/PageHeader.vue'
import ImageDropzone from '@/components/ImageDropzone.vue'
import ImageCompare  from '@/components/ImageCompare.vue'
import MetricsCards  from '@/components/MetricsCards.vue'
import ModelSelector from '@/components/ModelSelector.vue'

import { inferenceApi } from '@/api/inference'
import { modelsApi }    from '@/api/models'
import type { ColorizeResult, CheckpointInfo, ModelType, ColorizeMode } from '@/types'

// ── Constants ──────────────────────────────────────────────────────────────────
const modeOptions = [
  {
    value: 'grayscale' as ColorizeMode,
    label: 'Grayscale → Colour',
    hint:  'Upload a black-and-white image. Model predicts the ab channels.',
  },
  {
    value: 'color_photo' as ColorizeMode,
    label: 'Colour → Re-colour',
    hint:  'Upload a colour photo. L channel is extracted and re-coloured. Metrics vs original are computed.',
  },
]

// ── State ──────────────────────────────────────────────────────────────────────
const checkpoints  = ref<CheckpointInfo[]>([])
const loadingMeta  = ref(true)
const loading      = ref(false)
const error        = ref<string | null>(null)
const selectedFile = ref<File | null>(null)
const result       = ref<ColorizeResult | null>(null)

const config = reactive({
  model:      'unet' as ModelType,
  checkpoint: '',
  mode:       'grayscale' as ColorizeMode,
})

const toast = useToast()

// ── Derived ────────────────────────────────────────────────────────────────────
const canColorize = computed(
  () => selectedFile.value !== null && config.checkpoint !== '',
)

/**
 * Build the panels array for ImageCompare based on mode.
 *
 * Grayscale mode:  [Input (gray)]          [Colorized]
 * Colour mode:     [Original (colour)] [Extracted (gray)] [Colorized]
 */
const resultPanels = computed(() => {
  if (!result.value) return []
  const r = result.value

  if (config.mode === 'grayscale') {
    return [
      { label: 'Input (grayscale)', src: `data:image/png;base64,${r.grayscale}` },
      {
        label: 'Colorized',
        src:   `data:image/png;base64,${r.colorized}`,
        downloadName: `colorized_${selectedFile.value?.name ?? 'result.png'}`,
      },
    ]
  }

  return [
    { label: 'Original colour',    src: `data:image/png;base64,${r.original}` },
    { label: 'Extracted grayscale', src: `data:image/png;base64,${r.grayscale}` },
    {
      label: 'Re-coloured',
      src:   `data:image/png;base64,${r.colorized}`,
      downloadName: `recolored_${selectedFile.value?.name ?? 'result.png'}`,
    },
  ]
})

// ── Lifecycle ──────────────────────────────────────────────────────────────────
onMounted(async () => {
  try {
    checkpoints.value = await modelsApi.listCheckpoints()
  } finally {
    loadingMeta.value = false
  }
})

// ── Handlers ───────────────────────────────────────────────────────────────────
function onFileSelected(file: File) {
  selectedFile.value = file
  result.value = null
  error.value  = null
}

function onFileCleared() {
  selectedFile.value = null
  result.value = null
  error.value  = null
}

async function runColorize() {
  if (!selectedFile.value) return
  loading.value = true
  error.value   = null
  result.value  = null

  try {
    result.value = await inferenceApi.colorize(
      selectedFile.value,
      config.model,
      config.checkpoint,
      config.mode,
    )
    toast.success('Colorization complete!')
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Colorization failed.'
    error.value = msg
    toast.error(msg)
  } finally {
    loading.value = false
  }
}

function downloadResult() {
  if (!result.value) return
  const a       = document.createElement('a')
  a.href        = `data:image/png;base64,${result.value.colorized}`
  a.download    = `colorized_${selectedFile.value?.name ?? 'result.png'}`
  a.click()
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function formatSize(bytes: number): string {
  if (bytes < 1024)        return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
</script>

<style scoped>
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from,  .fade-leave-to      { opacity: 0; }

.slide-enter-active, .slide-leave-active { transition: opacity 0.25s ease, transform 0.25s ease; }
.slide-enter-from,   .slide-leave-to     { opacity: 0; transform: translateY(10px); }
</style>
