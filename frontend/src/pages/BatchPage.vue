<template>
  <div class="p-6 space-y-6">
    <!-- ── Page header ──────────────────────────────────────────────────────── -->
    <PageHeader
      title="Batch Colorize"
      description="Upload multiple images and colorize them all at once."
    />

    <!-- ── Configuration panel ──────────────────────────────────────────────── -->
    <!-- Skeleton while loading config -->
    <div v-if="initLoading" class="card space-y-5">
      <div class="h-4 w-20 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div v-for="i in 3" :key="i" class="space-y-2">
          <div class="h-3 w-16 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
          <div class="h-9 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
        </div>
      </div>
    </div>

    <div v-else class="card space-y-5">
      <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Settings</h2>

      <!-- No checkpoints warning -->
      <div
        v-if="checkpoints.length === 0"
        class="rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800
               px-4 py-3 text-sm text-amber-700 dark:text-amber-400 flex items-center gap-2"
      >
        <AlertCircle class="w-4 h-4 shrink-0" />
        No checkpoints found. <RouterLink to="/training" class="underline font-medium ml-1">Train a model first &rarr;</RouterLink>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <!-- Model -->
        <div>
          <label class="label">Model</label>
          <select v-model="selectedModel" class="select" :disabled="processing">
            <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name }}</option>
          </select>
        </div>

        <!-- Checkpoint -->
        <div>
          <label class="label">Checkpoint</label>
          <select v-model="selectedCheckpoint" class="select" :disabled="processing">
            <option value="">— select —</option>
            <option
              v-for="ck in filteredCheckpoints"
              :key="ck.path"
              :value="ck.path"
            >
              {{ ck.filename }} ({{ ck.size_mb }} MB)
            </option>
          </select>
        </div>

        <!-- Mode -->
        <div>
          <label class="label">Input Mode</label>
          <select v-model="selectedMode" class="select" :disabled="processing">
            <option value="grayscale">Grayscale image</option>
            <option value="color_photo">Colour photo (auto-converted)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- ── Drop zone ────────────────────────────────────────────────────────── -->
    <div
      :class="[
        'relative border-2 border-dashed rounded-2xl transition-colors duration-200 cursor-pointer',
        isDragging
          ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/10'
          : 'border-gray-300 dark:border-gray-600 hover:border-brand-400 hover:bg-gray-50 dark:hover:bg-gray-800/50',
      ]"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="onDrop"
      @click="fileInputRef?.click()"
    >
      <input
        ref="fileInputRef"
        type="file"
        multiple
        accept="image/*"
        class="sr-only"
        @change="onFileInput"
      />

      <div class="flex flex-col items-center justify-center py-12 pointer-events-none">
        <Upload :class="['w-10 h-10 mb-3', isDragging ? 'text-brand-500' : 'text-gray-400 dark:text-gray-500']" />
        <p class="text-sm font-medium text-gray-600 dark:text-gray-300">
          Drop images here, or <span class="text-brand-500">browse</span>
        </p>
        <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">PNG, JPG, JPEG, WebP — any number of files</p>
      </div>
    </div>

    <!-- ── Selected files list ──────────────────────────────────────────────── -->
    <div v-if="selectedFiles.length > 0" class="card">
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">
          {{ selectedFiles.length }} file{{ selectedFiles.length !== 1 ? 's' : '' }} selected
        </h2>
        <button
          class="text-xs text-red-500 hover:text-red-600 transition-colors"
          @click="clearFiles"
          :disabled="processing"
        >
          Clear all
        </button>
      </div>

      <ul class="space-y-1 max-h-48 overflow-y-auto pr-1">
        <li
          v-for="(file, idx) in selectedFiles"
          :key="idx"
          class="flex items-center justify-between text-sm py-1 gap-2"
        >
          <div class="flex items-center gap-2 min-w-0">
            <ImageIcon class="w-4 h-4 text-gray-400 dark:text-gray-500 shrink-0" />
            <span class="truncate text-gray-700 dark:text-gray-300">{{ file.name }}</span>
          </div>
          <span class="text-xs text-gray-400 dark:text-gray-500 shrink-0">{{ formatSize(file.size) }}</span>
        </li>
      </ul>
    </div>

    <!-- ── Action bar ────────────────────────────────────────────────────────── -->
    <div class="flex items-center gap-4">
      <button
        class="btn btn-primary flex items-center gap-2"
        :disabled="!canProcess || processing"
        @click="processAll"
      >
        <Loader2 v-if="processing" class="w-4 h-4 animate-spin" />
        <Zap v-else class="w-4 h-4" />
        {{ processing ? `Processing ${doneCount}/${selectedFiles.length}…` : 'Process All' }}
      </button>

      <button
        v-if="results.length > 0"
        class="btn btn-secondary flex items-center gap-2"
        @click="downloadAll"
      >
        <Download class="w-4 h-4" />
        Download All
      </button>

      <!-- Validation hint -->
      <p v-if="!canProcess && !processing" class="text-xs text-gray-400 dark:text-gray-500">
        <template v-if="selectedFiles.length === 0">Select at least one image.</template>
        <template v-else-if="!selectedCheckpoint">Choose a checkpoint first.</template>
      </p>
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

    <!-- ── Summary metrics ──────────────────────────────────────────────────── -->
    <div v-if="results.length > 0 && avgMetrics" class="grid grid-cols-3 gap-4">
      <div class="card text-center">
        <p class="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Avg PSNR</p>
        <p class="text-2xl font-bold mt-1" :class="psnrColor(avgMetrics.psnr)">{{ avgMetrics.psnr.toFixed(2) }} dB</p>
      </div>
      <div class="card text-center">
        <p class="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Avg SSIM</p>
        <p class="text-2xl font-bold mt-1" :class="ssimColor(avgMetrics.ssim)">{{ avgMetrics.ssim.toFixed(3) }}</p>
      </div>
      <div class="card text-center">
        <p class="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide">Processed</p>
        <p class="text-2xl font-bold mt-1 text-brand-500">{{ results.length }}</p>
      </div>
    </div>

    <!-- ── Results gallery ───────────────────────────────────────────────────── -->
    <section v-if="results.length > 0">
      <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">
        Results — {{ results.length }} image{{ results.length !== 1 ? 's' : '' }}
      </h2>

      <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        <div
          v-for="(result, idx) in results"
          :key="idx"
          class="card overflow-hidden p-0"
        >
          <!-- Image comparison (grayscale | colorized) -->
          <div class="grid grid-cols-2 gap-px bg-gray-200 dark:bg-gray-700">
            <div class="relative bg-gray-100 dark:bg-gray-800">
              <img
                :src="`data:image/png;base64,${result.grayscale}`"
                class="w-full aspect-square object-cover"
                alt="Grayscale"
              />
              <span class="absolute bottom-1 left-1 text-[10px] bg-black/50 text-white px-1.5 py-0.5 rounded">
                Input
              </span>
            </div>
            <div class="relative bg-gray-100 dark:bg-gray-800">
              <img
                :src="`data:image/png;base64,${result.colorized}`"
                class="w-full aspect-square object-cover"
                alt="Colorized"
              />
              <span class="absolute bottom-1 left-1 text-[10px] bg-black/50 text-white px-1.5 py-0.5 rounded">
                Colorized
              </span>
            </div>
          </div>

          <!-- Footer row -->
          <div class="px-3 py-2 flex items-center justify-between gap-2">
            <p class="text-xs text-gray-600 dark:text-gray-400 truncate font-medium">
              {{ result.filename }}
            </p>
            <div class="flex items-center gap-2 shrink-0">
              <span v-if="result.metrics.psnr != null" class="text-xs text-gray-500 dark:text-gray-400">
                PSNR {{ result.metrics.psnr.toFixed(1) }}
              </span>
              <button
                class="text-brand-500 hover:text-brand-600 transition-colors"
                :title="`Download ${result.filename}`"
                @click="downloadSingle(result)"
              >
                <Download class="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { Upload, ImageIcon, Zap, Loader2, Download, AlertCircle } from 'lucide-vue-next'
import { useToast } from 'vue-toastification'
import PageHeader from '@/components/PageHeader.vue'
import { modelsApi } from '@/api/models'
import { inferenceApi } from '@/api/inference'
import type { ModelType, ColorizeMode, ColorizeResult, ModelInfo, CheckpointInfo } from '@/types'

// ── Types ──────────────────────────────────────────────────────────────────────
type BatchResult = ColorizeResult & { filename: string }

interface AvgMetrics { psnr: number; ssim: number }

// ── Refs ───────────────────────────────────────────────────────────────────────
const fileInputRef     = ref<HTMLInputElement | null>(null)
const isDragging       = ref(false)
const selectedFiles    = ref<File[]>([])
const selectedModel    = ref<ModelType>('unet')
const selectedCheckpoint = ref('')
const selectedMode     = ref<ColorizeMode>('grayscale')
const processing       = ref(false)
const initLoading      = ref(true)
const doneCount        = ref(0)
const error            = ref<string | null>(null)
const results          = ref<BatchResult[]>([])

const models      = ref<ModelInfo[]>([])
const checkpoints = ref<CheckpointInfo[]>([])

const toast = useToast()

// ── Lifecycle ──────────────────────────────────────────────────────────────────
onMounted(async () => {
  try {
    const [modelList, ckptList] = await Promise.allSettled([
      modelsApi.listModels(),
      modelsApi.listCheckpoints(),
    ])
    if (modelList.status === 'fulfilled') models.value = modelList.value
    if (ckptList.status  === 'fulfilled') checkpoints.value = ckptList.value
  } finally {
    initLoading.value = false
  }
})

// ── Computed ───────────────────────────────────────────────────────────────────
const filteredCheckpoints = computed(() =>
  checkpoints.value.filter(c => !c.model_hint || c.model_hint === selectedModel.value),
)

const canProcess = computed(
  () => selectedFiles.value.length > 0 && selectedCheckpoint.value !== '',
)

const avgMetrics = computed((): AvgMetrics | null => {
  const withPsnr = results.value.filter(r => r.metrics?.psnr != null)
  const withSsim = results.value.filter(r => r.metrics?.ssim != null)
  if (!withPsnr.length && !withSsim.length) return null
  return {
    psnr: withPsnr.length ? withPsnr.reduce((s, r) => s + r.metrics!.psnr!, 0) / withPsnr.length : 0,
    ssim: withSsim.length ? withSsim.reduce((s, r) => s + r.metrics!.ssim!, 0) / withSsim.length : 0,
  }
})

// ── File handling ──────────────────────────────────────────────────────────────
function onDrop(event: DragEvent) {
  isDragging.value = false
  const files = Array.from(event.dataTransfer?.files ?? []).filter((f) =>
    f.type.startsWith('image/'),
  )
  selectedFiles.value = [...selectedFiles.value, ...files]
}

function onFileInput(event: Event) {
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files ?? [])
  selectedFiles.value = [...selectedFiles.value, ...files]
  // Reset to allow re-selecting same files
  input.value = ''
}

function clearFiles() {
  selectedFiles.value = []
  results.value = []
  error.value = null
  doneCount.value = 0
}

// ── Processing ─────────────────────────────────────────────────────────────────
async function processAll() {
  if (!canProcess.value) return
  processing.value = true
  error.value = null
  results.value = []
  doneCount.value = 0

  try {
    // Process in chunks of 4 to avoid overwhelming the server
    const CHUNK = 4
    const allFiles = selectedFiles.value
    const newResults: BatchResult[] = []

    for (let i = 0; i < allFiles.length; i += CHUNK) {
      const chunk = allFiles.slice(i, i + CHUNK)
      const chunkResults = await inferenceApi.colorizeBatch(
        chunk,
        selectedModel.value,
        selectedCheckpoint.value,
        selectedMode.value,
      )
      newResults.push(...chunkResults)
      doneCount.value += chunk.length
    }

    results.value = newResults
    toast.success(`Batch complete — ${newResults.length} image${newResults.length !== 1 ? 's' : ''} colorized.`)
  } catch (err: unknown) {
    error.value = err instanceof Error ? err.message : 'Batch processing failed.'
    toast.error(error.value ?? 'Batch processing failed.')
  } finally {
    processing.value = false
  }
}

// ── Download helpers ───────────────────────────────────────────────────────────
function downloadSingle(result: BatchResult) {
  const a = document.createElement('a')
  a.href = `data:image/png;base64,${result.colorized}`
  a.download = `colorized_${result.filename}`
  a.click()
}

function downloadAll() {
  for (const result of results.value) {
    downloadSingle(result)
  }
}

// ── Formatters ─────────────────────────────────────────────────────────────────
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function psnrColor(v: number): string {
  if (v >= 25) return 'text-green-600 dark:text-green-400'
  if (v >= 18) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

function ssimColor(v: number): string {
  if (v >= 0.8) return 'text-green-600 dark:text-green-400'
  if (v >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}
</script>
