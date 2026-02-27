<!--
  ImageCompare — side-by-side image panel with optional metrics footer.

  Each `panel` item: { label: string, src: string (base64 or URL) }
-->
<template>
  <div class="space-y-3">
    <!-- Image grid — 1/2/3 columns depending on panel count -->
    <div
      :class="[
        'grid gap-px bg-gray-200 dark:bg-gray-700 rounded-2xl overflow-hidden',
        gridCols,
      ]"
    >
      <div
        v-for="panel in panels"
        :key="panel.label"
        class="relative bg-gray-100 dark:bg-gray-800 group"
      >
        <img
          :src="panel.src"
          :alt="panel.label"
          class="w-full object-cover"
          :style="{ maxHeight: `${maxHeight}px` }"
          loading="lazy"
        />

        <!-- Label overlay -->
        <span
          class="absolute bottom-1.5 left-1.5 text-[10px] font-semibold uppercase tracking-wider
                 bg-black/60 text-white px-2 py-0.5 rounded-full pointer-events-none"
        >
          {{ panel.label }}
        </span>

        <!-- Download button (appears on hover) -->
        <button
          v-if="panel.downloadName"
          class="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity
                 bg-black/60 hover:bg-black/80 text-white rounded-full p-1"
          :title="`Download ${panel.downloadName}`"
          @click.prevent="download(panel)"
        >
          <Download class="w-3.5 h-3.5" />
        </button>
      </div>
    </div>

    <!-- Metrics row (optional) -->
    <div v-if="psnr !== undefined || ssim !== undefined" class="flex flex-wrap gap-4 text-sm">
      <div v-if="psnr !== undefined" class="flex items-baseline gap-1">
        <span class="text-gray-500 dark:text-gray-400 text-xs">PSNR</span>
        <span class="font-semibold text-gray-800 dark:text-gray-200 tabular-nums">
          {{ psnr != null ? `${psnr.toFixed(2)} dB` : '—' }}
        </span>
      </div>
      <div v-if="ssim !== undefined" class="flex items-baseline gap-1">
        <span class="text-gray-500 dark:text-gray-400 text-xs">SSIM</span>
        <span class="font-semibold text-gray-800 dark:text-gray-200 tabular-nums">
          {{ ssim != null ? ssim.toFixed(4) : '—' }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Download } from 'lucide-vue-next'

interface Panel {
  label:         string
  src:           string   // base64 data URI or URL
  downloadName?: string   // if set, shows a download button on hover
}

const props = withDefaults(
  defineProps<{
    panels:     Panel[]
    psnr?:      number | null
    ssim?:      number | null
    maxHeight?: number
  }>(),
  {
    maxHeight: 400,
  },
)

const gridCols = computed(() => {
  const n = props.panels.length
  if (n === 1) return 'grid-cols-1'
  if (n === 2) return 'grid-cols-2'
  return 'grid-cols-3'
})

function download(panel: Panel) {
  const a = document.createElement('a')
  a.href     = panel.src
  a.download = panel.downloadName ?? 'image.png'
  a.click()
}
</script>
