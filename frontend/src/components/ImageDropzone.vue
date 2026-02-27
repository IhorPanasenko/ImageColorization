<!--
  ImageDropzone — drag-and-drop image upload zone.

  Emits:
    file-selected(File) — when a file is chosen via drop or click
    file-cleared()      — when the user removes the current preview

  Props:
    file        — currently selected File (for controlled usage)
    accept      — file accept string (default 'image/*')
    maxSizeMb   — max file size in MB (default 32)
-->
<template>
  <div>
    <!-- Drop zone / preview container -->
    <div
      :class="[
        'relative border-2 border-dashed rounded-2xl cursor-pointer transition-colors duration-200',
        isDragging
          ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/10'
          : file
            ? 'border-brand-300 dark:border-brand-700 bg-gray-50 dark:bg-gray-800'
            : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800/50 hover:border-brand-400 hover:bg-gray-50 dark:hover:bg-gray-800',
      ]"
      @click="inputRef?.click()"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="onDrop"
    >
      <input
        ref="inputRef"
        type="file"
        :accept="accept"
        class="sr-only"
        @change="onInputChange"
      />

      <!-- Preview state -->
      <template v-if="previewUrl">
        <img
          :src="previewUrl"
          :alt="file?.name ?? 'preview'"
          class="w-full rounded-2xl object-contain pointer-events-none"
          :style="{ maxHeight: `${previewMaxHeight}px` }"
        />
        <div class="absolute inset-x-0 bottom-0 rounded-b-2xl
                    bg-gradient-to-t from-black/60 to-transparent px-3 py-2
                    flex items-center justify-between">
          <span class="text-xs text-white truncate max-w-[70%]">{{ file?.name }}</span>
          <button
            class="text-xs text-white/80 hover:text-white bg-black/30 hover:bg-black/50
                   px-2 py-0.5 rounded-md transition-colors"
            @click.stop="clearFile"
          >
            Change
          </button>
        </div>
      </template>

      <!-- Empty state -->
      <template v-else>
        <div class="flex flex-col items-center justify-center py-10 pointer-events-none px-4">
          <Upload :class="['w-10 h-10 mb-3', isDragging ? 'text-brand-500' : 'text-gray-400 dark:text-gray-500']" />
          <p class="text-sm font-medium text-gray-600 dark:text-gray-300">
            Drop an image here, or <span class="text-brand-500">browse</span>
          </p>
          <p class="text-xs text-gray-400 mt-1">{{ accept }} · max {{ maxSizeMb }} MB</p>
        </div>
      </template>
    </div>

    <!-- Validation error -->
    <p v-if="validationError" class="mt-1.5 text-xs text-red-500 dark:text-red-400">
      {{ validationError }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { Upload } from 'lucide-vue-next'

const props = withDefaults(
  defineProps<{
    file?:             File | null
    accept?:           string
    maxSizeMb?:        number
    previewMaxHeight?: number
  }>(),
  {
    accept:            'image/*',
    maxSizeMb:         32,
    previewMaxHeight:  320,
  },
)

const emit = defineEmits<{
  'file-selected': [file: File]
  'file-cleared':  []
}>()

const inputRef       = ref<HTMLInputElement | null>(null)
const isDragging     = ref(false)
const previewUrl     = ref<string | null>(null)
const validationError = ref<string | null>(null)

// Keep preview in sync with parent-controlled `file` prop
watch(
  () => props.file,
  (f) => {
    if (f) {
      buildPreview(f)
    } else {
      previewUrl.value = null
    }
  },
  { immediate: true },
)

function buildPreview(f: File) {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(f)
}

function validate(f: File): boolean {
  validationError.value = null
  const isImage = f.type.startsWith('image/')
  if (!isImage) {
    validationError.value = 'Please select an image file.'
    return false
  }
  const limitBytes = props.maxSizeMb * 1024 * 1024
  if (f.size > limitBytes) {
    validationError.value = `File is too large — max ${props.maxSizeMb} MB.`
    return false
  }
  return true
}

function onDrop(e: DragEvent) {
  isDragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f && validate(f)) {
    buildPreview(f)
    emit('file-selected', f)
  }
}

function onInputChange(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (f && validate(f)) {
    buildPreview(f)
    emit('file-selected', f)
  }
  // Reset so re-selecting same file fires change event next time
  if (inputRef.value) inputRef.value.value = ''
}

function clearFile() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = null
  validationError.value = null
  emit('file-cleared')
}
</script>
