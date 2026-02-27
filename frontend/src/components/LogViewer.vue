<template>
  <div class="relative">
    <!-- Toolbar -->
    <div class="flex items-center justify-between px-3 py-1.5
                bg-gray-800 dark:bg-gray-950 rounded-t-lg border-b border-gray-700">
      <span class="text-[11px] text-gray-400 font-mono tracking-wide">
        {{ lines.length }} line{{ lines.length !== 1 ? 's' : '' }}
      </span>
      <button
        v-if="lines.length > 0"
        class="text-[11px] text-gray-500 hover:text-gray-300 transition-colors"
        @click="$emit('clear')"
        title="Clear log"
      >
        clear
      </button>
    </div>

    <!-- Log output -->
    <pre
      ref="containerRef"
      :class="[
        'bg-gray-900 dark:bg-gray-950 text-green-400 rounded-b-lg p-3',
        'text-[11px] leading-5 font-mono overflow-auto whitespace-pre-wrap break-all',
      ]"
      :style="{ maxHeight: `${maxHeight}px` }"
    >{{ displayText }}</pre>

    <!-- Empty state -->
    <div
      v-if="lines.length === 0"
      class="absolute inset-0 top-8 flex items-center justify-center
             text-xs text-gray-600 font-mono pointer-events-none"
    >
      No output yet…
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'

const props = withDefaults(
  defineProps<{
    lines: string[]
    maxHeight?: number
    autoScroll?: boolean
  }>(),
  {
    maxHeight: 300,
    autoScroll: true,
  },
)

defineEmits<{ clear: [] }>()

const containerRef = ref<HTMLPreElement | null>(null)

const displayText = computed(() => props.lines.join('\n'))

watch(
  () => props.lines.length,
  async () => {
    if (!props.autoScroll) return
    await nextTick()
    if (containerRef.value) {
      containerRef.value.scrollTop = containerRef.value.scrollHeight
    }
  },
)
</script>
