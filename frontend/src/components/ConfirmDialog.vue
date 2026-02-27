<template>
  <Teleport to="body">
    <Transition name="dialog-fade">
      <div
        v-if="open"
        class="fixed inset-0 z-50 flex items-center justify-center p-4"
        role="dialog"
        aria-modal="true"
        :aria-labelledby="titleId"
      >
        <!-- Backdrop -->
        <div
          class="absolute inset-0 bg-black/50 backdrop-blur-sm"
          @click="$emit('cancel')"
        />

        <!-- Panel -->
        <div
          class="relative bg-white dark:bg-gray-800 rounded-2xl shadow-2xl
                 p-6 w-full max-w-sm border border-gray-200 dark:border-gray-700"
        >
          <!-- Icon slot (optional) -->
          <div v-if="$slots.icon" class="mb-4 flex justify-center">
            <slot name="icon" />
          </div>

          <h3
            :id="titleId"
            class="text-base font-semibold text-gray-900 dark:text-gray-50"
          >
            {{ title }}
          </h3>

          <p v-if="message" class="mt-2 text-sm text-gray-500 dark:text-gray-400">
            {{ message }}
          </p>

          <div class="mt-5 flex gap-3 justify-end">
            <button
              class="btn btn-secondary"
              @click="$emit('cancel')"
            >
              {{ cancelLabel }}
            </button>
            <button
              :class="['btn', confirmClass]"
              @click="$emit('confirm')"
            >
              {{ confirmLabel }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    open: boolean
    title: string
    message?: string
    confirmLabel?: string
    cancelLabel?: string
    variant?: 'danger' | 'primary'
  }>(),
  {
    confirmLabel: 'Confirm',
    cancelLabel:  'Cancel',
    variant:      'primary',
  },
)

defineEmits<{
  confirm: []
  cancel:  []
}>()

const titleId = computed(() => `dialog-title-${Math.random().toString(36).slice(2)}`)

const confirmClass = computed(() =>
  props.variant === 'danger' ? 'btn-danger' : 'btn-primary',
)
</script>

<style scoped>
.dialog-fade-enter-active,
.dialog-fade-leave-active {
  transition: opacity 0.15s ease;
}
.dialog-fade-enter-from,
.dialog-fade-leave-to {
  opacity: 0;
}
</style>
