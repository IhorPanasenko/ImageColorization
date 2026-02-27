<!--
  ModelSelector — reusable model-type + checkpoint dropdown pair.

  Emits v-model-compatible events:
    update:model       → e.g. 'unet'
    update:checkpoint  → e.g. '/outputs/checkpoints/unet_final.pth'

  Props:
    model          — current ModelType value
    checkpoint     — current checkpoint path ('') 
    checkpoints    — full list from /api/models/checkpoints
    loading        — disables both selects while parent fetches data
    includeNone    — whether to show "— none —" in checkpoint dropdown (default true)
-->
<template>
  <div :class="['grid gap-4', horizontal ? 'grid-cols-2' : 'grid-cols-1']">
    <!-- Model type -->
    <div>
      <label class="label">{{ modelLabel }}</label>
      <select
        :value="model"
        class="select"
        :disabled="loading"
        @change="$emit('update:model', ($event.target as HTMLSelectElement).value as ModelType)"
      >
        <option value="baseline">Baseline CNN</option>
        <option value="unet">U-Net</option>
        <option value="gan">Pix2Pix GAN</option>
        <option value="fusion">Fusion GAN</option>
      </select>
    </div>

    <!-- Checkpoint -->
    <div>
      <label class="label">{{ checkpointLabel }}</label>
      <select
        :value="checkpoint"
        class="select"
        :disabled="loading || filteredCheckpoints.length === 0"
        @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)"
      >
        <option value="">{{ filteredCheckpoints.length === 0 ? '— no checkpoints —' : '— select —' }}</option>
        <option
          v-for="ck in filteredCheckpoints"
          :key="ck.path"
          :value="ck.path"
        >
          {{ ck.filename }} ({{ ck.size_mb }} MB)
        </option>
      </select>
      <p v-if="filteredCheckpoints.length === 0 && !loading" class="mt-1 text-xs text-amber-600 dark:text-amber-400">
        No checkpoints found for this model — train one first.
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ModelType, CheckpointInfo } from '@/types'

const props = withDefaults(
  defineProps<{
    model:             ModelType
    checkpoint:        string
    checkpoints:       CheckpointInfo[]
    loading?:          boolean
    horizontal?:       boolean
    filterByModel?:    boolean
    modelLabel?:       string
    checkpointLabel?:  string
  }>(),
  {
    loading:           false,
    horizontal:        true,
    filterByModel:     true,
    modelLabel:        'Model',
    checkpointLabel:   'Checkpoint',
  },
)

defineEmits<{
  'update:model':      [v: ModelType]
  'update:checkpoint': [v: string]
}>()

/** Filter checkpoints to those whose filename hints at the selected model. */
const filteredCheckpoints = computed<CheckpointInfo[]>(() => {
  if (!props.filterByModel) return props.checkpoints
  const key = props.model.toLowerCase()
  return props.checkpoints.filter((ck) => {
    const f = ck.filename.toLowerCase()
    switch (key) {
      case 'baseline': return f.startsWith('baseline')
      case 'unet':     return f.startsWith('unet') || f.startsWith('u_net')
      case 'gan':      return f.startsWith('gan') || f.startsWith('disc') || f.startsWith('pix2pix')
      case 'fusion':   return f.startsWith('fusion')
      default:         return true
    }
  })
})
</script>
