<template>
  <div>
    <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-50">{{ title }}</h1>
    <p v-if="description" class="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
      {{ description }}
    </p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
  description?: string
}>()
</script>
