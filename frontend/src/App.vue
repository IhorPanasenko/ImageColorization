<template>
  <!-- Root element carries the `dark` class so Tailwind dark: variants work globally -->
  <div :class="['h-screen flex overflow-hidden transition-colors duration-200', isDark ? 'dark' : '']">

    <!-- ── Mobile sidebar backdrop ─────────────────────────────────────────── -->
    <Transition name="fade">
      <div
        v-if="sidebarOpen"
        class="fixed inset-0 z-20 bg-black/40 lg:hidden"
        @click="sidebarOpen = false"
      />
    </Transition>

    <!-- ── Sidebar ──────────────────────────────────────────────────────────── -->
    <aside
      :class="[
        'fixed lg:static inset-y-0 left-0 z-30',
        'flex flex-col w-64 shrink-0',
        'bg-white dark:bg-gray-800',
        'border-r border-gray-200 dark:border-gray-700',
        'transition-transform duration-300 ease-in-out',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
      ]"
    >
      <!-- Logo / brand -->
      <div class="h-16 flex items-center gap-2.5 px-5 border-b border-gray-200 dark:border-gray-700 shrink-0">
        <Palette class="w-6 h-6 text-brand-500" />
        <span class="text-[15px] font-bold tracking-tight text-gray-900 dark:text-gray-50 leading-none">
          ColorizeAI
        </span>
      </div>

      <!-- Navigation links -->
      <nav class="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          :class="[
            'group flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
            isActive(item.to)
              ? 'bg-brand-50 dark:bg-brand-900/30 text-brand-600 dark:text-brand-400'
              : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-gray-100',
          ]"
          @click="sidebarOpen = false"
        >
          <component :is="item.icon" class="w-5 h-5 shrink-0" />
          {{ item.label }}
        </RouterLink>
      </nav>

      <!-- Footer version badge -->
      <div class="border-t border-gray-200 dark:border-gray-700 px-4 py-3">
        <span class="text-[11px] text-gray-400 dark:text-gray-500">Ensemble v1.0</span>
      </div>
    </aside>

    <!-- ── Main column ──────────────────────────────────────────────────────── -->
    <div class="flex-1 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">

      <!-- Top header bar -->
      <header class="h-16 shrink-0 flex items-center justify-between px-4
                     bg-white dark:bg-gray-800
                     border-b border-gray-200 dark:border-gray-700 shadow-sm">

        <!-- Hamburger (mobile only) -->
        <button
          class="lg:hidden p-2 rounded-lg text-gray-500 dark:text-gray-400
                 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          @click="sidebarOpen = !sidebarOpen"
          aria-label="Toggle sidebar"
        >
          <Menu class="w-5 h-5" />
        </button>

        <!-- Page title (mobile centre) -->
        <span class="lg:hidden text-sm font-semibold text-gray-700 dark:text-gray-200 mx-auto">
          Image Colorization Ensemble
        </span>

        <!-- Right: dark-mode toggle -->
        <div class="ml-auto flex items-center gap-1">
          <button
            class="p-2 rounded-lg text-gray-500 dark:text-gray-400
                   hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
            @click="toggleDark"
          >
            <Sun v-if="isDark" class="w-5 h-5" />
            <Moon v-else class="w-5 h-5" />
          </button>
        </div>
      </header>

      <!-- Scrollable page content -->
      <main class="flex-1 overflow-y-auto">
        <RouterView />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import {
  LayoutDashboard,
  Activity,
  Wand2,
  BarChart2,
  GitCompare,
  History,
  Layers,
  Palette,
  Menu,
  Sun,
  Moon,
} from 'lucide-vue-next'

// ── Dark mode ──────────────────────────────────────────────────────────────────
const isDark = ref(false)

function applyDark(value: boolean) {
  isDark.value = value
  if (value) {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
}

function toggleDark() {
  const next = !isDark.value
  applyDark(next)
  localStorage.setItem('colorize-theme', next ? 'dark' : 'light')
}

onMounted(() => {
  const stored = localStorage.getItem('colorize-theme')
  if (stored === 'dark') {
    applyDark(true)
  } else if (stored === 'light') {
    applyDark(false)
  } else {
    // Fall back to OS preference
    applyDark(window.matchMedia('(prefers-color-scheme: dark)').matches)
  }
})

// ── Sidebar ────────────────────────────────────────────────────────────────────
const sidebarOpen = ref(false)

// ── Navigation items ───────────────────────────────────────────────────────────
const navItems = [
  { to: '/',          label: 'Dashboard', icon: LayoutDashboard },
  { to: '/training',  label: 'Training',  icon: Activity        },
  { to: '/colorize',  label: 'Colorize',  icon: Wand2           },
  { to: '/metrics',   label: 'Metrics',   icon: BarChart2       },
  { to: '/compare',   label: 'Compare',   icon: GitCompare      },
  { to: '/history',   label: 'History',   icon: History         },
  { to: '/batch',     label: 'Batch',     icon: Layers          },
]

const route = useRoute()

function isActive(path: string): boolean {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
</script>

<style>
/* Sidebar backdrop fade transition */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
