import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: () => import('@/pages/DashboardPage.vue'),
    },
    {
      path: '/colorize',
      name: 'colorize',
      component: () => import('@/pages/ColorizePage.vue'),
    },
    {
      path: '/training',
      name: 'training',
      component: () => import('@/pages/TrainingPage.vue'),
    },
    {
      path: '/metrics',
      name: 'metrics',
      component: () => import('@/pages/MetricsPage.vue'),
    },
    {
      path: '/compare',
      name: 'compare',
      component: () => import('@/pages/ComparePage.vue'),
    },
    {
      path: '/history',
      name: 'history',
      component: () => import('@/pages/HistoryPage.vue'),
    },
    {
      path: '/batch',
      name: 'batch',
      component: () => import('@/pages/BatchPage.vue'),
    },
  ],
})

export default router
