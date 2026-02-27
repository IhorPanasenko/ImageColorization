import api from './client'
import type { TrainingParams, TrainingRun } from '@/types'

export const trainingApi = {
  start: (params: TrainingParams) =>
    api.post<{ run_id: string }>('/training/start', params).then((r) => r.data),

  status: (runId: string) =>
    api.get<TrainingRun>(`/training/status/${runId}`).then((r) => r.data),

  stop: (runId: string) =>
    api.post(`/training/stop/${runId}`).then((r) => r.data),

  listRuns: () =>
    api.get<TrainingRun[]>('/training/runs').then((r) => r.data),

  /** Returns an EventSource for SSE streaming. */
  stream: (runId: string): EventSource =>
    new EventSource(`/api/training/stream/${runId}`),
}
