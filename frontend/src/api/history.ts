import api from './client'

export interface ParsedLog {
  run_id: string
  epochs: number[]
  losses: number[]
  lrs: number[]
  lines: string[]
}

export interface TbScalar {
  step: number
  value: number
}

export interface TensorboardData {
  model_type: string
  tags: Record<string, TbScalar[]>
  warning?: string
  error?: string
}

export const historyApi = {
  /** List all historical runs (same data as training/runs, via history scope). */
  listRuns: () =>
    api.get('/history/runs').then((r) => r.data),

  /** Return parsed epoch/loss/lr arrays + last 200 log lines for one run. */
  getLogs: (runId: string): Promise<ParsedLog> =>
    api.get<ParsedLog>(`/history/logs/${runId}`).then((r) => r.data),

  /** Return TensorBoard scalar data for a model_type (e.g. "unet", "gan"). */
  getTensorboardData: (modelType: string): Promise<TensorboardData> =>
    api.get<TensorboardData>(`/history/tensorboard-data/${modelType}`).then((r) => r.data),

  /** Delete a run record from runs.json. */
  deleteRun: (runId: string) =>
    api.delete(`/history/${runId}`).then((r) => r.data),
}
