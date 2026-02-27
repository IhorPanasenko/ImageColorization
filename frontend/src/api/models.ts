import api from './client'
import type { ModelInfo, CheckpointInfo } from '@/types'

export const modelsApi = {
  listModels: () =>
    api.get<ModelInfo[]>('/models').then((r) => r.data),

  listCheckpoints: () =>
    api.get<CheckpointInfo[]>('/models/checkpoints').then((r) => r.data),
}
