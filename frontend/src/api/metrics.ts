import api from './client'
import type { ModelType, EvalResult, ColorizeResult } from '@/types'

export const metricsApi = {
  evaluateSingle: (imagePath: string, model: ModelType, checkpoint: string) =>
    api.post<{ model: string; checkpoint: string; metrics: { psnr: number | null; ssim: number | null } }>(
      '/metrics/evaluate',
      { image_path: imagePath, model, checkpoint },
    ).then((r) => r.data),

  batchEvaluate: (model: ModelType, checkpoint: string) =>
    api.post<EvalResult>(
      '/metrics/batch_evaluate',
      { model, checkpoint },
    ).then((r) => r.data),

  compareModels: (
    imagePath: string,
    models: { model: ModelType; checkpoint: string; label: string }[],
  ) =>
    api.post<(ColorizeResult & { label: string })[]>(
      '/metrics/compare',
      { image_path: imagePath, models },
    ).then((r) => r.data),
}
