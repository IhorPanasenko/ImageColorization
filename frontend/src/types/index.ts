// ─── Training ────────────────────────────────────────────────────────────────

export type ModelType =
  | 'baseline'
  | 'unet'
  | 'gan'
  | 'fusion'
  | 'classical_welsh'
  | 'classical_levin'

/** Classical method IDs — these require a reference image, not a checkpoint. */
export const CLASSICAL_MODEL_IDS: ReadonlySet<ModelType> = new Set<ModelType>([
  'classical_welsh',
  'classical_levin',
])

export interface TrainingParams {
  model: ModelType
  epochs: number
  batch_size: number
  lr: number
  lambda_l1?: number
  data_path: string
  resume_g?: string
  resume_d?: string
}

export type RunStatus = 'running' | 'finished' | 'failed' | 'stopped'

/** Shape of each SSE event emitted by GET /api/training/stream/<run_id> */
export interface TrainingProgress {
  epoch: number | null
  total_epochs: number | null
  loss: number | null
  loss_d?: number | null   // GAN discriminator loss
  loss_g?: number | null   // GAN generator loss
  lr?: number | null
  status?: RunStatus
  line?: string            // raw log line for the terminal viewer
}

export interface TrainingRun {
  run_id: string
  model: ModelType
  params: TrainingParams
  status: RunStatus
  epoch: number
  total_epochs: number
  loss: number | null
  started_at: number
  finished_at: number | null
  log_tail?: string[]
}

// ─── Inference ───────────────────────────────────────────────────────────────

export type ColorizeMode = 'grayscale' | 'color_photo'

export interface ColorizeResult {
  colorized: string    // base64 PNG
  grayscale: string    // base64 PNG
  original: string     // base64 PNG
  ground_truth?: string // base64 PNG (only present in color_photo mode)
  metrics: {
    psnr: number | null
    ssim: number | null
    lpips?: number | null
  }
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

export interface ImageMetrics {
  filename: string
  psnr: number | null
  ssim: number | null
  error?: string
}

export interface EvalResult {
  model: ModelType
  checkpoint: string
  per_image: ImageMetrics[]
  avg_psnr: number | null
  avg_ssim: number | null
  num_images: number
}

// ─── Checkpoints / Models ────────────────────────────────────────────────────

export interface CheckpointInfo {
  path: string
  filename: string
  size_mb: number
  model_hint?: string
}

export interface ModelInfo {
  id: ModelType
  name: string
  description: string
  /** True when the algorithm is classical and needs a reference image instead of a checkpoint. */
  classical?: boolean
}
