/**
 * Training lifecycle composable.
 *
 * Wraps useSSE + trainingApi into a single ergonomic interface.
 *
 * Responsibilities:
 *  - Start / stop a training run via the API
 *  - Open an SSE stream for live progress
 *  - Accumulate loss history for chart rendering
 *  - Buffer the last N log lines for the terminal viewer
 *  - Expose computed reactive state (isRunning, progressPct, …)
 */
import { ref, computed, watch } from 'vue'
import { trainingApi } from '@/api/training'
import { useSSE }       from '@/composables/useSSE'
import type { TrainingParams, TrainingRun, TrainingProgress } from '@/types'

export interface LossPoint {
  epoch: number
  loss:  number
  lossD?: number
  lossG?: number
}

const MAX_LOG_LINES = 500
const LOG_TRIM_STEP = 100

export function useTraining() {
  // ── Core state ───────────────────────────────────────────────────────────────
  const runId      = ref<string | null>(null)
  const status     = ref<TrainingRun | null>(null)
  const lossHistory = ref<LossPoint[]>([])
  const logLines   = ref<string[]>([])

  // ── Async loading flags ───────────────────────────────────────────────────────
  const starting = ref(false)
  const stopping = ref(false)

  // ── SSE wiring ───────────────────────────────────────────────────────────────
  const streamUrl = computed<string | null>(() =>
    runId.value ? `/api/training/stream/${runId.value}` : null,
  )
  const { data: sseData, connected: sseConnected, error: sseError } =
    useSSE<TrainingProgress>(streamUrl)

  watch(sseData, (prog) => {
    if (!prog) return

    // ── Merge progress into status snapshot ────────────────────────────────────
    if (status.value) {
      if (prog.epoch        != null) status.value.epoch        = prog.epoch
      if (prog.total_epochs != null) status.value.total_epochs = prog.total_epochs
      if (prog.loss         != null) status.value.loss         = prog.loss
      if (prog.status)               status.value.status       = prog.status
    }

    // ── Accumulate loss history (de-duplicate by epoch) ───────────────────────
    if (prog.epoch != null && prog.loss != null) {
      const last = lossHistory.value[lossHistory.value.length - 1]
      if (!last || last.epoch !== prog.epoch) {
        lossHistory.value.push({
          epoch: prog.epoch,
          loss:  prog.loss,
          ...(prog.loss_d != null && { lossD: prog.loss_d }),
          ...(prog.loss_g != null && { lossG: prog.loss_g }),
        })
      }
    }

    // ── Buffer log lines ──────────────────────────────────────────────────────
    if (prog.line) {
      logLines.value.push(prog.line)
      if (logLines.value.length > MAX_LOG_LINES) {
        logLines.value.splice(0, LOG_TRIM_STEP)
      }
    }

    // ── On terminal status, refresh the full status object from REST ──────────
    if (prog.status && ['finished', 'failed', 'stopped'].includes(prog.status)) {
      if (runId.value) {
        trainingApi.status(runId.value)
          .then((s) => { status.value = s })
          .catch(() => { /* ignore — UI will show stale state */ })
      }
    }
  })

  // ── Actions ──────────────────────────────────────────────────────────────────
  async function start(params: TrainingParams): Promise<string> {
    starting.value = true
    lossHistory.value = []
    logLines.value = []
    status.value = null

    try {
      const { run_id } = await trainingApi.start(params)
      runId.value = run_id
      // Immediately fetch an initial status object so the UI renders without
      // waiting for the first SSE frame.
      status.value = await trainingApi.status(run_id)
      return run_id
    } finally {
      starting.value = false
    }
  }

  async function stop(): Promise<void> {
    if (!runId.value) return
    stopping.value = true
    try {
      await trainingApi.stop(runId.value)
    } finally {
      stopping.value = false
    }
  }

  // ── Computed helpers ─────────────────────────────────────────────────────────
  const isRunning = computed(() => status.value?.status === 'running')

  const progressPct = computed(() => {
    const s = status.value
    if (!s || !s.total_epochs) return 0
    return Math.min(100, Math.round((s.epoch / s.total_epochs) * 100))
  })

  const isGan = computed(() =>
    lossHistory.value.some((p) => p.lossD !== undefined),
  )

  return {
    // State
    runId,
    status,
    lossHistory,
    logLines,
    // Flags
    starting,
    stopping,
    isRunning,
    progressPct,
    isGan,
    // SSE meta
    sseConnected,
    sseError,
    // Actions
    start,
    stop,
  }
}
