/**
 * Generic Server-Sent Events composable.
 *
 * Usage:
 *   const url = computed(() => runId.value ? `/api/training/stream/${runId.value}` : null)
 *   const { data, connected, error } = useSSE<TrainingProgress>(url)
 *
 * - Connects automatically when `urlRef` becomes non-null.
 * - Disconnects automatically when `urlRef` is set to null.
 * - Cleans up on component unmount.
 * - EventSource auto-reconnects on network blips; we just track state.
 */
import { ref, watch, onUnmounted, type Ref } from 'vue'

export function useSSE<T = unknown>(urlRef: Ref<string | null>) {
  const data      = ref<T | null>(null)
  const connected = ref(false)
  const error     = ref<string | null>(null)

  let es: EventSource | null = null

  function connect(url: string) {
    if (es) { es.close(); es = null }
    error.value = null

    es = new EventSource(url)

    es.onopen = () => {
      connected.value = true
      error.value = null
    }

    es.onmessage = (event: MessageEvent) => {
      try {
        data.value = JSON.parse(event.data) as T
      } catch {
        // non-JSON frames are intentionally ignored
      }
    }

    es.onerror = () => {
      connected.value = false
      error.value = 'SSE connection lost — auto-retrying…'
      // The browser will automatically retry; we do NOT close() here so that
      // the EventSource can re-establish the connection on its own.
    }
  }

  function disconnect() {
    es?.close()
    es = null
    connected.value = false
  }

  watch(
    urlRef,
    (url) => {
      if (url) connect(url)
      else disconnect()
    },
    { immediate: true },
  )

  onUnmounted(disconnect)

  return { data, connected, error, disconnect }
}
