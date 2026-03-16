# Plan: Metrics Benchmark & Enhanced Visual Comparison

## TL;DR
Add a **Benchmark tab** to the existing `/metrics` page that runs all 6 models (4 neural + 2 classical) on the server-side test set, then displays results with grouped bar charts, ranking tables, and radar. Enhance the existing `/compare` page with LPIPS + inference_time on radar axes, plus a colour-coded ranking summary table. Requires one new backend endpoint and `inference_time_ms` support for classical algorithms.

---

## Progress Tracker

| Step | Phase | Description | Status |
|------|-------|-------------|--------|
| 1 | Backend | Add `inference_time_ms` to classical colorizer | ✅ DONE |
| 2 | Backend | Add `POST /api/metrics/benchmark` endpoint | ✅ DONE |
| 3 | Backend | Add `benchmark()` to frontend API layer | ✅ DONE |
| 4 | Types | Add TypeScript types (`BenchmarkModelConfig`, etc.) | ✅ DONE |
| 5 | Frontend | Add tab UI (Single Model / All Models Benchmark) to MetricsPage | ✅ DONE |
| 6 | Frontend | Benchmark tab — config section (auto-populated model slots) | ✅ DONE |
| 7 | Frontend | Benchmark tab — 4 grouped bar charts (PSNR, SSIM, LPIPS, Time) | ✅ DONE |
| 8 | Frontend | Benchmark tab — colour-coded ranking table with gold/silver/bronze | ✅ DONE |
| 9 | Frontend | Benchmark tab — per-image drill-down (expandable, dropdown) | ✅ DONE |
| 10 | Frontend | Extend RadarChart axes (add LPIPS + Speed) | ✅ DONE |
| 11 | Frontend | Pass LPIPS + inference_time_ms to radar from ComparePage | ✅ DONE |
| 12 | Frontend | Add ranking table to ComparePage below radar | ✅ DONE |
| 13 | Component | Create `RankingTable.vue` shared component | ✅ DONE |
| 14 | Component | Create `BenchmarkBarChart.vue` shared component | ✅ DONE |

> Update status: ⬜ TODO → 🔄 IN PROGRESS → ✅ DONE

---

## Phase 1: Backend — New Bulk Benchmark Endpoint + Classical Timing

### Step 1 — Add `inference_time_ms` to classical colorizer *(backend fix)*

**File:** `api/services/classical_colorizer.py` (lines 100–120)

**What to do:**
- Import `time_inference` from `ml/src/utils/metrics.py` (same as neural colorizer does)
- Wrap the `colorize_welsh()` / `colorize_levin()` call with `time_inference()`:
  ```python
  from utils.metrics import time_inference, compute_psnr, compute_ssim, compute_lpips

  def _run():
      if method == 'welsh':
          return colorize_welsh(target_path, reference_path, ...)
      return colorize_levin(target_path, reference_path, ...)

  pred_rgb, elapsed_ms = time_inference(_run, device='cpu')
  ```
- Add `'inference_time_ms': round(elapsed_ms, 2)` to the returned `metrics` dict

**Before (current `result` dict):**
```python
result: dict[str, Any] = {
    'colorized': ...,
    'grayscale': ...,
    'original':  ...,
    'metrics':   {'psnr': None, 'ssim': None, 'lpips': None},
}
```

**After:**
```python
result: dict[str, Any] = {
    'colorized': ...,
    'grayscale': ...,
    'original':  ...,
    'metrics':   {'psnr': None, 'ssim': None, 'lpips': None,
                  'inference_time_ms': round(elapsed_ms, 2)},
}
```

---

### Step 2 — Add `POST /api/metrics/benchmark` endpoint *(new endpoint)*

**Files:**
- `api/services/metrics_service.py` — add `benchmark()` method
- `api/routes/metrics.py` — add route

**Request body:**
```json
{
  "models": [
    {"model": "baseline",        "checkpoint": "outputs/checkpoints/baseline_cnn_best.pth", "label": "Baseline CNN"},
    {"model": "unet",            "checkpoint": "outputs/checkpoints/unet_best.pth",         "label": "U-Net"},
    {"model": "gan",             "checkpoint": "outputs/checkpoints/gan_generator_best.pth","label": "GAN"},
    {"model": "fusion",          "checkpoint": "outputs/checkpoints/gan_generator_best.pth","label": "Fusion"},
    {"model": "classical_welsh", "checkpoint": "",                                           "label": "Welsh 2002"},
    {"model": "classical_levin", "checkpoint": "",                                           "label": "Levin 2004"}
  ],
  "max_images": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "label": "U-Net",
      "model": "unet",
      "per_image": [
        {"filename": "img1.jpg", "psnr": 27.3, "ssim": 0.82, "lpips": 0.14, "inference_time_ms": 42.1}
      ],
      "avg_psnr": 27.3,
      "avg_ssim": 0.82,
      "avg_lpips": 0.14,
      "avg_inference_time_ms": 42.1,
      "num_images": 1
    }
  ]
}
```

**Implementation logic for `MetricsService.benchmark()`:**
```python
def benchmark(self, model_configs, sample_dir, max_images=None):
    CLASSICAL_IDS = {'classical_welsh', 'classical_levin'}
    
    # Collect test images
    images = sorted(glob.glob(os.path.join(sample_dir, '*.jpg')) + ...)
    if max_images:
        images = images[:max_images]
    
    # For classical: reference = first image; evaluate on rest
    reference_path = images[0] if images else None
    
    results = []
    for cfg in model_configs:
        model_id = cfg['model']
        is_classical = model_id in CLASSICAL_IDS
        eval_images = images[1:] if is_classical else images  # skip ref for classical
        
        per_image = []
        for img_path in eval_images:
            try:
                if is_classical:
                    r = self._classical_colorizer.colorize(
                        img_path, reference_path, method=..., mode='color_photo'
                    )
                else:
                    r = self._colorizer.colorize(
                        img_path, model_id, cfg['checkpoint'], mode='color_photo'
                    )
                metrics = r.get('metrics', {})
                per_image.append({
                    'filename': os.path.basename(img_path),
                    'psnr': metrics.get('psnr'),
                    'ssim': metrics.get('ssim'),
                    'lpips': metrics.get('lpips'),
                    'inference_time_ms': metrics.get('inference_time_ms'),
                })
            except Exception as e:
                per_image.append({'filename': os.path.basename(img_path), 'error': str(e)})
        
        # Average over valid results
        valid = [r for r in per_image if 'error' not in r]
        results.append({
            'label': cfg.get('label', model_id),
            'model': model_id,
            'per_image': per_image,
            'avg_psnr': mean([r['psnr'] for r in valid if r['psnr']]) or None,
            'avg_ssim': mean([r['ssim'] for r in valid if r['ssim']]) or None,
            'avg_lpips': mean([r['lpips'] for r in valid if r['lpips']]) or None,
            'avg_inference_time_ms': mean([r['inference_time_ms'] for r in valid if r['inference_time_ms']]) or None,
            'num_images': len(valid),
        })
    
    return {'results': results}
```

**Note:** `MetricsService.__init__` needs a `_classical_colorizer` attribute — add:
```python
from api.services.classical_colorizer import ClassicalColorizer

CLASSICAL_METHOD_MAP = {'classical_welsh': 'welsh', 'classical_levin': 'levin'}

def __init__(self):
    self._colorizer = Colorizer()
    self._classical_colorizer = ClassicalColorizer()
```

---

### Step 3 — Add `benchmark()` to frontend API layer

**File:** `frontend/src/api/metrics.ts`

```typescript
benchmark: (
  models: import('@/types').BenchmarkModelConfig[],
  maxImages?: number,
) =>
  api.post<import('@/types').BenchmarkResult>(
    '/metrics/benchmark',
    { models, max_images: maxImages },
    { timeout: 600_000 },  // 10 min — running 6 models × N images
  ).then((r) => r.data),
```

---

### Step 4 — Add TypeScript types

**File:** `frontend/src/types/index.ts`

Add after the `EvalResult` interface:

```typescript
// ─── Benchmark ───────────────────────────────────────────────────────────────

export interface BenchmarkModelConfig {
  model: ModelType
  checkpoint: string
  label: string
}

export interface BenchmarkModelResult {
  label: string
  model: ModelType
  per_image: ImageMetrics[]
  avg_psnr: number | null
  avg_ssim: number | null
  avg_lpips: number | null
  avg_inference_time_ms: number | null
  num_images: number
}

export interface BenchmarkResult {
  results: BenchmarkModelResult[]
}
```

Also update `ImageMetrics` to add `lpips`:
```typescript
export interface ImageMetrics {
  filename: string
  psnr: number | null
  ssim: number | null
  lpips?: number | null          // ← add this
  inference_time_ms?: number | null
  error?: string
}
```

---

## Phase 2: Metrics Page — Add Benchmark Tab

### Step 5 — Add tab UI to MetricsPage

**File:** `frontend/src/pages/MetricsPage.vue`

Add a tab switcher at the top of the page (before the config card):

```html
<!-- Tab switcher -->
<div class="flex gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-xl w-fit">
  <button
    :class="['px-4 py-1.5 rounded-lg text-sm font-medium transition-colors',
      activeTab === 'single'
        ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
        : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300']"
    @click="activeTab = 'single'"
  >
    Single Model
  </button>
  <button
    :class="['px-4 py-1.5 rounded-lg text-sm font-medium transition-colors',
      activeTab === 'benchmark'
        ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
        : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300']"
    @click="activeTab = 'benchmark'"
  >
    <FlaskConical class="w-3.5 h-3.5 inline mr-1" />
    All Models Benchmark
  </button>
</div>

<!-- Single Model tab (existing content wrapped in v-if) -->
<div v-if="activeTab === 'single'">
  <!-- ...all current MetricsPage content... -->
</div>

<!-- Benchmark tab (new content) -->
<div v-else-if="activeTab === 'benchmark'">
  <!-- ...see steps 6-9... -->
</div>
```

Add `activeTab = ref<'single' | 'benchmark'>('single')` to script.

---

### Step 6 — Benchmark tab config section

Inside the benchmark tab:

```html
<div class="card space-y-4">
  <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Model Configuration</h2>
  
  <!-- Model slots — 2-column grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
    <div v-for="(slot, i) in benchmarkSlots" :key="i" class="rounded-xl border ... p-3 space-y-2">
      <!-- Toggle to include/exclude model -->
      <label class="flex items-center gap-2">
        <input type="checkbox" v-model="slot.enabled" class="rounded" />
        <span class="text-sm font-medium">{{ slot.label }}</span>
        <span class="text-xs text-gray-400">({{ slot.model }})</span>
      </label>
      
      <!-- Checkpoint selector (hidden for classical) -->
      <select
        v-if="!isClassical(slot.model)"
        v-model="slot.checkpoint"
        class="select !py-1 !text-xs w-full"
        :disabled="!slot.enabled"
      >
        <option value="">— select checkpoint —</option>
        <option v-for="c in checkpointsForModel(slot.model)" :key="c.path" :value="c.path">
          {{ c.filename }}
        </option>
      </select>
      <p v-else class="text-xs text-gray-400 dark:text-gray-500 italic">
        Uses first test image as colour reference
      </p>
    </div>
  </div>

  <!-- Max images + run button -->
  <div class="flex items-center gap-3 flex-wrap">
    <label class="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2">
      Max images:
      <input type="number" v-model.number="maxImages" min="1" max="100"
             class="input !py-0.5 !text-xs w-16" />
      <span class="text-gray-400">(blank = all)</span>
    </label>
    <button class="btn btn-primary flex items-center gap-2" :disabled="!canBenchmark || benchLoading" @click="runBenchmark">
      <Loader2 v-if="benchLoading" class="w-4 h-4 animate-spin" />
      <FlaskConical v-else class="w-4 h-4" />
      {{ benchLoading ? `Running… (${benchProgress})` : 'Run Benchmark' }}
    </button>
  </div>
</div>
```

**Script logic:**
```typescript
const benchmarkSlots = reactive<BenchmarkModelConfig & { enabled: boolean }[]>([
  { model: 'baseline',        checkpoint: '', label: 'Baseline CNN', enabled: true },
  { model: 'unet',            checkpoint: '', label: 'U-Net',        enabled: true },
  { model: 'gan',             checkpoint: '', label: 'GAN',          enabled: true },
  { model: 'fusion',          checkpoint: '', label: 'Fusion',       enabled: true },
  { model: 'classical_welsh', checkpoint: '', label: 'Welsh 2002',   enabled: true },
  { model: 'classical_levin', checkpoint: '', label: 'Levin 2004',   enabled: true },
])

// Auto-fill best checkpoint for each neural model when checkpoints load
watch(allCheckpoints, (cks) => {
  for (const slot of benchmarkSlots) {
    if (isClassical(slot.model) || slot.checkpoint) continue
    const best = cks.find(c => c.filename.includes(slot.model) && c.filename.includes('best'))
      || cks.find(c => c.model_hint === slot.model)
    if (best) slot.checkpoint = best.path
  }
})
```

---

### Step 7 — Benchmark tab: 4 grouped bar charts

**New component** `frontend/src/components/BenchmarkBarChart.vue`

**Props:**
```typescript
interface ModelBar {
  label: string
  value: number | null
  color: string
}

defineProps<{
  models: ModelBar[]
  metricName: string   // "PSNR", "SSIM", "LPIPS", "Inference Time"
  unit: string         // "dB", "", "", "ms"
  lowerIsBetter?: boolean
}>()
```

**Template** — renders a single `<Bar>` with:
- One dataset (one bar per model)
- Each bar with its `color`
- Y-axis label = `metricName (unit)`
- Tooltip: value + unit

**In benchmark tab — 2×2 grid:**
```html
<div v-if="benchResult" class="grid grid-cols-1 md:grid-cols-2 gap-5">
  <div class="card">
    <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-3">PSNR (higher is better)</h3>
    <BenchmarkBarChart :models="barDataFor('avg_psnr')" metric-name="PSNR" unit="dB" />
  </div>
  <div class="card">
    <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-3">SSIM (higher is better)</h3>
    <BenchmarkBarChart :models="barDataFor('avg_ssim')" metric-name="SSIM" unit="" />
  </div>
  <div class="card">
    <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-3">LPIPS (lower is better)</h3>
    <BenchmarkBarChart :models="barDataFor('avg_lpips')" metric-name="LPIPS" unit="" :lower-is-better="true" />
  </div>
  <div class="card">
    <h3 class="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-3">Inference Time (lower is better)</h3>
    <BenchmarkBarChart :models="barDataFor('avg_inference_time_ms')" metric-name="Time" unit="ms" :lower-is-better="true" />
  </div>
</div>
```

---

### Step 8 — Benchmark tab: ranking table

**New component** `frontend/src/components/RankingTable.vue`

**Props:**
```typescript
export interface RankingRow {
  label: string
  model: string
  color?: string
  psnr: number | null
  ssim: number | null
  lpips: number | null
  inferenceTimeMs: number | null
}

defineProps<{
  rows: RankingRow[]
}>()
```

**Rank formula** (computed internally):
```typescript
function computeScore(row: RankingRow, maxPsnr: number, maxTime: number): number {
  const psnrNorm  = row.psnr   !== null ? Math.min(row.psnr / maxPsnr, 1)       : 0
  const ssimNorm  = row.ssim   !== null ? Math.max(0, Math.min(row.ssim, 1))     : 0
  const lpipsNorm = row.lpips  !== null ? Math.max(0, 1 - row.lpips)             : 0
  const speedNorm = row.inferenceTimeMs !== null && maxTime > 0
    ? Math.max(0, 1 - row.inferenceTimeMs / maxTime) : 0
  return 0.35 * psnrNorm + 0.35 * ssimNorm + 0.2 * lpipsNorm + 0.1 * speedNorm
}
```

**Table layout:**
| Rank | Model | PSNR ↕ | SSIM ↕ | LPIPS ↕ | Time ↕ | Score |
|------|-------|---------|--------|---------|-------|-------|
| 🥇 1 | U-Net | 27.3 dB | 0.82 | 0.14 | 42 ms | 0.87 |

**Colour-coded cells:** reuse quality thresholds from `MetricsCards.vue`

**Rank badges:**
- Rank 1 → `Trophy` icon (gold / amber-400)
- Rank 2 → `Medal` icon (silver / gray-400)
- Rank 3 → `Medal` icon (bronze / amber-700)
- Rank 4+ → number only

---

### Step 9 — Benchmark tab: per-image drill-down

Inside benchmark tab results, below ranking table:

```html
<details class="card">
  <summary class="cursor-pointer text-sm font-semibold text-gray-700 dark:text-gray-300 py-1">
    Per-Image Details
    <ChevronDown class="w-4 h-4 inline ml-1" />
  </summary>
  
  <div class="mt-4 space-y-4">
    <div class="flex items-center gap-3">
      <label class="text-xs text-gray-500">Image:</label>
      <select v-model="selectedImage" class="select !py-1 !text-xs">
        <option v-for="fname in imageFilenames" :key="fname" :value="fname">{{ fname }}</option>
      </select>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'psnr')" ... />
      <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'ssim')" ... />
      <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'lpips')" ... />
      <BenchmarkBarChart :models="perImageBarDataFor(selectedImage, 'inference_time_ms')" ... />
    </div>
  </div>
</details>
```

---

## Phase 3: Compare Page — Enhanced Visualizations

### Step 10 — Extend RadarChart with `inferenceTime` axis

**File:** `frontend/src/components/RadarChart.vue`

**Update `RadarSlot` interface:**
```typescript
export interface RadarSlot {
  label:          string
  psnr:           number | null
  ssim:           number | null
  lpips?:         number | null
  inferenceTime?: number | null  // ← add: forward-pass ms; lower = better; normalised to 0-1 (inverted)
  color:          string
}
```

**Update `normalise()` function:**
```typescript
function normalise(slot: RadarSlot, maxTime: number): number[] {
  const psnrNorm  = slot.psnr  !== null ? Math.min(slot.psnr / MAX_PSNR.value, 1) : 0
  const ssimNorm  = slot.ssim  !== null ? Math.max(0, Math.min(slot.ssim, 1))      : 0

  const lpipsNorm = slot.lpips !== null && slot.lpips !== undefined
    ? Math.max(0, Math.min(1 - slot.lpips, 1)) : null

  const speedNorm = slot.inferenceTime !== null && slot.inferenceTime !== undefined && maxTime > 0
    ? Math.max(0, 1 - slot.inferenceTime / maxTime) : null

  const axes: number[] = [psnrNorm, ssimNorm]
  if (lpipsNorm !== null) axes.push(lpipsNorm)
  if (speedNorm !== null) axes.push(speedNorm)
  return axes
}
```

**Computed `maxInferenceTime`** (from all slots):
```typescript
const maxInferenceTime = computed(() =>
  Math.max(...props.slots.map(s => s.inferenceTime ?? 0), 1)
)
```

**Update `radarLabels`** computed:
```typescript
const hasSpeed = computed(() =>
  props.slots.some(s => s.inferenceTime !== null && s.inferenceTime !== undefined)
)

const radarLabels = computed(() => {
  const labels = ['PSNR', 'SSIM']
  if (hasLpips.value)  labels.push('Perceptual (1−LPIPS)')
  if (hasSpeed.value)  labels.push('Speed (norm)')
  return labels
})
```

**Update `maxPsnr` prop** documentation to note that normalisation is done internally.

---

### Step 11 — Pass LPIPS + inferenceTime to radar from ComparePage

**File:** `frontend/src/pages/ComparePage.vue`

Update `radarSlots` computed:
```typescript
const radarSlots = computed((): RadarSlot[] =>
  results.value.map((r, i) => ({
    label:         r.label,
    psnr:          r.result.metrics.psnr,
    ssim:          r.result.metrics.ssim,
    lpips:         r.result.metrics.lpips ?? null,       // ← add
    inferenceTime: r.result.metrics.inference_time_ms ?? null,  // ← add
    color:         SLOT_COLORS[i],
  })),
)
```

---

### Step 12 — Add ranking table to ComparePage

**File:** `frontend/src/pages/ComparePage.vue`

Add import:
```typescript
import RankingTable from '@/components/RankingTable.vue'
import type { RankingRow } from '@/components/RankingTable.vue'
```

Add computed:
```typescript
const rankingRows = computed((): RankingRow[] =>
  results.value.map((r, i) => ({
    label:           r.label,
    model:           r.model,
    color:           SLOT_COLORS[i],
    psnr:            r.result.metrics.psnr,
    ssim:            r.result.metrics.ssim,
    lpips:           r.result.metrics.lpips ?? null,
    inferenceTimeMs: r.result.metrics.inference_time_ms ?? null,
  }))
)
```

Add to template (below the radar chart card):
```html
<div class="card space-y-3">
  <h2 class="text-sm font-semibold text-gray-700 dark:text-gray-300">Rankings</h2>
  <RankingTable :rows="rankingRows" />
</div>
```

---

## Phase 4: Shared Components

### Step 13 — `RankingTable.vue`

**Location:** `frontend/src/components/RankingTable.vue`

Key features:
- Sortable by any column (click header to sort, click again to reverse)
- Colour-coded cells using same thresholds as `MetricsCards.vue`
- Rank computed from combined weighted score
- Trophy/medal icons for top 3 (lucide `Trophy` icon with gold/silver/bronze colours)
- Handles null values gracefully (show `—`)
- Shows colour dot per model (uses `row.color` if provided)
- Exports `RankingRow` interface type

---

### Step 14 — `BenchmarkBarChart.vue`

**Location:** `frontend/src/components/BenchmarkBarChart.vue`

Key features:
- Wraps `vue-chartjs` `Bar`
- Registers: `CategoryScale`, `LinearScale`, `BarElement`, `Title`, `Tooltip`
- Each bar gets its model's colour
- `lowerIsBetter` prop toggles a subtle annotation or just guides colour-coding (red=highest, green=lowest)
- Dark mode aware (hasDarkClass() pattern from MetricsPage)
- Fixed height: 180px
- Exports the component with proper TypeScript props

---

## Relevant Files

### Backend (modify)
- `api/services/classical_colorizer.py` — Step 1: add `inference_time_ms` + `time_inference` usage
- `api/services/metrics_service.py` — Step 2: add `benchmark()` method + `_classical_colorizer`
- `api/routes/metrics.py` — Step 2: add `/api/metrics/benchmark` route

### Frontend (modify)
- `frontend/src/types/index.ts` — Step 4: add Benchmark types; update `ImageMetrics`
- `frontend/src/api/metrics.ts` — Step 3: add `benchmark()` API call
- `frontend/src/pages/MetricsPage.vue` — Steps 5–9: tabs + benchmark tab content
- `frontend/src/pages/ComparePage.vue` — Steps 11–12: pass LPIPS/time to radar, add ranking table
- `frontend/src/components/RadarChart.vue` — Step 10: add `inferenceTime` axis to `RadarSlot`

### Frontend (create new)
- `frontend/src/components/RankingTable.vue` — Step 13
- `frontend/src/components/BenchmarkBarChart.vue` — Step 14

---

## Verification Checklist

- [ ] **Step 1 verify**: `python -c "from api.services.classical_colorizer import ClassicalColorizer; c = ClassicalColorizer(); print('ok')"` — no import errors
- [ ] **Step 2 verify**: `curl -X POST http://localhost:5000/api/metrics/benchmark -H 'Content-Type: application/json' -d '{"models":[{"model":"classical_welsh","checkpoint":"","label":"Welsh"}],"max_images":2}'` — returns `{"results":[...]}` with `inference_time_ms`
- [ ] **Step 3–4 verify**: `cd frontend && npx tsc --noEmit` — zero type errors
- [ ] **Step 5–9 verify**: `/metrics` → "All Models Benchmark" tab → Run Benchmark → 4 bar charts appear, ranking table shows gold/silver/bronze badges
- [ ] **Step 10–12 verify**: `/compare` → upload image → run 3 models → radar has 4 axes (PSNR/SSIM/LPIPS/Speed), ranking table appears below
- [ ] **Classical check**: Welsh and Levin appear in benchmark with non-null metrics and `inference_time_ms`
- [ ] **Dark mode**: Both pages look correct in dark mode

---

## Decisions

- Classical models use **first sorted test image** as reference for all others; that first image is excluded from classical evaluation
- Benchmark lives as a **tab within `/metrics`**, not a separate page
- Charts: **4 individual bar charts** (one per metric) + **ranking table** with colour-coded cells
- Compare page: extend radar to **4 axes** (PSNR, SSIM, 1−LPIPS, Speed) + add ranking table below
- No new npm dependencies — `chart.js` + `vue-chartjs` already installed
- `RankingTable.vue` and `BenchmarkBarChart.vue` are shared between both pages
- Benchmark API timeout: 10 minutes (600 s) — 6 models × N images on CPU can be slow

## Further Considerations

1. **Long-running benchmarks**: Running 6 models × N images could take several minutes on CPU. Current plan uses a simple spinner. If needed, SSE progress streaming (similar to the Training page's `useSSE` composable) can be added as a follow-up.
2. **Caching**: Results are not cached server-side — the test set is small enough for interactive use.
