# UI Implementation Plan — Image Colorization Ensemble

> **Stack:** Flask (REST API) + Vue 3 (Vite + TypeScript + Tailwind CSS)
> **Architecture:** Monorepo with `ml/`, `api/`, `frontend/` top-level split
> **Created:** 2026-02-25

---

## Table of Contents

1. [Project Restructure](#phase-0-project-restructure)
2. [Backend — Flask API](#phase-1-backend--flask-api)
3. [Frontend — Vue 3 + Vite](#phase-2-frontend--vue-3--vite)
4. [Integration & Polish](#phase-3-integration--polish)
5. [Target Directory Structure](#target-directory-structure)
6. [Decisions Log](#decisions-log)

---

## Phase 0: Project Restructure ✅ COMPLETED

> All steps below have been implemented.

The existing ML code lives at the repository root (`src/`, `scripts/`, `tests/`).
Before adding the API and frontend, reorganize into a clean monorepo layout.

### Step 0.1 ✅ — Move ML code into `ml/`

Move these items into a new `ml/` directory:

| Current location | New location |
|------------------|-------------|
| `src/` | `ml/src/` |
| `scripts/` | `ml/scripts/` |
| `tests/` | `ml/tests/` |
| `requirements.txt` | `ml/requirements.txt` |

**Keep at root (shared by all three packages):**
- `data/` — training images, test samples (shared between ML training and API inference)
- `outputs/` — checkpoints, runs, images (shared between ML training and API)
- `docs/` — all documentation
- `Dockerfile` — will be rewritten to build all three tiers
- `.devcontainer/` — VS Code config
- `.gitignore` — updated for all three packages
- `README.md` — updated
- `LICENSE`
- `notes.txt`

### Step 0.2 ✅ — Fix all `sys.path` and import paths inside `ml/`

After the move, every training script's `sys.path.append` must resolve to `ml/`
(one level up from `ml/scripts/trains/`). The `src` package import prefix stays the
same (`from src.models.u_net import UNet`) — it just lives inside `ml/` now.

Files to update:
- `ml/scripts/trains/train_baseline.py` — `sys.path` → `../../` (resolves to `ml/`)
- `ml/scripts/trains/train_unet.py` — same
- `ml/scripts/trains/train_gan.py` — same (already correct: `../..`)
- `ml/scripts/trains/train_fusion.py` — same (already correct: `../..`)
- `ml/scripts/trains/train.py` — same (already correct: `../..`)
- `ml/scripts/evaluate.py` — `sys.path` → `..` (resolves to `ml/`)
- `ml/tests/test_*.py` — `sys.path` → `..` (resolves to `ml/`)

### Step 0.3 ✅ — Create `api/` scaffold

```
api/
├── __init__.py
├── app.py                  # Flask app factory, blueprint registration
├── requirements.txt        # flask, flask-cors, gunicorn
├── routes/
│   ├── __init__.py
│   ├── training.py         # /api/training/* endpoints
│   ├── inference.py        # /api/colorize/* endpoints
│   ├── metrics.py          # /api/metrics/* endpoints
│   ├── models.py           # /api/models, /api/checkpoints endpoints
│   └── history.py          # /api/history/* endpoints
└── services/
    ├── __init__.py
    ├── train_runner.py     # subprocess management for training jobs
    ├── colorizer.py        # model loading + inference wrapper
    ├── metrics_service.py  # evaluation + comparison logic
    └── checkpoint_service.py # checkpoint discovery
```

### Step 0.4 ✅ — Create `frontend/` scaffold

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── public/
├── src/
│   ├── main.ts
│   ├── App.vue
│   ├── router/index.ts
│   ├── api/                # Axios API client modules
│   ├── components/         # Shared UI components
│   ├── composables/        # Vue composables (useTraining, useSSE, etc.)
│   ├── pages/              # Route-level page components
│   ├── types/              # TypeScript interfaces
│   └── assets/             # Static assets, CSS
```

### Step 0.5 ✅ — Create root `run.py`

A single entry point at the repository root:
```
python run.py          # starts Flask API on port 5000 (serves Vue dist/ in prod)
python run.py --dev    # starts Flask API on port 5000 (CORS enabled, Vue dev on 5173)
```

### Step 0.6 ✅ — Update `.gitignore`

Add entries for:
```
frontend/node_modules/
frontend/dist/
api/__pycache__/
*.log
outputs/uploads/
```

### Step 0.7 ✅ — Update `Dockerfile`

Rewrite as a multi-stage build:
1. **Stage 1 (node):** build Vue frontend → produces `frontend/dist/`
2. **Stage 2 (python):** install ML + API Python deps, copy source + built frontend
3. Expose ports 5000 (web app) + 6006 (TensorBoard)
4. Default CMD: `python run.py`

---

## Phase 1: Backend — Flask API ✅ COMPLETED

> All steps below have been implemented.

### Step 1.1 ✅ — `api/app.py` — Flask app factory

- Create Flask app, register all blueprints from `api/routes/`
- In production mode: serve Vue static files from `frontend/dist/`
- In dev mode: enable CORS for `http://localhost:5173` (Vite dev server)
- Configure upload folder: `outputs/uploads/`
- Set max upload size: 16 MB

### Step 1.2 ✅ — `api/services/train_runner.py` — Training subprocess manager

Core responsibilities:
- Launch training scripts (`ml/scripts/trains/train_*.py`) as **background `subprocess.Popen`**
- Each run gets a unique `run_id` (UUID)
- Redirect stdout/stderr to `outputs/runs/{run_id}/train.log`
- Parse tqdm/print output via regex → extract epoch, loss, LR, ETA
- Store run state in memory + persist to `outputs/runs/runs.json`
- Support **stopping** via `process.terminate()`
- Construct CLI args from JSON params (epochs, batch_size, lr, lambda_l1, data_path, resume)

### Step 1.3 ✅ — `api/routes/training.py` — Training endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/training/start` | Accept `{model, epochs, batch_size, lr, lambda_l1, data_path, resume_g, resume_d}`, start subprocess, return `{run_id}` |
| `GET` | `/api/training/status/<run_id>` | Return `{status, epoch, total_epochs, loss, loss_d, loss_g, lr, log_tail}` |
| `GET` | `/api/training/stream/<run_id>` | **SSE** endpoint — tail log file, push parsed progress events every 2s |
| `POST` | `/api/training/stop/<run_id>` | Terminate subprocess |
| `GET` | `/api/training/runs` | List all runs (active + historical) |

### Step 1.4 ✅ — `api/services/colorizer.py` — Inference wrapper

- `colorize_image(image_path, model_type, checkpoint_path)` →
  load model (same logic as `ml/scripts/evaluate.py` `load_model()`),
  run `prepare_grayscale_input()`, inference, `lab_to_rgb()`,
  return `(pred_rgb, gray_rgb, gt_rgb)` as numpy arrays
- `colorize_from_color(image_path, model_type, checkpoint_path)` →
  color photo → extract L → inference → return `(pred_rgb, gray_rgb, original_rgb)`.
  This is the "color → grayscale → re-colorize" flow.
- `colorize_batch(image_paths, model_type, checkpoint_path)` →
  process a list, return list of result dicts
- **Model cache** (LRU dict, max 4 entries) to avoid reloading weights per request

### Step 1.5 ✅ — `api/routes/inference.py` — Colorize endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/colorize` | Multipart file upload + `{model, checkpoint, mode}` form fields. Save to `outputs/uploads/`, run colorizer, return JSON with base64-encoded images |
| `POST` | `/api/colorize/batch` | Multiple files, return array of results |
| `GET` | `/api/colorize/result/<filename>` | Serve saved result image |

### Step 1.6 ✅ — `api/services/metrics_service.py` — Evaluation logic

- `evaluate_single(pred_rgb, gt_rgb, device)` → `{psnr, ssim, lpips}`
- `evaluate_samples(model_type, checkpoint, sample_dir)` →
  run all images from `data/test_samples/`, compute per-image + averaged metrics,
  save comparison strips, return structured results
- `compare_models(image_path, model_configs)` →
  run one image through multiple models, return metrics + images for each

### Step 1.7 ✅ — `api/routes/metrics.py` — Metrics endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/metrics/evaluate` | Accept `{model, checkpoint, sample_dir}`, return per-image + averaged metrics |
| `POST` | `/api/metrics/compare` | Accept image + list of `{model, checkpoint}` configs, return comparison |

### Step 1.8 ✅ — `api/services/checkpoint_service.py` — Checkpoint discovery

- `list_checkpoints()` → `[{filename, model_type, epoch, size_mb, modified}]`
  parsed from filenames (`unet_epoch_10.pth`, `gan_generator_final.pth`, etc.)
- `get_available_models()` → which model types have at least one checkpoint

### Step 1.9 ✅ — `api/routes/models.py` — Model & checkpoint endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/models` | List all 4 model types with metadata (name, description, param count, available checkpoints) |
| `GET` | `/api/checkpoints` | List all checkpoints |
| `GET` | `/api/checkpoints/<model_type>` | Filtered by model type |

### Step 1.10 ✅ — `api/routes/history.py` — Training history endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/history/runs` | Read `runs.json`, return all past training runs |
| `GET` | `/api/history/logs/<run_id>` | Return parsed training logs (epoch, loss, lr arrays) |
| `GET` | `/api/history/tensorboard-data/<model_type>` | Parse TensorBoard event files from `outputs/runs/{model_type}/` via `EventAccumulator`, return `[{step, value}]` arrays for each scalar tag |

---

## Phase 2: Frontend — Vue 3 + Vite

### Step 2.1 ✅ — Scaffold the Vue app

Inside `frontend/`, initialize via `npm create vite@latest . -- --template vue-ts`.

Install dependencies:
- **Routing:** `vue-router`
- **Styling:** `tailwindcss`, `@tailwindcss/forms`, `postcss`, `autoprefixer`
- **Charts:** `vue-chartjs` + `chart.js` (loss curves, metrics charts, radar charts)
- **Icons:** `lucide-vue-next`
- **HTTP:** `axios`
- **File upload:** `vue3-dropzone` or manual drag-and-drop
- **Notifications:** `vue-toastification`

### Step 2.2 ✅ — Create layout shell

`App.vue` — persistent sidebar navigation + main content area:
- **Sidebar links:** Dashboard, Training, Colorize, Metrics, Compare, History, Batch
- **Header bar:** project title, dark/light mode toggle
- **Responsive:** sidebar collapses on mobile
- **Implemented:** full CSS sidebar with `translate-x` mobile drawer, dark-mode toggle (localStorage + OS preference), `lucide-vue-next` icons per route, active-link highlighting with `brand-*` colours.

### Step 2.3 ✅ — API client layer

`src/api/` folder with typed Axios modules:
- `training.ts` — `startTraining()`, `getStatus()`, `stopTraining()`, `getRuns()`
- `inference.ts` — `colorizeImage()`, `colorizeBatch()`
- `metrics.ts` — `runEvaluation()`, `compareModels()`
- `models.ts` — `getModels()`, `getCheckpoints()`
- `history.ts` — `getHistory()`, `getLogs()`, `getTensorboardData()`

Each function returns typed responses matching TypeScript interfaces in `src/types/`.

### Step 2.4 ✅ — TypeScript interfaces

`src/types/index.ts`:
- `ModelInfo { id, name, description, paramCount, checkpoints[] }`
- `Checkpoint { filename, modelType, epoch, sizeMb, modified }`
- `TrainingConfig { model, epochs, batchSize, lr, lambdaL1?, dataPath, resumeG?, resumeD? }`
- `TrainingRun { runId, model, status, epoch, totalEpochs, loss, lossD?, lossG?, lr, startedAt, finishedAt? }`
- `TrainingProgress { epoch, totalEpochs, loss, lossD?, lossG?, lr, logLines[] }` (SSE events)
- `ColorizeResult { grayscale, prediction, groundTruth?, psnr?, ssim?, lpips? }` (base64 images)
- `MetricsResult { images: ImageMetrics[], averages: { psnr, ssim, lpips } }`
- `CompareResult { models: { modelType, checkpoint, prediction, metrics }[] }`

### Step 2.5 ✅ — SSE composable

`src/composables/useSSE.ts`:
- Generic `useSSE<T>(urlRef)` — connects when URL is non-null, auto-disconnects on unmount, returns `{ data, connected, error }`
- EventSource auto-reconnects on network blip; state (connected/error) tracked reactively

`src/composables/useTraining.ts`:
- Wraps `useSSE` + `trainingApi` into a single composable
- Accumulates `lossHistory: LossPoint[]` (de-duplicated by epoch) for chart
- Buffers up to 500 log lines with auto-trim
- Exposes `{ runId, status, lossHistory, logLines, starting, stopping, isRunning, progressPct, isGan, sseConnected, start, stop }`

### Step 2.6 ✅ — Dashboard page

`src/pages/DashboardPage.vue`:
- **Stats cards:** number of checkpoints per model type, dataset size, active runs
- **Quick actions:** "Start Training" → navigates to Training page, "Colorize Image" → navigates to Colorize page
- **Active run widget:** if a training run is in progress, show a mini progress bar + loss here
- **Recent results:** last 5 colorization results as thumbnails

### Step 2.7 ✅ — Training page ⭐

`src/pages/TrainingPage.vue` — the main training control panel:

**Configuration form:**
- Model selector dropdown: `Baseline CNN`, `U-Net`, `Pix2Pix GAN`, `Fusion GAN`
- Parameter fields (dynamically shown based on model):
  - `Epochs` — number input (default: 20)
  - `Batch Size` — number input (default: 16 for baseline/unet, 8 for GAN/fusion)
  - `Learning Rate` — number input with scientific notation (default varies)
  - `Lambda L1` — number input, **only shown for GAN and Fusion** (default: 100.0)
  - `Data Path` — text input (default: `./data/coco/val2017`)
  - `Resume From` — dropdown populated from `/api/checkpoints/{model}`, with "None (start fresh)" option
  - For GAN/Fusion: separate resume dropdowns for Generator and Discriminator
- **"Start Training" button** → `POST /api/training/start`

**Live progress panel** (appears after start):
- Progress bar: "Epoch 7/20 — 35%"
- Current loss value(s): single loss for Baseline/UNet, D-loss + G-loss for GAN/Fusion
- Current learning rate
- Elapsed time + estimated remaining
- **Mini loss chart** — updates in real-time via SSE, shows loss over steps
- **Stop Training button** (red, with confirmation dialog)
- **Log viewer** — scrollable terminal-styled box, last 50 lines, auto-scrolls to bottom

### Step 2.8 ✅ — Colorize page

`src/pages/ColorizePage.vue`:

- **Upload zone** — drag-and-drop area, accepts `.jpg/.png/.jpeg/.bmp/.webp`
- **Mode toggle** (radio buttons):
  - "Grayscale → Color" — upload a B&W photo, model colorizes it
  - "Color → B&W → Re-color" — upload a color photo, system converts to grayscale, then colorizes. Shows all three: original color → extracted grayscale → model output
- **Model + Checkpoint selectors** — two dropdowns
- **"Colorize" button** → `POST /api/colorize`
- **Result display:** side-by-side image panels:
  - Mode 1: `[Grayscale Input] [Model Output]`
  - Mode 2: `[Original Color] [Extracted Grayscale] [Model Output]`
- **Download button** for the colorized result
- If ground truth is available (mode 2): show PSNR, SSIM, LPIPS below the images

### Step 2.9 ✅ — Metrics / Evaluation page

`src/pages/MetricsPage.vue`:

- **Configuration:** model + checkpoint selector, sample source (default: `data/test_samples/` or upload custom)
- **"Run Evaluation" button** → `POST /api/metrics/evaluate`
- **Results:**
  - **Summary cards** — avg PSNR, avg SSIM, avg LPIPS with color-coded quality indicators (green = good, yellow = ok, red = poor)
  - **Per-image results table:** thumbnail, filename, PSNR, SSIM, LPIPS, "View" button (opens comparison strip in a modal)
  - **Bar chart** — per-image PSNR/SSIM values (vue-chartjs `BarChart`)

### Step 2.10 ✅ — Model Comparison page

`src/pages/ComparePage.vue`:

- **Image upload** — single image
- **Model multi-select** — checkboxes for each available model/checkpoint combo (auto-populated from `/api/checkpoints`). Minimum 2, maximum all 4.
- **"Compare" button** → `POST /api/metrics/compare`
- **Results grid** — one column per model:
  - Model name header
  - Colorized output image
  - Metrics below each image: PSNR, SSIM, LPIPS
- **Radar chart** — overlaid comparison of normalized PSNR/SSIM/(1-LPIPS) across all selected models
- **Winner badge** — highlights the model with the best average metric

### Step 2.11 ✅ — Training History page

`src/pages/HistoryPage.vue`:

- **Runs table:** model type, epochs, status badge (✓ completed / ✗ failed / ● running / ◼ stopped), date, final loss, duration
- **Click a run → detail panel:**
  - **Loss curve chart** (`LineChart`): Loss/epoch over time
  - For GAN/Fusion: dual-line chart with `Loss_D/epoch` and `Loss_G/epoch`
  - **LR schedule chart**
  - **Full log viewer** — scrollable, searchable
- Data sourced from TensorBoard event files via `/api/history/tensorboard-data/{model}`

### Step 2.12 ✅ — Batch Processing page

`src/pages/BatchPage.vue`:

- **Multi-file upload** — drag folder or select multiple files
- **Model + checkpoint selector**
- **"Process All" button** → `POST /api/colorize/batch`
- **Results:** thumbnail gallery grid, each card shows input → output, hover for metrics
- **Summary stats** — avg PSNR/SSIM/LPIPS across all processed images
- **"Download All" button** — triggers ZIP download

### Step 2.13 ✅ — Shared components

| Component | Purpose |
|-----------|---------|
| `ModelSelector.vue` | Reusable model type + checkpoint dropdown pair |
| `ImageDropzone.vue` | Drag-and-drop upload zone with preview |
| `ImageCompare.vue` | Side-by-side before/after image display |
| `MetricsCards.vue` | PSNR / SSIM / LPIPS summary cards with color coding |
| `LossChart.vue` | Recharts line chart for training loss curves |
| `ProgressBar.vue` | Animated epoch progress bar with text |
| `LogViewer.vue` | Terminal-styled scrollable log output |
| `ConfirmDialog.vue` | Modal confirmation (for stop training, etc.) |
| `PageHeader.vue` | Consistent page title + description |
| `StatusBadge.vue` | Colored badge for training status |

---

## Phase 3: Integration & Polish ✅ COMPLETED

### Step 3.1 ✅ — `api/requirements.txt`

```
flask>=3.0
flask-cors>=4.0
gunicorn>=21.0
```

### Step 3.2 ✅ — Root `run.py`

Single entry point:
```python
# python run.py         → production (serves Vue dist/ via Flask)
# python run.py --dev   → development (CORS enabled, use Vite dev server on 5173)
```

Adds `ml/` to `sys.path` so API services can import `from src.models...`.

### Step 3.3 ✅ — `frontend/vite.config.ts`

Configure dev proxy:
```typescript
server: {
  port: 5173,
  proxy: {
    '/api': 'http://localhost:5000'
  }
}
```

### Step 3.4 ✅ — Update `.gitignore`

Add:
```
frontend/node_modules/
frontend/dist/
api/__pycache__/
outputs/uploads/
*.log
```

### Step 3.5 ✅ — Update `Dockerfile`

Multi-stage build:
1. **Stage 1 (node:20-slim):** `cd frontend && npm ci && npm run build`
2. **Stage 2 (python:3.11-slim):** install system libs, `pip install` ML + API deps, copy all source + `frontend/dist/`
3. Expose ports `5000` (web) + `6006` (TensorBoard)
4. CMD: `python run.py`

### Step 3.6 ✅ — Error handling & loading states

- Skeleton loaders on all data-fetching pages
- Toast notifications: training started/stopped/completed/failed, colorization complete, evaluation done
- Graceful fallback when no checkpoints are available ("Train a model first")
- File type validation on upload
- API error responses → user-friendly messages

### Step 3.7 ✅ — Update `README.md`

Document:
- New project structure
- How to run: `pip install -r ml/requirements.txt -r api/requirements.txt`, `cd frontend && npm install && npm run build`, `python run.py`
- Dev mode instructions
- API endpoint reference
- Docker build & run instructions

---

## Target Directory Structure

```
ImageColorizationAnsamble/
├── run.py                     # ← single entry point (Flask app)
├── Dockerfile                 # ← multi-stage: Node build + Python serve
├── .gitignore
├── .devcontainer/
│   └── devcontainer.json
├── LICENSE
├── README.md
├── notes.txt
│
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   ├── PROJECT_CONTEXT.md
│   └── UI_IMPLEMENTATION_PLAN.md  # ← this file
│
├── data/                      # ← shared, gitignored
│   ├── coco/val2017/
│   └── test_samples/
│
├── outputs/                   # ← shared, gitignored
│   ├── checkpoints/
│   ├── images/
│   ├── runs/
│   └── uploads/               # ← new: API file uploads
│
├── ml/                        # ← moved from root
│   ├── requirements.txt
│   ├── src/
│   │   ├── __init__.py
│   │   ├── losses/__init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── baseline_cnn.py
│   │   │   ├── discriminator.py
│   │   │   ├── global_hints.py
│   │   │   ├── u_net.py
│   │   │   └── unet_fusion.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── common.py
│   │       ├── dataset.py
│   │       └── metrics.py
│   ├── scripts/
│   │   ├── evaluate.py
│   │   └── trains/
│   │       ├── train.py
│   │       ├── train_baseline.py
│   │       ├── train_unet.py
│   │       ├── train_gan.py
│   │       └── train_fusion.py
│   └── tests/
│       ├── test_dataset.py
│       ├── test_discriminator.py
│       ├── test_losses.py
│       ├── test_metrics.py
│       └── test_model.py
│
├── api/                       # ← new: Flask backend
│   ├── __init__.py
│   ├── app.py
│   ├── requirements.txt
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   └── history.py
│   └── services/
│       ├── __init__.py
│       ├── train_runner.py
│       ├── colorizer.py
│       ├── metrics_service.py
│       └── checkpoint_service.py
│
└── frontend/                  # ← new: Vue 3 SPA
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── public/
    └── src/
        ├── main.ts
        ├── App.vue
        ├── router/index.ts
        ├── api/
        │   ├── training.ts
        │   ├── inference.ts
        │   ├── metrics.ts
        │   ├── models.ts
        │   └── history.ts
        ├── composables/
        │   ├── useSSE.ts
        │   └── useTraining.ts
        ├── components/
        │   ├── ModelSelector.vue
        │   ├── ImageDropzone.vue
        │   ├── ImageCompare.vue
        │   ├── MetricsCards.vue
        │   ├── LossChart.vue
        │   ├── ProgressBar.vue
        │   ├── LogViewer.vue
        │   ├── ConfirmDialog.vue
        │   ├── PageHeader.vue
        │   └── StatusBadge.vue
        ├── pages/
        │   ├── DashboardPage.vue
        │   ├── TrainingPage.vue
        │   ├── ColorizePage.vue
        │   ├── MetricsPage.vue
        │   ├── ComparePage.vue
        │   ├── HistoryPage.vue
        │   └── BatchPage.vue
        ├── types/
        │   └── index.ts
        └── assets/
            └── main.css
```

---

## Decisions Log

| Decision | Chosen | Over | Reason |
|----------|--------|------|--------|
| UI framework | Flask + Vue 3 | Gradio, Streamlit | Maximum customization, proper SPA, user preference |
| Vue variant | Vue 3 + Composition API + TypeScript | Vue 2, Options API | Modern, better TS support, composables |
| Build tool | Vite | Webpack, Vue CLI | Fastest DX, native ESM, simple config |
| CSS framework | Tailwind CSS | Vuetify, Bootstrap | Utility-first, lightweight, full control over design |
| Charts | vue-chartjs + chart.js | Apache ECharts, D3 | Lightweight, good Vue integration, covers all chart types needed |
| Training progress | SSE (Server-Sent Events) | WebSocket, polling | Simpler than WS for one-way streaming, native browser support |
| Project structure | Root split: `ml/` + `api/` + `frontend/` | Keep flat, packages/ | Clean separation, shared `data/` and `outputs/`, minimal nesting |
| State management | No Pinia (use composables + props) | Pinia | App is small enough — composable refs + route params suffice |
| Model caching | LRU dict (max 4) in colorizer service | No cache | Avoids reloading ~100MB weights on every request |
| Training subprocess | `subprocess.Popen` + log parsing | In-process threading | Isolation, can kill cleanly, reuses existing CLI scripts unchanged |
| Run persistence | JSON manifest file | SQLite, PostgreSQL | Single-user academic project, no need for a database |
