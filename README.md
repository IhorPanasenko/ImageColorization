# Automatic Image Colorization Ensemble — Master's Thesis

**Research and Comparative Analysis of Methods for Automatic Image Colorization**

This repository contains the full practical implementation for a Master's Thesis project that develops and benchmarks four progressively more sophisticated deep learning architectures for automatic image colorization. The project includes a complete **Flask REST API** backend and a **Vue 3 single-page application** for training, inference, evaluation, and history browsing — all in one monorepo.

---

## Architecture Overview

All models operate in the **CIE L\*a\*b\*** color space: the `L` (lightness) channel is the input and the `ab` (chrominance) channels are predicted.

| Stage | Model | Loss | Key idea |
|-------|-------|------|----------|
| 1 | **Baseline CNN** | MSE | Simple encoder-decoder; produces desaturated "sepia" tones |
| 2 | **U-Net** | L1 | Skip connections preserve spatial detail; sharper, more vibrant colors |
| 3 | **Pix2Pix GAN** | GAN + L1×100 | Adversarial training forces photorealistic local textures |
| 4 | **Fusion GAN** | GAN + L1×100 | Global ResNet-18 semantic features fused into the GAN bottleneck |

---

## Project Structure

```
ImageColorizationAnsamble/
├── run.py                      ← single entry point (starts Flask web server)
├── Dockerfile                  ← multi-stage: Node build + Python serve
├── .gitignore
├── README.md
│
├── docs/
│   ├── IMPLEMENTATION_PLAN.md
│   ├── PROJECT_CONTEXT.md
│   └── UI_IMPLEMENTATION_PLAN.md
│
├── data/                       ← shared, gitignored
│   ├── coco/val2017/           ← COCO val2017 images (not committed)
│   └── test_samples/           ← hand-picked images for evaluation
│
├── outputs/                    ← shared, gitignored
│   ├── checkpoints/            ← model weights (.pth)
│   ├── images/                 ← comparison strips from evaluate.py
│   ├── runs/                   ← TensorBoard event files + training logs
│   └── uploads/                ← temporary API uploads
│
├── ml/                         ← ML training & evaluation
│   ├── requirements.txt
│   ├── src/
│   │   ├── losses/             ← GANLoss
│   │   ├── models/             ← baseline_cnn, u_net, unet_fusion, discriminator, global_hints
│   │   └── utils/              ← dataset, common, metrics
│   ├── scripts/
│   │   ├── evaluate.py
│   │   └── trains/             ← train_baseline, train_unet, train_gan, train_fusion, train.py
│   └── tests/                  ← pytest test suite
│
├── api/                        ← Flask REST API
│   ├── requirements.txt
│   ├── app.py                  ← Flask app factory
│   ├── routes/                 ← training, inference, metrics, models, history
│   └── services/               ← train_runner, colorizer, metrics_service, checkpoint_service
│
└── frontend/                   ← Vue 3 + Vite SPA
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── src/
        ├── api/                ← typed Axios modules
        ├── components/         ← shared UI components
        ├── composables/        ← useSSE, useTraining
        ├── pages/              ← Dashboard, Training, Colorize, Metrics, Compare, History, Batch
        └── types/              ← TypeScript interfaces
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Apple Silicon MPS / NVIDIA CUDA / CPU (auto-detected)

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd ImageColorizationAnsamble

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
# ML dependencies
pip install -r ml/requirements.txt

# API dependencies
pip install -r api/requirements.txt
```

### 3. Set up data

Download the COCO val2017 dataset and place images at `data/coco/val2017/`:

```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/coco/
```

### 4. Build the frontend

```bash
cd frontend
npm install
npm run build       # produces frontend/dist/
cd ..
```

### 5. Start the web application

```bash
# Production mode — serves Vue SPA from frontend/dist/
python run.py

# Development mode — enables CORS for Vite dev server on :5173
python run.py --dev
```

Open **http://localhost:5000** in your browser.

#### Dev mode (hot-reload frontend)

```bash
# Terminal 1 — Flask API
python run.py --dev

# Terminal 2 — Vite dev server
cd frontend && npm run dev
```

Open **http://localhost:5173** — all `/api/*` requests are proxied to Flask on `:5000`.

#### Optional flags

```bash
python run.py --host 0.0.0.0 --port 8080 --debug
```

---

## Web UI Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | Model stats, active runs, quick actions |
| Training | `/training` | Configure & start training with live SSE progress |
| Colorize | `/colorize` | Upload an image, select model, download result |
| Metrics | `/metrics` | Run evaluation on test samples, view PSNR/SSIM/LPIPS |
| Compare | `/compare` | Upload one image, compare all models side-by-side |
| History | `/history` | Browse past training runs, loss curves, logs |
| Batch | `/batch` | Colorize multiple images at once |

---

## REST API Reference

All endpoints are prefixed with `/api`.

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/training/start` | Start a training run `{model, epochs, batch_size, lr, ...}` |
| `GET` | `/training/status/<run_id>` | Current run status + parsed progress |
| `GET` | `/training/stream/<run_id>` | **SSE** live progress stream |
| `POST` | `/training/stop/<run_id>` | Stop a running training process |
| `GET` | `/training/runs` | List all runs (active + historical) |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/colorize` | Colorize a single uploaded image |
| `POST` | `/colorize/batch` | Colorize multiple uploaded images |
| `GET` | `/colorize/result/<filename>` | Retrieve a saved result image |

### Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/metrics/evaluate` | Run PSNR/SSIM/LPIPS evaluation on test samples |
| `POST` | `/metrics/compare` | Compare multiple models on one image |

### Models & Checkpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List all 4 model types with metadata |
| `GET` | `/checkpoints` | List all available checkpoints |
| `GET` | `/checkpoints/<model_type>` | Checkpoints filtered by model type |

### History

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/history/runs` | All past training runs |
| `GET` | `/history/logs/<run_id>` | Parsed logs (epochs, losses, LRs, lines) |
| `GET` | `/history/tensorboard-data/<model_type>` | TensorBoard scalar data by tag |
| `DELETE` | `/history/<run_id>` | Delete a run record |

---

## CLI Training (without the Web UI)

All training scripts share the same CLI interface.

```bash
# Activate venv first
source venv/bin/activate

# Stage 1 — Baseline CNN
python ml/scripts/trains/train_baseline.py \
    --epochs 20 --batch_size 16 \
    --data_path ./data/coco/val2017 \
    --save_dir ./outputs/checkpoints

# Stage 2 — U-Net
python ml/scripts/trains/train_unet.py \
    --epochs 20 --batch_size 16 \
    --data_path ./data/coco/val2017

# Stage 3 — Pix2Pix GAN
python ml/scripts/trains/train_gan.py \
    --epochs 20 --batch_size 8 \
    --data_path ./data/coco/val2017

# Stage 4 — Fusion GAN
python ml/scripts/trains/train_fusion.py \
    --epochs 20 --batch_size 8 \
    --data_path ./data/coco/val2017

# Unified dispatcher
python ml/scripts/trains/train.py --model unet --epochs 20 \
    --data_path ./data/coco/val2017
```

### Resuming a run

```bash
python ml/scripts/trains/train_unet.py --epochs 40 \
    --data_path ./data/coco/val2017 \
    --resume ./outputs/checkpoints/unet_epoch_20.pth
```

### TensorBoard

```bash
tensorboard --logdir ./outputs/runs
# Open http://localhost:6006
```

---

## CLI Evaluation (without the Web UI)

```bash
python ml/scripts/evaluate.py \
    --model unet \
    --checkpoint ./outputs/checkpoints/unet_final.pth \
    --img_path ./data/test_samples
```

Saves `[Grayscale | Prediction | Ground Truth]` comparison strips to `outputs/images/` and prints a PSNR / SSIM / LPIPS table.

| Model flag | Checkpoint example |
|------------|--------------------|
| `baseline` | `baseline_cnn_final.pth` |
| `unet` | `unet_final.pth` |
| `gan` | `gan_generator_final.pth` |
| `fusion` | `fusion_generator_final.pth` |

---

## Testing

```bash
cd ml
python -m pytest tests/ -v

# Skip slow LPIPS tests
python -m pytest tests/ -v --ignore=tests/test_metrics.py
```

---

## Docker

### Build

```bash
docker build -t colorization-ensemble .
```

The Dockerfile uses a **two-stage build**:

1. **Stage 1 (node:20-slim)** — installs frontend dependencies and runs `npm run build`
2. **Stage 2 (python:3.11-slim)** — installs ML + API Python packages, copies source and the built Vue `dist/`

### Run

```bash
# Production web server on port 5000
docker run -p 5000:5000 colorization-ensemble

# Mount your checkpoints and data
docker run -p 5000:5000 \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/data:/workspace/data \
  colorization-ensemble

# Also expose TensorBoard
docker run -p 5000:5000 -p 6006:6006 \
  -v $(pwd)/outputs:/workspace/outputs \
  colorization-ensemble
```

---

## Quantitative Metrics

| Metric | Measures | Better |
|--------|----------|--------|
| **PSNR** | Pixel-level accuracy (dB) | Higher |
| **SSIM** | Structural / perceptual similarity | Higher (max 1.0) |
| **LPIPS** | Learned perceptual similarity (AlexNet) | Lower (min 0.0) |

---

## Hardware Notes

The codebase automatically selects the best available device:

```
Priority: Apple Silicon MPS → NVIDIA CUDA → CPU
```

Override with `--device mps|cuda|cpu` on any CLI script.

| Hardware | Recommended `--batch_size` |
|----------|---------------------------|
| Apple M-series (MPS) | 8 |
| NVIDIA GPU (≥8 GB VRAM) | 16 |
| CPU only | 2–4 |

---

## License

See [LICENSE](LICENSE).
