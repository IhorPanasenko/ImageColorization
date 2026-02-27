# Implementation Plan: Finish the Image Colorization Project

> Generated: 2026-02-24 | Based on audit of all source files vs PROJECT_CONTEXT.md

---

## TL;DR

The project has **4 model architectures** defined but is roughly **60% complete** overall. Models and training scripts for Stages 1–3 exist but contain import bugs. Stage 4 (Fusion GAN) has model code but **no training script**. The evaluation pipeline is a stub, metrics are unimplemented, and several utility files are empty. This plan fixes all bugs, fills every gap, and brings the project to a thesis-ready state in **14 ordered steps**.

---

## Current State Summary

| Component | Status | Blockers |
|-----------|--------|----------|
| Baseline CNN (Stage 1) | Model ✅, Training ✅, Checkpoints ✅ | Broken imports in `train_baseline.py` |
| U-Net (Stage 2) | Model ✅, Training ✅, Checkpoints ✅ | Broken imports (`unet` vs `u_net`) |
| Pix2Pix GAN (Stage 3) | Model ✅, Training ✅, **No checkpoints** | Discriminator has dead code; needs training run |
| Fusion GAN (Stage 4) | Model ✅, Training ✅, **No checkpoints** | Needs training run (Step 12) |
| Evaluation | ✅ Complete — full pipeline in `evaluate.py` | — |
| Metrics | ✅ Complete — `src/utils/metrics.py` | — |
| Utilities | `dataset.py` ✅, `common.py` ✅, `metrics.py` ✅ | — |
| Losses | ✅ `GANLoss` in `src/losses/__init__.py` | — |
| Tests | 3 files, some broken imports | No tests for GlobalHintNet, UNetFusion |
| README | 3 lines | No setup/usage documentation |

---

## Step-by-Step Implementation Plan

### Phase 1: Fix Critical Bugs & Infrastructure (Steps 1–4)

These steps unblock everything else by fixing imports, missing packages, and dead code.

---

#### ~~Step 1: Resolve the `u_net.py` ↔ `unet` Import Mismatch~~ ✅ Already Consistent

**Actual state (verified):** All 6 files that import UNet already consistently use `from src.models.u_net` (with underscore), which matches the actual filename `u_net.py`. No mismatch exists. The original audit was incorrect.

**No action required.** The terminal test `python3 -c "from src.models.unet import UNet"` failed because that test command itself used the wrong name — it does not reflect a bug in the source code.

---

#### ~~Step 2: Resolve the `dataset.py` Import Path Mismatch~~ ✅ Already Consistent

**Actual state (verified):** All 5 files that import `ColorizationDataset` already use `from src.utils.dataset import ColorizationDataset`. No mismatch exists.

**No action required.**

---

#### Step 3: Add Missing `__init__.py` Files ✅ Complete

**Actual state (verified):** `src/models/__init__.py` and `src/models/gan/__init__.py` already existed. Only `src/utils/__init__.py` was missing.

**Action taken:** Created `src/utils/__init__.py`.

**Verification:** `python -c "import src.models; import src.models.gan; import src.utils; print('OK')"`

---

#### Step 4: Clean Up `discriminator.py` (Remove Dead Code) ✅ Complete

**Problem resolved:** The duplicate `__init__` (lines 22–55) and broken `forward` (lines 57–66) have been removed. `PatchDiscriminator` now has exactly one `__init__` and one `forward`, both correct.

**Verification:** `python -m pytest tests/test_discriminator.py -v`

---

### Phase 2: Build Shared Utilities (Steps 5–7)

Create the reusable building blocks that training, evaluation, and metrics all depend on.

---

#### Step 5: Implement `src/utils/common.py` — Shared Utility Functions ✅ Complete

**Implemented:**
1. `get_device() → torch.device` — MPS → CUDA → CPU detection.
2. `lab_to_rgb(L, ab) → np.ndarray` — denormalizes tensors and converts Lab → RGB via `skimage`.
3. `prepare_grayscale_input(img_path, target_size) → (L_tensor, original_rgb)` — loads image, extracts normalized L, returns (1,1,H,W) tensor + ground-truth RGB.
4. `save_comparison_strip(grayscale, prediction, ground_truth, save_path, title)` — saves `[Grayscale | Prediction | Ground Truth]` matplotlib figure as PNG.

---

#### Step 6: Implement `src/utils/metrics.py` — Quantitative Evaluation ✅ Complete

**Implemented:**
1. `compute_psnr(pred, target) → float` — uses `skimage.metrics.peak_signal_noise_ratio`.
2. `compute_ssim(pred, target) → float` — uses `skimage.metrics.structural_similarity` with `channel_axis=2`.
3. `compute_lpips(pred, target, device) → float` — uses `lpips.LPIPS(net='alex')`; model loaded once at module level.
4. `evaluate_batch(pred_images, target_images, device) → dict` — averages all three metrics over a list of image pairs.

---

#### Step 7: Move `GANLoss` to `src/losses/` — Reusable Loss Module ✅ Complete

**Done:**
1. `GANLoss` class moved from `train_gan.py` into `src/losses/__init__.py` with type annotations.
2. `train_gan.py` updated: `GANLoss` class definition removed, `from src.losses import GANLoss` added to imports.
3. `train_fusion.py` (Step 8) will import from same location.

---

### Phase 3: Complete Stage 4 — Fusion Training (Step 8)

---

#### Step 8: Implement `scripts/trains/train_fusion.py` — Ensemble GAN Training ✅ Complete

**Implemented:** Full Fusion GAN training loop (119 lines). `GlobalHintNet` frozen (`hint_net.eval()`, no optimizer); `with torch.no_grad()` wraps the hint forward pass. `UNetFusion` generator + `PatchDiscriminator`. GANLoss imported from `src.losses`. Checkpoints saved with `fusion_` prefix. `sys.path` points two levels up (`'..', '..'`).

~~**Current state:** File is completely empty (0 lines).~~

**Implement by following the pattern of `train_gan.py`**, with these key differences:

1. **Models initialized:**
   - `net_G = UNetFusion().to(device)` (instead of `UNet`)
   - `net_D = PatchDiscriminator().to(device)` (same discriminator)
   - `hint_net = GlobalHintNet().to(device)` + `hint_net.eval()` (frozen feature extractor)

2. **Forward pass change in training loop:**
   ```
   global_hint = hint_net(real_L)        # (Batch, 512) — semantic context
   fake_ab = net_G(real_L, global_hint)  # UNetFusion takes two args
   ```

3. **Loss functions:** Same as GAN — `GANLoss` (imported from `src.losses`) + L1 * λ=100.

4. **Optimizers:** Adam for G and D with `lr=2e-4, betas=(0.5, 0.999)`. `hint_net` has no optimizer (frozen).

5. **Checkpoint saving:**
   - `fusion_generator_epoch_{N}.pth`, `fusion_discriminator_epoch_{N}.pth` every 5 epochs.
   - `fusion_generator_final.pth`, `fusion_discriminator_final.pth` at end.
   - Save to `./outputs/checkpoints/`.

6. **CLI args:** Same as `train_gan.py` — `--epochs`, `--batch_size`, `--lr`, `--data_path`, `--save_dir`.

**Verification:** `python scripts/trains/train_fusion.py --epochs 1 --batch_size 2` completes without error.

---

### Phase 4: Complete Evaluation Pipeline (Steps 9–10)

---

#### Step 9: Complete `scripts/evaluate.py` — Full Inference & Visualization ✅ Complete

**Implemented:** Complete rewrite (188 lines). `load_model()` handles all 4 types and returns `(model, hint_net)`. `process_image()` runs full Lab inference pipeline. `evaluate_single()` computes PSNR/SSIM/LPIPS and saves comparison strip. `main()` handles both single-image and directory modes, auto-creates `outputs/images/`, and prints aggregate summary table.

~~**Current state:** `process_image()` is an empty stub.~~

**Action — Implement the following inside `evaluate.py`:**

1. **Fix imports:**
   - Uncomment `UNetFusion` and `GlobalHintNet` imports.
   - Add `from src.utils.common import get_device, lab_to_rgb, save_comparison_strip`.
   - Add `from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips`.

2. **Implement `process_image(img_path, model, hint_net, device) → (pred_rgb, gt_rgb, gray_rgb)`:**
   - Open image, convert to Lab, normalize L and ab.
   - Run `model(L_tensor)` (or `model(L_tensor, hint_net(L_tensor))` for fusion).
   - Use `lab_to_rgb()` to convert prediction back to RGB.
   - Return prediction, ground truth, and grayscale for comparison.

3. **Implement `evaluate_single(img_path, model, hint_net, device, save_dir):`**
   - Call `process_image`.
   - Compute PSNR, SSIM, LPIPS between prediction and ground truth.
   - Call `save_comparison_strip` to create `[Gray | Pred | GT]` image.
   - Return metrics dict.

4. **Implement `main()` function:**
   - Parse args, auto-detect device.
   - Load model via `load_model()` (fix fusion branch to actually load).
   - If `--img_path` is a directory, iterate over all images; if a file, process one.
   - Aggregate metrics across all images, print summary table.
   - Create `outputs/images/` directory automatically.

5. **Fix `load_model()` for fusion:**
   - Uncomment and complete the fusion branch.
   - Load `UNetFusion` weights + create frozen `GlobalHintNet`.
   - Return `(model, hint_net)` tuple.

**Verification:** Run on a single test image: `python scripts/evaluate.py --model baseline --checkpoint ./checkpoints/baseline_cnn_final.pth --img_path ./data/coco/val2017/000000000139.jpg`

---

#### Step 10: Add Sample Test Images to `data/test_samples/` ✅ Complete

**Action taken:** Copied 10 evenly-spaced images from `data/coco/val2017/` into `data/test_samples/` using Python `shutil.copy`.

**Files copied:** `000000000139.jpg`, `000000057027.jpg`, `000000119365.jpg`, `000000173091.jpg`, `000000229753.jpg`, `000000289594.jpg`, `000000347544.jpg`, `000000405249.jpg`, `000000463174.jpg`, `000000521601.jpg`

**Verification:** `ls data/test_samples/ | wc -l` → 10

---

### Phase 5: Training Runs (Steps 11–12)

---

#### Step 11: Train GAN (Stage 3) — Generate Missing Checkpoints

**Current state:** `train_gan.py` is complete but was never run (no checkpoints exist).

**Action:**
```bash
python scripts/trains/train_gan.py --epochs 20 --batch_size 8 --data_path ./data/coco/val2017 --save_dir ./outputs/checkpoints
```

**Expected output:**
- `outputs/checkpoints/gan_generator_epoch_{5,10,15,20}.pth`
- `outputs/checkpoints/gan_discriminator_epoch_{5,10,15,20}.pth`
- `outputs/checkpoints/gan_generator_final.pth`
- `outputs/checkpoints/gan_discriminator_final.pth`

**Verification:** File sizes should be ~100–200 MB for generator, ~10–50 MB for discriminator.

---

#### Step 12: Train Fusion GAN (Stage 4) — Generate Missing Checkpoints

**Action:**
```bash
python scripts/trains/train_fusion.py --epochs 20 --batch_size 8 --data_path ./data/coco/val2017 --save_dir ./outputs/checkpoints
```

**Expected output:**
- `outputs/checkpoints/fusion_generator_epoch_{5,10,15,20}.pth`
- `outputs/checkpoints/fusion_discriminator_epoch_{5,10,15,20}.pth`
- `outputs/checkpoints/fusion_generator_final.pth`
- `outputs/checkpoints/fusion_discriminator_final.pth`

**Note:** Training will take longer than plain GAN due to the additional ResNet forward pass per batch.

**Verification:** File sizes comparable to GAN checkpoints.

---

### Phase 6: Testing & Documentation (Steps 13–14)

---

#### Step 13: Expand Test Suite ✅ Complete

**Current state:** 3 test files covering BaselineCNN, UNet, PatchDiscriminator, ColorizationDataset. No tests for GlobalHintNet, UNetFusion, GANLoss, or metrics.

**Implemented:**

1. **`tests/test_model.py`** — added:
   - `test_global_hint_net_output_shape`: input `(2, 1, 256, 256)` → output `(2, 512)`.
   - `test_unet_fusion_output_shape`: input `(2, 1, 256, 256)` + hint `(2, 512)` → output `(2, 2, 256, 256)`.
   - `test_unet_fusion_with_global_hint_net`: end-to-end — L through GlobalHintNet, then UNetFusion.

2. **`tests/test_losses.py`** (new file):
   - `test_gan_loss_real`: GANLoss returns positive scalar.
   - `test_gan_loss_fake`: GANLoss returns positive scalar.
   - `test_gan_loss_real_higher_than_fake_on_positive_logits`: validates directional correctness.

3. **`tests/test_metrics.py`** (new file — skipped automatically if `lpips` not installed):
   - `test_psnr_identical_images`, `test_psnr_different_images`
   - `test_ssim_identical_images`, `test_ssim_different_images`
   - `test_lpips_identical_images`, `test_lpips_different_images`
   - `test_evaluate_batch_returns_all_keys`

---

#### Step 14: Expand `README.md` — Full Project Documentation ✅ Complete

**Current state:** Single paragraph, no instructions.

**Implemented:** Complete rewrite with all 8 sections: project title & description, architecture overview table (4 stages), setup instructions (venv + pip + dataset download), training commands for all 4 stages, evaluation commands with all flags documented, project structure tree, testing section, hardware notes with device priority and recommended batch sizes, and quantitative metrics table.

---

## Dependency Graph

```
Step 1 (fix unet import) ──┐
Step 2 (fix dataset import) ├── Step 4 (clean discriminator)
Step 3 (add __init__.py) ───┘        │
                                     ├── Step 5 (common.py)
                                     ├── Step 6 (metrics.py)
                                     ├── Step 7 (GANLoss to src/losses)
                                     │        │
                                     │        ├── Step 8 (train_fusion.py)
                                     │        │
                                     ├── Step 9 (evaluate.py)
                                     │        │
                                     ├── Step 10 (test samples)
                                     │        │
                                     ├────────┼── Step 11 (train GAN)
                                     │        ├── Step 12 (train Fusion)
                                     │        │
                                     ├── Step 13 (expand tests)
                                     └── Step 14 (README.md)
```

Steps 1–3 must be done first (they unblock everything).  
Steps 5, 6, 7 can be done in parallel.  
Steps 8 and 9 depend on Steps 5–7.  
Steps 11–12 depend on Steps 8–9 and require GPU time.  
Steps 13–14 can be done at any point after Step 7.

---

## Verification Checklist

After all steps are complete, the following must all pass:

- [x] `python -m pytest tests/ -v` — all tests green
- [ ] `python scripts/trains/train_baseline.py --epochs 1 --batch_size 2` — runs without error
- [ ] `python scripts/trains/train_unet.py --epochs 1 --batch_size 2` — runs without error
- [ ] `python scripts/trains/train_gan.py --epochs 1 --batch_size 2` — runs without error
- [ ] `python scripts/trains/train_fusion.py --epochs 1 --batch_size 2` — runs without error
- [ ] `python scripts/evaluate.py --model baseline --checkpoint ./checkpoints/baseline_cnn_final.pth --img_path ./data/test_samples` — produces images in `outputs/images/` and prints metrics
- [ ] `python scripts/evaluate.py --model unet --checkpoint ./outputs/checkpoints/unet_final.pth --img_path ./data/test_samples` — same
- [ ] `python scripts/evaluate.py --model gan --checkpoint ./outputs/checkpoints/gan_generator_final.pth --img_path ./data/test_samples` — same (after Step 11)
- [ ] `python scripts/evaluate.py --model fusion --checkpoint ./outputs/checkpoints/fusion_generator_final.pth --img_path ./data/test_samples` — same (after Step 12)
- [ ] `outputs/images/` contains comparison strips for all 4 models
- [x] All imports consistently use `from src.models.u_net` matching the actual filename `u_net.py`
- [x] All imports consistently use `from src.utils.dataset` matching the actual file location
- [x] `src/utils/__init__.py` exists (created)

---

## Optional Enhancements (Post-Thesis-Minimum)

All 7 enhancements have been implemented. ✅

1. ✅ **Consolidate checkpoint paths** — All training scripts now default `--save_dir` to `./outputs/checkpoints/`. README updated. Legacy `./checkpoints/` directory kept for reference only.
2. ✅ **Add `--resume` flag** — `--resume` (baseline/U-Net) and `--resume_g` / `--resume_d` (GAN/Fusion) added to all training scripts. Epoch is parsed from the filename and the LR scheduler fast-forwards to the correct position.
3. ✅ **TensorBoard logging** — `SummaryWriter` added to all 4 training scripts (step-level and epoch-level scalars). Logs written to `./outputs/runs/{baseline,unet,gan,fusion}/`. Gracefully skipped if `tensorboard` is not installed.
4. ✅ **Batch evaluation table** — `evaluate.py` now prints an aligned table with `─` dividers, per-image rows (Image / PSNR / SSIM / LPIPS), and an average row at the bottom.
5. ✅ **Unified `train.py`** — `scripts/trains/train.py` is a thin dispatcher: `--model {baseline,unet,gan,fusion}` routes to the matching trainer function. Supports all shared + model-specific flags (e.g. `--lambda_l1`, `--resume_g`, `--resume_d`).
6. ✅ **Configurable λ + LR schedules** — `--lambda_l1` flag added (default 100.0, used everywhere `GAN_LAMBDA_L1`/`FUSION_LAMBDA_L1` was hard-coded). LR decay: `StepLR` (halves twice) for baseline/U-Net; linear `LambdaLR` decay over the second half of training for GAN/Fusion.
7. ✅ **Docker/devcontainer** — `Dockerfile` (Python 3.11-slim, CPU PyTorch, OpenCV system deps, port 6006) and `.devcontainer/devcontainer.json` (VS Code extensions, TensorBoard port forwarding, data directory mount) added.
