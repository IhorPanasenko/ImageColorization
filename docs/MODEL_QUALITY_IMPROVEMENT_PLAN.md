# Model Quality Improvement Plan

This document describes four actionable improvements to raise model PSNR/SSIM/LPIPS
beyond the current ~21–26 dB range.  They are listed from highest to lowest expected
impact, and can be pursued independently.

---

## 1. Expand the Training Dataset

**Current state:** ~4,050 training images from COCO val2017 (5,000 total, 10% held
out for validation).  This is two orders of magnitude smaller than the datasets used
by published colorization methods (Zhang 2016 used ImageNet: 1.28 M images).

**Plan:**
1. Download **COCO train2017** (118 K images, ~18 GB).
   ```bash
   wget http://images.cocodataset.org/zips/train2017.zip -P data/coco/
   unzip data/coco/train2017.zip -d data/coco/
   ```
2. Update `--data_path` in every training script to point at
   `data/coco/train2017` (or merge both sets).
3. Optionally supplement with **ImageNet** (or a 100 K-image subset via
   `torchvision.datasets.ImageNet`) for greater scene diversity.
4. Increase augmentation in all `transform` blocks:
   ```python
   transforms.Compose([
       transforms.Resize((286, 286)),
       transforms.RandomCrop(256),
       transforms.RandomHorizontalFlip(),
       transforms.ColorJitter(brightness=0.2, contrast=0.2),  # L-channel robustness
   ])
   ```
5. Retrain all models with the larger dataset and the same hyperparameters.

**Expected gain:** +3–5 dB PSNR based on published ablations for similar
architectures.  The biggest beneficiary is UNet/Fusion because skip connections can
only generalise over texture patterns they've seen during training.

---

## 2. Unfreeze and Fine-tune GlobalHintNet on L-channel Input

**Current state:** `GlobalHintNet` wraps a pre-trained ImageNet ResNet18 whose first
layer receives the L channel replicated to 3 channels.  The network is permanently
frozen (`hint_net.eval()`, no gradient updates).  Because ResNet18 was trained on
colour RGB images, its early filters are tuned for colour differences — not
luminance-only textures.  The global hint it produces at inference is therefore a
luminance-semantic embedding, not a colour-semantic one.

**Plan:**
1. **Unfreeze** `GlobalHintNet` in `train_fusion.py`:
   ```python
   # Remove these lines / change eval() to train():
   # hint_net.eval()                          ← remove freeze
   hint_net.train()
   optimizer_hint = optim.Adam(
       hint_net.parameters(), lr=args.lr * 0.1   # lower LR for fine-tuning
   )
   ```
2. Update the training loop to step `optimizer_hint` together with
   `optimizer_G`:
   ```python
   optimizer_hint.zero_grad()
   # ... existing generator backward pass ...
   optimizer_hint.step()
   ```
3. Save `hint_net` state alongside the generator in every checkpoint:
   ```python
   torch.save({
       'model_state_dict': net_G.state_dict(),
       'hint_net_state_dict': hint_net.state_dict(),
       ...
   }, checkpoint_path)
   ```
4. Update `colorizer.py` `_build_model` and `_load_model` to load
   `hint_net_state_dict` when present (with a fallback to the default
   ImageNet weights for old checkpoints).

**Why it helps:** A fine-tuned hint network learns to extract colour-relevant
features *from the L channel itself* — e.g., sky luminance → blue, foliage → green.
This is the core insight behind "automatic colorization with learned colour priors"
(Larsson et al., 2016).

---

## 3. Improve BaselineCNN Architecture

**Current state:** `BaselineCNN` is a minimal 3-encoder / 3-decoder network with no
skip connections.  It has ~180 K parameters.  It consistently produces the worst
numbers (21–22 dB) because its bottleneck (4×4 spatial at 256-px input) loses all
fine spatial detail.

**Plan (two options — pick one):**

### Option A — Add skip connections (minimal change)
Modify `ml/src/models/baseline_cnn.py` to concatenate encoder feature maps to the
corresponding decoder layers (i.e., promote it to a shallow U-Net):
```python
class BaselineCNN(nn.Module):
    def forward(self, x):
        e1 = self.enc1(x)      # (B, 64,  128, 128)
        e2 = self.enc2(e1)     # (B, 128,  64,  64)
        e3 = self.enc3(e2)     # (B, 256,  32,  32)
        d1 = self.dec1(e3)                    # (B, 128, 64, 64)
        d2 = self.dec2(torch.cat([d1, e2], 1))  # skip from enc2
        d3 = self.dec3(torch.cat([d2, e1], 1))  # skip from enc1
        return self.output(d3)
```
Update decoder input channels accordingly (128+128, 64+64).
This requires retraining only `baseline_cnn`.

### Option B — Accept it as a deliberate weak baseline
Keep `BaselineCNN` unchanged as a controlled "no skip connections" comparison
point for the thesis, and shift focus to UNet / Fusion improvements.
This requires no code changes.

---

## 4. Use Perceptual Metrics as the Primary Quality Signal

**Current state:** PSNR is the headline metric, but colorization is a **one-to-many
problem**: grass can plausibly be green, yellow, or brown — all correct answers, but
none will exactly match the original pixel values.  PSNR penalises every deviation
regardless of whether it is perceptually acceptable.  This creates a structural
ceiling of ~24–27 dB even for visually excellent results.

**Plan:**

### 4a — Elevate LPIPS in the UI and reports
Treat **LPIPS** (lower = better) and **SSIM** as the primary quality indicators in
the thesis.  LPIPS (AlexNet perceptual distance) correlates far better with human
judgement for colorization than PSNR does.  Update the benchmark ranking weights in
`RankingTable.vue`:
```typescript
// Current:  0.35 PSNR + 0.35 SSIM + 0.20 (1-LPIPS) + 0.10 Speed
// Proposed: 0.20 PSNR + 0.35 SSIM + 0.35 (1-LPIPS) + 0.10 Speed
```

### 4b — Add a heavier perceptual loss during training
`train_fusion.py` already supports `--lambda_perceptual` (default 10).  Increase it:
```bash
python ml/scripts/trains/train_fusion.py \
    --lambda_perceptual 20 \
    --lambda_l1 75
```
Reducing L1 weight slightly prevents the model from regressing to the grey mean
(which minimises L1 but produces low-saturation, washed-out colours).

### 4c — Add colour histogram loss
Implement a histogram loss that penalises global colour distribution mismatch rather
than per-pixel error.  A lightweight version:
```python
def histogram_loss(pred_ab: torch.Tensor, real_ab: torch.Tensor, bins: int = 32) -> torch.Tensor:
    # Normalised histogram over ab channels, [−1, 1] range
    loss = 0.0
    for c in range(2):
        p_hist = torch.histc(pred_ab[:, c], bins=bins, min=-1, max=1) / pred_ab.numel()
        r_hist = torch.histc(real_ab[:, c], bins=bins, min=-1, max=1) / real_ab.numel()
        loss += torch.mean((p_hist - r_hist) ** 2)
    return loss
```
Add to `ml/src/losses/__init__.py` and register in `train_fusion.py` with a small
weight (e.g. `lambda_histogram=5`).

---

## Priority Order

| # | Improvement | Effort | Expected PSNR gain | Notes |
|---|---|---|---|---|
| 1 | Expand dataset (COCO train2017) | Medium (download + retrain) | **+3–5 dB** | Highest ROI |
| 2 | Unfreeze GlobalHintNet | Medium (code + retrain fusion only) | **+1–2 dB** | Fusion model only |
| 3 | BaselineCNN skip connections | Low (code + retrain baseline only) | **+1–2 dB** | Or keep as-is for thesis contrast |
| 4 | Perceptual-first metrics + losses | Low (UI weights, no retrain needed for 4a) | Visual quality ↑ | PSNR ceiling is structural |

Improvements 1 and 2 require retraining the affected models.
Improvement 4a (UI ranking weights) requires no retraining.
