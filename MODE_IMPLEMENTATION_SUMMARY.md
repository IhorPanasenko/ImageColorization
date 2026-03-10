# Mode Parameter Implementation Summary

## Overview
Implemented proper handling of the `mode` parameter in the colorization pipeline to differentiate between:
1. **Grayscale mode**: Colorizing black & white images (no ground truth)
2. **Color photo mode**: Re-colorizing color photos with metrics comparison

## Changes Made

### 1. Backend: `api/services/colorizer.py`
**Location**: `Colorizer.colorize()` method

**Change**: Added conditional logic to compute metrics only in `color_photo` mode:

```python
# Compute metrics only in color_photo mode
# (grayscale mode has no ground truth, metrics would be meaningless)
if mode == 'color_photo':
    try:
        result['ground_truth'] = _img_to_b64(gt_rgb)
        result['metrics']['psnr']  = float(compute_psnr(pred_rgb, gt_rgb))
        result['metrics']['ssim']  = float(compute_ssim(pred_rgb, gt_rgb))
        result['metrics']['lpips'] = float(compute_lpips(pred_rgb, gt_rgb, device=str(device)))
    except Exception:
        pass
```

**Behavior**:
- **Grayscale mode**: 
  - Metrics remain `null`
  - No `ground_truth` field in response
  - Appropriate for truly B&W images where there's no color reference
  
- **Color photo mode**: 
  - Computes PSNR, SSIM, and LPIPS metrics
  - Includes `ground_truth` field (original color)
  - Allows comparison of model prediction vs original colors

### 2. Frontend: `frontend/src/types/index.ts`
**Change**: Updated `ColorizeResult` interface to:
- Mark `ground_truth` as optional with clear comment
- Add `lpips` metric (was missing from type definition)

```typescript
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
```

### 3. Frontend: `frontend/src/pages/ColorizePage.vue`
**Status**: Already properly implemented!
- Conditional metrics display (only shows when not null)
- Different panel layouts for each mode
- Mode selector with clear descriptions

## How It Works

### Request Flow
1. User selects mode in UI (grayscale or color_photo)
2. Frontend sends image + mode to `/api/colorize`
3. Backend always:
   - Extracts L channel from input
   - Models predict ab channels
   - Converts LAB → RGB
4. Backend conditionally (based on mode):
   - Computes metrics vs original (color_photo only)
   - Includes ground_truth field (color_photo only)

### Image Processing Pipeline
Both modes use identical colorization:
```
Input → L channel extraction → Model prediction (ab) → LAB→RGB conversion
```

The difference is only in **post-processing**:
- Grayscale: Return colorized result only
- Color photo: Return colorized result + metrics vs original

## Testing

### Test Data Created
- `data/test_samples/test_grayscale.jpg` - Simple grayscale test image

### Manual Testing Steps

#### Test Grayscale Mode
1. Open http://localhost:5173/colorize
2. Select "Grayscale → Colour" mode
3. Upload `data/test_samples/test_grayscale.jpg`
4. Select a checkpoint (e.g., `fusion_generator_best.pth`)
5. Click "Colorize"
6. **Expected**: 
   - 2 panels: Input grayscale | Colorized
   - No metrics displayed
   - No ground truth comparison

#### Test Color Photo Mode
1. Open http://localhost:5173/colorize
2. Select "Colour → Re-colour" mode
3. Upload `data/test_samples/000000000139.jpg` (color image)
4. Select a checkpoint
5. Click "Colorize"
6. **Expected**: 
   - 3 panels: Original colour | Extracted grayscale | Re-coloured
   - Metrics displayed (PSNR, SSIM)
   - Quality comparison visible

## API Endpoint

**Endpoint**: `POST /api/colorize`

**Form Fields**:
- `file`: Image file
- `model`: "baseline" | "unet" | "gan" | "fusion"
- `checkpoint`: Path to .pth checkpoint
- `mode`: "grayscale" | "color_photo"

**Response** (grayscale mode):
```json
{
  "colorized": "base64...",
  "grayscale": "base64...",
  "original": "base64...",
  "metrics": {
    "psnr": null,
    "ssim": null,
    "lpips": null
  }
}
```

**Response** (color_photo mode):
```json
{
  "colorized": "base64...",
  "grayscale": "base64...",
  "original": "base64...",
  "ground_truth": "base64...",
  "metrics": {
    "psnr": 28.45,
    "ssim": 0.8234,
    "lpips": 0.123
  }
}
```

## Benefits
1. **Semantic clarity**: Mode explicitly defines intent
2. **Accurate metrics**: Only computed when meaningful
3. **Better UX**: Shows appropriate comparisons per use case
4. **Reduced confusion**: Clear distinction between "colorizing" vs "re-colorizing"

## Backward Compatibility
✅ Fully backward compatible:
- Default mode is `grayscale` (existing behavior)
- Frontend already had mode selector
- Type changes are additive (optional fields)
