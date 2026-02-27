import sys
import os
import argparse
import torch

# Resolve src package from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.baseline_cnn import BaselineCNN
from src.models.u_net import UNet
from src.models.unet_fusion import UNetFusion
from src.models.global_hints import GlobalHintNet
from src.utils.common import get_device, prepare_grayscale_input, lab_to_rgb, save_comparison_strip
from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate & compare colorization models")
    parser.add_argument("--model",      type=str, default="gan",
                        choices=["baseline", "unet", "gan", "fusion"],
                        help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/gan_generator_final.pth",
                        help="Path to the .pth weights file")
    parser.add_argument("--img_path",   type=str, default="./data/test_samples",
                        help="Path to a single image file or a directory of images")
    parser.add_argument("--save_dir",   type=str, default="./outputs/images",
                        help="Directory to save comparison strips")
    parser.add_argument("--device",     type=str, default="auto",
                        help="Device override: 'cuda', 'mps', or 'cpu' (default: auto-detect)")
    return parser.parse_args()


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    """
    Load a colorization model from a checkpoint file.

    Returns:
        (model, hint_net) — hint_net is a frozen GlobalHintNet for 'fusion',
                            None for all other model types.
    """
    print(f"--- Loading model: {model_type.upper()} ---")
    print(f"    Checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run the corresponding training script first."
        )

    hint_net = None

    if model_type == "baseline":
        model = BaselineCNN().to(device)
    elif model_type in ("unet", "gan"):
        model = UNet().to(device)
    elif model_type == "fusion":
        model = UNetFusion().to(device)
        hint_net = GlobalHintNet().to(device)
        hint_net.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print("    Model loaded successfully!")
    return model, hint_net


def process_image(img_path: str, model, hint_net, device: torch.device):
    """
    Run inference on a single image.

    Returns:
        pred_rgb  (H, W, 3) float32 [0,1] — model colorization output
        gt_rgb    (H, W, 3) float32 [0,1] — original ground-truth RGB
        gray_rgb  (H, W, 3) float32 [0,1] — grayscale input (3-channel for display)
    """
    L_tensor, gt_rgb = prepare_grayscale_input(img_path, target_size=256)
    L_tensor = L_tensor.to(device)  # (1, 1, 256, 256)

    with torch.no_grad():
        if hint_net is not None:
            global_hint = hint_net(L_tensor)           # (1, 512)
            pred_ab = model(L_tensor, global_hint)     # (1, 2, 256, 256)
        else:
            pred_ab = model(L_tensor)                  # (1, 2, 256, 256)

    pred_rgb = lab_to_rgb(L_tensor[0], pred_ab[0])    # (H, W, 3)

    # Grayscale as 3-channel image for the comparison strip
    import numpy as np
    L_np = L_tensor[0].cpu().squeeze().numpy()         # (H, W) in [0,1]
    gray_rgb = np.stack([L_np] * 3, axis=2)            # (H, W, 3)

    return pred_rgb, gt_rgb, gray_rgb


def evaluate_single(img_path: str, model, hint_net, device: torch.device,
                    save_dir: str, model_name: str) -> dict:
    """
    Evaluate one image: run inference, compute metrics, save comparison strip.

    Returns:
        dict with keys 'psnr', 'ssim', 'lpips'
    """
    pred_rgb, gt_rgb, gray_rgb = process_image(img_path, model, hint_net, device)

    psnr  = compute_psnr(pred_rgb, gt_rgb)
    ssim  = compute_ssim(pred_rgb, gt_rgb)
    lpips = compute_lpips(pred_rgb, gt_rgb, device=str(device))

    stem     = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(save_dir, f"{model_name}_{stem}.png")
    save_comparison_strip(
        gray_rgb, pred_rgb, gt_rgb,
        save_path=out_path,
        title=f"{model_name.upper()} — PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f}",
    )

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


def main():
    args = get_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, hint_net = load_model(args.model, args.checkpoint, device)

    # ── Collect images ────────────────────────────────────────────────────────
    img_path = args.img_path
    if os.path.isdir(img_path):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_paths = sorted([
            os.path.join(img_path, f)
            for f in os.listdir(img_path)
            if f.lower().endswith(exts)
        ])
        if not img_paths:
            raise FileNotFoundError(f"No images found in {img_path}")
    elif os.path.isfile(img_path):
        img_paths = [img_path]
    else:
        raise FileNotFoundError(f"--img_path not found: {img_path}")

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nEvaluating {len(img_paths)} image(s) with model: {args.model.upper()}")
    print(f"{'─' * 72}")
    print(f"  {'Image':<32} {'PSNR (dB)':>10} {'SSIM':>8} {'LPIPS':>8}")
    print(f"{'─' * 72}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_metrics = []
    for path in img_paths:
        metrics = evaluate_single(path, model, hint_net, device, args.save_dir, args.model)
        all_metrics.append(metrics)
        name = os.path.basename(path)
        psnr_str  = f"{metrics['psnr']:>10.2f}" if metrics['psnr'] != float('inf') else f"{'∞':>10}"
        print(f"  {name:<32} {psnr_str} {metrics['ssim']:>8.4f} {metrics['lpips']:>8.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        import numpy as np
        finite_psnr = [m["psnr"] for m in all_metrics if m["psnr"] != float('inf')]
        avg_psnr  = float(np.mean(finite_psnr))  if finite_psnr else float('inf')
        avg_ssim  = float(np.mean([m["ssim"]  for m in all_metrics]))
        avg_lpips = float(np.mean([m["lpips"] for m in all_metrics]))
        print(f"{'─' * 72}")
        print(f"  {'Average (' + str(len(all_metrics)) + ' images)':<32} {avg_psnr:>10.2f} {avg_ssim:>8.4f} {avg_lpips:>8.4f}")
    print(f"{'─' * 72}")

    print(f"\nComparison strips saved to: {args.save_dir}")



if __name__ == "__main__":
    main()
