"""
Train Fusion GAN (Stage 4) — improved version.

Changes vs original:
  - VGG perceptual loss (--lambda_perceptual)
  - Discriminator label smoothing (--label_smoothing)
  - Validation split with PSNR-based best checkpoint saving
  - Optional warm-start from pretrained UNetFusion / U-Net (--warmstart_g)
  - Gradient clipping for generator stability
  - Random horizontal flip data augmentation
"""

import sys
import os
import re
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Resolve paths relative to project root (3 levels up from scripts/trains/)
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ML_ROOT      = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ML_ROOT, '..'))
sys.path.append(_ML_ROOT)

# ── Apple Silicon MPS accelerator settings ─────────────────────────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from src.models.unet_fusion import UNetFusion
from src.models.global_hints import GlobalHintNet
from src.models.discriminator import PatchDiscriminator
from src.utils.dataset import ColorizationDataset
from src.utils.common import lab_to_rgb
from src.utils.metrics import compute_psnr, compute_ssim
from src.losses import GANLoss, PerceptualLoss, ab_to_pseudo_rgb


def get_args():
    parser = argparse.ArgumentParser(description="Train Fusion GAN (Stage 4) — Improved")
    # Core
    parser.add_argument("--epochs",              type=int,   default=50,      help="Number of training epochs")
    parser.add_argument("--batch_size",          type=int,   default=8,       help="Batch size (use ≤8 for MPS)")
    parser.add_argument("--lr",                  type=float, default=2e-4,    help="Learning rate (used for both G and D unless overridden)")
    parser.add_argument("--lr_g",                type=float, default=None,    help="Generator learning rate (overrides --lr)")
    parser.add_argument("--lr_d",                type=float, default=None,    help="Discriminator learning rate (overrides --lr)")
    # Loss weights
    parser.add_argument("--lambda_l1",           type=float, default=100.0,   help="Weight of L1 loss")
    parser.add_argument("--lambda_perceptual",   type=float, default=10.0,    help="Weight of VGG perceptual loss (0 = disable)")
    parser.add_argument("--label_smoothing",     type=float, default=0.1,     help="Label smoothing for discriminator (0 = disable)")
    # Paths
    parser.add_argument("--data_path",  type=str, default=os.path.join(_PROJECT_ROOT, "data", "coco", "val2017"))
    parser.add_argument("--save_dir",   type=str, default=os.path.join(_PROJECT_ROOT, "outputs", "checkpoints"))
    parser.add_argument("--log_dir",    type=str, default=os.path.join(_PROJECT_ROOT, "outputs", "runs"))
    # Resume / warm-start
    parser.add_argument("--resume_g",    type=str, default=None, help="Generator checkpoint to resume from")
    parser.add_argument("--resume_d",    type=str, default=None, help="Discriminator checkpoint to resume from")
    parser.add_argument("--warmstart_g", type=str, default=None,
                        help="Pretrained UNetFusion checkpoint to warm-start generator (loads matching keys only)")
    # Validation
    parser.add_argument("--val_ratio",   type=float, default=0.1,   help="Fraction of data reserved for validation")
    parser.add_argument("--val_every",   type=int,   default=5,    help="Run validation every N epochs")
    parser.add_argument("--num_samples", type=int,   default=None, help="Cap total images used (e.g. 1000). Useful to reduce training time.")
    args = parser.parse_args()
    if args.lr_g is None:
        args.lr_g = args.lr
    if args.lr_d is None:
        args.lr_d = args.lr
    return args


def _load_if_exists(model, path, device):
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        m = re.search(r'epoch_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else 0
    elif path:
        print(f"Warning: checkpoint not found: {path}")
    return 0


@torch.no_grad()
def validate(net_G, hint_net, val_loader, device):
    """Compute average PSNR and SSIM on the validation set."""
    net_G.eval()
    psnr_list, ssim_list = [], []
    for batch in val_loader:
        real_L  = batch['L'].to(device)
        real_ab = batch['ab'].to(device)
        global_hint = hint_net(real_L)
        fake_ab = net_G(real_L, global_hint)
        for i in range(real_L.size(0)):
            pred_rgb = lab_to_rgb(real_L[i], fake_ab[i])
            gt_rgb   = lab_to_rgb(real_L[i], real_ab[i])
            psnr_list.append(compute_psnr(pred_rgb, gt_rgb))
            ssim_list.append(compute_ssim(pred_rgb, gt_rgb))
    net_G.train()
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def train_fusion(args):
    # ── Device ─────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- STARTING IMPROVED FUSION GAN TRAINING on {device} ---")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── TensorBoard ────────────────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "fusion"))
        print(f"TensorBoard logs: {os.path.join(args.log_dir, 'fusion')}")
    except ImportError:
        writer = None

    # ── Data (with train/val split) ────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
    ])
    val_transform = transforms.Compose([transforms.Resize((256, 256))])

    train_dataset = ColorizationDataset(args.data_path, mode='train',
                                        transform=transform, val_ratio=args.val_ratio,
                                        num_samples=args.num_samples)
    val_dataset   = ColorizationDataset(args.data_path, mode='val',
                                        transform=val_transform, val_ratio=args.val_ratio,
                                        num_samples=args.num_samples)
    n_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=n_workers, persistent_workers=n_workers > 0,
                              prefetch_factor=2 if n_workers > 0 else None)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=n_workers, persistent_workers=n_workers > 0,
                              prefetch_factor=2 if n_workers > 0 else None)
    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # ── Models ─────────────────────────────────────────────────────────────────
    hint_net = GlobalHintNet().to(device)
    hint_net.eval()  # frozen — no optimizer
    net_G = UNetFusion().to(device)
    net_D = PatchDiscriminator().to(device)

    # Optional warm-start from pretrained checkpoint (matching keys only)
    if args.warmstart_g and os.path.exists(args.warmstart_g):
        print(f"Warm-starting generator from: {args.warmstart_g}")
        ws_state = torch.load(args.warmstart_g, map_location=device)
        if isinstance(ws_state, dict) and 'model_state_dict' in ws_state:
            ws_state = ws_state['model_state_dict']
        # Load only keys that match (allows partial warm-start from plain U-Net)
        model_dict = net_G.state_dict()
        ws_filtered = {k: v for k, v in ws_state.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(ws_filtered)
        net_G.load_state_dict(model_dict)
        print(f"  Loaded {len(ws_filtered)}/{len(model_dict)} weight tensors")

    # ── Losses ──────────────────────────────────────────────────────────────────
    criterion_GAN = GANLoss(label_smoothing=args.label_smoothing)
    criterion_L1  = nn.L1Loss()
    criterion_perceptual = None
    if args.lambda_perceptual > 0:
        criterion_perceptual = PerceptualLoss().to(device)
        criterion_perceptual.eval()
        print(f"VGG perceptual loss enabled (weight={args.lambda_perceptual})")

    # ── Optimizers ─────────────────────────────────────────────────────────────
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    def lr_lambda(epoch):
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, args.epochs - decay_start))

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)

    # ── Resume ─────────────────────────────────────────────────────────────────
    epoch_g     = _load_if_exists(net_G, args.resume_g, device)
    epoch_d     = _load_if_exists(net_D, args.resume_d, device)
    start_epoch = max(epoch_g, epoch_d)
    global_step = start_epoch * len(train_loader)
    for _ in range(start_epoch):
        scheduler_G.step()
        scheduler_D.step()
    if start_epoch:
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_psnr = -float('inf')

    for epoch in range(start_epoch, args.epochs):
        net_G.train()
        net_D.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        sum_D, sum_G, sum_perc = 0.0, 0.0, 0.0

        for batch in loop:
            real_L  = batch['L'].to(device)
            real_ab = batch['ab'].to(device)

            _amp = device.type == "mps"

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_amp):
                    global_hint = hint_net(real_L)

            # ── Discriminator step ─────────────────────────────────────────────
            optimizer_D.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_amp):
                fake_ab     = net_G(real_L, global_hint)
                pred_real   = net_D(real_L, real_ab)
                pred_fake   = net_D(real_L, fake_ab.detach())
                loss_D_real = criterion_GAN(pred_real, target_is_real=True)
                loss_D_fake = criterion_GAN(pred_fake, target_is_real=False)
                loss_D      = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ── Generator step ─────────────────────────────────────────────────
            optimizer_G.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=_amp):
                pred_fake_for_G = net_D(real_L, fake_ab)
                loss_G_GAN = criterion_GAN(pred_fake_for_G, target_is_real=True)
                loss_G_L1  = criterion_L1(fake_ab, real_ab) * args.lambda_l1
                loss_G     = loss_G_GAN + loss_G_L1

            # VGG perceptual loss (run in float32 for numerical stability)
            loss_perc = torch.tensor(0.0, device=device)
            if criterion_perceptual is not None:
                pred_pseudo = ab_to_pseudo_rgb(real_L.float(), fake_ab.float())
                real_pseudo = ab_to_pseudo_rgb(real_L.float(), real_ab.float())
                loss_perc = criterion_perceptual(pred_pseudo, real_pseudo) * args.lambda_perceptual
                loss_G = loss_G + loss_perc

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=1.0)
            optimizer_G.step()

            sum_D += loss_D.item()
            sum_G += loss_G.item()
            sum_perc += loss_perc.item()
            loop.set_postfix(D=f"{loss_D.item():.4f}", G=f"{loss_G.item():.4f}")
            if writer:
                writer.add_scalar("Loss_D/step", loss_D.item(), global_step)
                writer.add_scalar("Loss_G/step", loss_G.item(), global_step)
                if args.lambda_perceptual > 0:
                    writer.add_scalar("Loss_Perceptual/step", loss_perc.item(), global_step)
            global_step += 1

        n = len(train_loader)
        current_lr_g = optimizer_G.param_groups[0]["lr"]
        scheduler_G.step()
        scheduler_D.step()
        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Avg D: {sum_D/n:.4f} | Avg G: {sum_G/n:.4f} | "
              f"Avg Perc: {sum_perc/n:.4f} | LR: {current_lr_g:.6f}")
        if writer:
            writer.add_scalar("Loss_D/epoch", sum_D / n, epoch)
            writer.add_scalar("Loss_G/epoch", sum_G / n, epoch)
            writer.add_scalar("LR", current_lr_g, epoch)

        # ── Periodic checkpoint ─────────────────────────────────────────────────
        if (epoch + 1) % 5 == 0:
            torch.save(net_G.state_dict(),
                       os.path.join(args.save_dir, f"fusion_generator_epoch_{epoch+1}.pth"))
            torch.save(net_D.state_dict(),
                       os.path.join(args.save_dir, f"fusion_discriminator_epoch_{epoch+1}.pth"))
            print(f"  Checkpoints saved at epoch {epoch+1}")

        # ── Validation-based best checkpoint ────────────────────────────────────
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_psnr, val_ssim = validate(net_G, hint_net, val_loader, device)
            print(f"  VAL => PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")
            if writer:
                writer.add_scalar("Val/PSNR", val_psnr, epoch)
                writer.add_scalar("Val/SSIM", val_ssim, epoch)
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_ckpt = os.path.join(args.save_dir, "fusion_generator_best.pth")
                torch.save(net_G.state_dict(), best_ckpt)
                print(f"  ** New best val PSNR {best_val_psnr:.2f} — saved {best_ckpt}")

    # ── Final save ─────────────────────────────────────────────────────────────
    torch.save(net_G.state_dict(), os.path.join(args.save_dir, "fusion_generator_final.pth"))
    torch.save(net_D.state_dict(), os.path.join(args.save_dir, "fusion_discriminator_final.pth"))
    print(f"Fusion GAN Training Finished! Best val PSNR: {best_val_psnr:.2f} dB")
    if writer:
        writer.close()


if __name__ == "__main__":
    args = get_args()
    train_fusion(args)
