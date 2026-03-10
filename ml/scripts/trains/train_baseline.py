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

# Resolve paths relative to project root (3 levels up from scripts/trains/)
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ML_ROOT      = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ML_ROOT, '..'))
sys.path.append(_ML_ROOT)

# ── Apple Silicon MPS accelerator settings ─────────────────────────────────
# Allow MPS ops that aren't natively supported to fall back to CPU silently
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Remove 60% unified-memory cap so MPS can use all available GPU memory
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from src.models.baseline_cnn import BaselineCNN
from src.utils.dataset import ColorizationDataset
from src.utils.common import lab_to_rgb
from src.utils.metrics import compute_psnr, compute_ssim


def get_args():
    parser = argparse.ArgumentParser(description="Train Baseline CNN (Stage 1)")
    parser.add_argument("--epochs",     type=int,   default=20,                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=16,                      help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,                    help="Initial learning rate")
    parser.add_argument("--data_path",  type=str,   default=os.path.join(_PROJECT_ROOT, "data", "coco", "val2017"),   help="Path to training images")
    parser.add_argument("--save_dir",   type=str,   default=os.path.join(_PROJECT_ROOT, "outputs", "checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--resume",     type=str,   default=None,                    help="Path to a .pth checkpoint to resume from")
    parser.add_argument("--log_dir",    type=str,   default=os.path.join(_PROJECT_ROOT, "outputs", "runs"),        help="TensorBoard log directory")
    parser.add_argument("--patience",   type=int,   default=7,                       help="Early stopping: epochs without improvement before stopping (0 = disabled)")
    parser.add_argument("--val_ratio",    type=float, default=0.1,   help="Fraction of data reserved for validation")
    parser.add_argument("--val_every",    type=int,   default=5,    help="Run validation every N epochs")
    parser.add_argument("--num_samples",  type=int,   default=None, help="Cap total images used (e.g. 1000). Useful to reduce training time.")
    return parser.parse_args()

def train_baseline(args):
    # ── Device ─────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- TRAINING BASELINE CNN on {device} ---")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── TensorBoard ────────────────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "baseline"))
        print(f"TensorBoard logs: {os.path.join(args.log_dir, 'baseline')}")
    except ImportError:
        writer = None

    # ── Data (with train/val split) ────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
    ])
    val_transform = transforms.Compose([transforms.Resize((256, 256))])
    try:
        train_dataset = ColorizationDataset(args.data_path, mode='train',
                                            transform=transform, val_ratio=args.val_ratio,
                                            num_samples=args.num_samples)
        val_dataset   = ColorizationDataset(args.data_path, mode='val',
                                            transform=val_transform, val_ratio=args.val_ratio,
                                            num_samples=args.num_samples)
        n_workers = min(8, os.cpu_count() or 1)
        loader   = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=n_workers, persistent_workers=n_workers > 0,
                              prefetch_factor=2 if n_workers > 0 else None)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=n_workers, persistent_workers=n_workers > 0,
                                prefetch_factor=2 if n_workers > 0 else None)
        print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    model     = BaselineCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Step-decay LR: halve twice over the full training run
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        m = re.search(r'epoch_(\d+)', os.path.basename(args.resume))
        start_epoch = int(m.group(1)) if m else 0
        global_step = start_epoch * len(loader)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, continuing from epoch {start_epoch + 1}")
    elif args.resume:
        print(f"Warning: resume checkpoint not found: {args.resume}")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_loss      = float('inf')
    no_improve     = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loop         = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0

        for batch in loop:
            L  = batch['L'].to(device)
            ab = batch['ab'].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                 enabled=(device.type == "mps")):
                outputs = model(L)
                loss    = criterion(outputs, ab)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.5f}")
            if writer:
                writer.add_scalar("Loss/step", loss.item(), global_step)
            global_step += 1

        avg_loss   = running_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"  Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")
        if writer:
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
            writer.add_scalar("LR",         current_lr, epoch)

        # ── Validation-based best checkpoint ────────────────────────────────────
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            import numpy as np
            model.eval()
            val_psnr_list = []
            with torch.no_grad():
                for vb in val_loader:
                    vL  = vb['L'].to(device)
                    vab = vb['ab'].to(device)
                    vpred = model(vL)
                    for i in range(vL.size(0)):
                        pred_rgb = lab_to_rgb(vL[i], vpred[i])
                        gt_rgb   = lab_to_rgb(vL[i], vab[i])
                        val_psnr_list.append(compute_psnr(pred_rgb, gt_rgb))
            model.train()
            val_psnr = float(np.mean(val_psnr_list))
            print(f"  VAL PSNR: {val_psnr:.2f} dB")
            if writer:
                writer.add_scalar("Val/PSNR", val_psnr, epoch)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            best_ckpt  = os.path.join(args.save_dir, "baseline_cnn_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ** New best loss {best_loss:.5f} — saved {best_ckpt}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{args.patience} epochs")

        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(args.save_dir, f"baseline_cnn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved: {ckpt}")

        # ── Early stopping ─────────────────────────────────────────────────────
        if args.patience > 0 and no_improve >= args.patience:
            print(f"\n  Early stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {args.patience} consecutive epochs).")
            break

    # ── Final save ─────────────────────────────────────────────────────────────
    final = os.path.join(args.save_dir, "baseline_cnn_final.pth")
    torch.save(model.state_dict(), final)
    print(f"Baseline Training Finished! Final model: {final}")
    if writer:
        writer.close()


if __name__ == "__main__":
    args = get_args()
    train_baseline(args)