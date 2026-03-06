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

from src.models.baseline_cnn import BaselineCNN
from src.utils.dataset import ColorizationDataset


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

    # ── Data ───────────────────────────────────────────────────────────────────
    transform = transforms.Compose([transforms.Resize((256, 256))])
    try:
        dataset = ColorizationDataset(args.data_path, transform=transform)
        n_workers = min(4, os.cpu_count() or 1)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=n_workers, persistent_workers=n_workers > 0,
                             prefetch_factor=2 if n_workers > 0 else None)
        print(f"Data loaded: {len(dataset)} images")
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

        # ── Best checkpoint ────────────────────────────────────────────────────
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