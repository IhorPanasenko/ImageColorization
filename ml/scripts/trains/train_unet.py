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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.u_net import UNet
from src.utils.dataset import ColorizationDataset


def get_args():
    parser = argparse.ArgumentParser(description="Train U-Net (Stage 2)")
    parser.add_argument("--epochs",     type=int,   default=20,                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=16,                      help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-4,                    help="Initial learning rate")
    parser.add_argument("--data_path",  type=str,   default="./data/coco/val2017",   help="Path to training images")
    parser.add_argument("--save_dir",   type=str,   default="./outputs/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume",     type=str,   default=None,                    help="Path to a .pth checkpoint to resume from")
    parser.add_argument("--log_dir",    type=str,   default="./outputs/runs",        help="TensorBoard log directory")
    return parser.parse_args()

def train_unet(args):
    # ── Device ─────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- TRAINING U-NET on {device} ---")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── TensorBoard ────────────────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "unet"))
        print(f"TensorBoard logs: {os.path.join(args.log_dir, 'unet')}")
    except ImportError:
        writer = None

    # ── Data ───────────────────────────────────────────────────────────────────
    transform = transforms.Compose([transforms.Resize((256, 256))])
    try:
        dataset = ColorizationDataset(args.data_path, transform=transform)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"Data loaded: {len(dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    model     = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
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

        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(args.save_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved: {ckpt}")

    # ── Final save ─────────────────────────────────────────────────────────────
    final = os.path.join(args.save_dir, "unet_final.pth")
    torch.save(model.state_dict(), final)
    print(f"U-Net Training Finished! Final model: {final}")
    if writer:
        writer.close()


if __name__ == "__main__":
    args = get_args()
    train_unet(args)