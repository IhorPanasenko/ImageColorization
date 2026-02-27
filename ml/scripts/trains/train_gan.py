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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.u_net import UNet
from src.models.discriminator import PatchDiscriminator
from src.utils.dataset import ColorizationDataset
from src.losses import GANLoss


def get_args():
    parser = argparse.ArgumentParser(description="Train Pix2Pix GAN (Stage 3)")
    parser.add_argument("--epochs",     type=int,   default=20,                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,                       help="Batch size (two models in memory)")
    parser.add_argument("--lr",         type=float, default=2e-4,                    help="Initial learning rate for G and D")
    parser.add_argument("--lambda_l1",  type=float, default=100.0,                   help="Weight of L1 loss relative to GAN loss")
    parser.add_argument("--data_path",  type=str,   default="./data/coco/val2017",   help="Path to training images")
    parser.add_argument("--save_dir",   type=str,   default="./outputs/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_g",   type=str,   default=None,                    help="Generator checkpoint to resume from")
    parser.add_argument("--resume_d",   type=str,   default=None,                    help="Discriminator checkpoint to resume from")
    parser.add_argument("--log_dir",    type=str,   default="./outputs/runs",        help="TensorBoard log directory")
    return parser.parse_args()


def _load_if_exists(model, path, device):
    """Load a state dict into model if checkpoint path exists; return parsed epoch."""
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        m = re.search(r'epoch_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else 0
    elif path:
        print(f"Warning: checkpoint not found: {path}")
    return 0


def train_gan(args):
    # ── Device ─────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- STARTING GAN TRAINING on {device} ---")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── TensorBoard ────────────────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "gan"))
        print(f"TensorBoard logs: {os.path.join(args.log_dir, 'gan')}")
    except ImportError:
        writer = None

    # ── Data ───────────────────────────────────────────────────────────────────
    transform = transforms.Compose([transforms.Resize((256, 256))])
    dataset   = ColorizationDataset(args.data_path, transform=transform)
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Data loaded: {len(dataset)} images")

    # ── Models ──────────────────────────────────────────────────────────────────
    print("Initializing models...")
    net_G = UNet().to(device)
    net_D = PatchDiscriminator().to(device)

    criterion_GAN = GANLoss()
    criterion_L1  = nn.L1Loss()

    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Linear LR decay starting at the halfway point
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
    global_step = start_epoch * len(loader)
    for _ in range(start_epoch):
        scheduler_G.step()
        scheduler_D.step()
    if start_epoch:
        print(f"Resumed from epoch {start_epoch}, continuing from epoch {start_epoch + 1}")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        net_G.train()
        net_D.train()
        loop         = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        sum_D, sum_G = 0.0, 0.0

        for batch in loop:
            real_L  = batch['L'].to(device)
            real_ab = batch['ab'].to(device)

            # ── Step 1: Discriminator ──────────────────────────────────────────
            optimizer_D.zero_grad()
            fake_ab      = net_G(real_L)
            pred_real    = net_D(real_L, real_ab)
            pred_fake    = net_D(real_L, fake_ab.detach())
            loss_D_real  = criterion_GAN(pred_real, target_is_real=True)
            loss_D_fake  = criterion_GAN(pred_fake, target_is_real=False)
            loss_D       = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ── Step 2: Generator ─────────────────────────────────────────────
            optimizer_G.zero_grad()
            pred_fake_for_G = net_D(real_L, fake_ab)
            loss_G_GAN      = criterion_GAN(pred_fake_for_G, target_is_real=True)
            loss_G_L1       = criterion_L1(fake_ab, real_ab) * args.lambda_l1
            loss_G          = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            sum_D += loss_D.item()
            sum_G += loss_G.item()
            loop.set_postfix(D=f"{loss_D.item():.4f}", G=f"{loss_G.item():.4f}")
            if writer:
                writer.add_scalar("Loss_D/step", loss_D.item(), global_step)
                writer.add_scalar("Loss_G/step", loss_G.item(), global_step)
            global_step += 1

        n            = len(loader)
        current_lr_g = optimizer_G.param_groups[0]["lr"]
        scheduler_G.step()
        scheduler_D.step()
        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Avg D: {sum_D/n:.4f} | Avg G: {sum_G/n:.4f} | LR: {current_lr_g:.6f}")
        if writer:
            writer.add_scalar("Loss_D/epoch", sum_D / n, epoch)
            writer.add_scalar("Loss_G/epoch", sum_G / n, epoch)
            writer.add_scalar("LR",           current_lr_g, epoch)

        if (epoch + 1) % 5 == 0:
            torch.save(net_G.state_dict(),
                       os.path.join(args.save_dir, f"gan_generator_epoch_{epoch+1}.pth"))
            torch.save(net_D.state_dict(),
                       os.path.join(args.save_dir, f"gan_discriminator_epoch_{epoch+1}.pth"))
            print(f"  Checkpoints saved at epoch {epoch+1}")

    # ── Final save ─────────────────────────────────────────────────────────────
    torch.save(net_G.state_dict(), os.path.join(args.save_dir, "gan_generator_final.pth"))
    torch.save(net_D.state_dict(), os.path.join(args.save_dir, "gan_discriminator_final.pth"))
    print("GAN Training Finished!")
    if writer:
        writer.close()


if __name__ == "__main__":
    args = get_args()
    train_gan(args)