import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- FIX IMPORTS ---
# Додаємо корінь проекту в sys.path, щоб бачити пакет src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.baseline_cnn import BaselineCNN
from models.u_net import UNet
from utils.dataset import ColorizationDataset

# --- CONFIGURATION VIA ARGS ---
def get_args():
    parser = argparse.ArgumentParser(description="Train Colorization Models")
    parser.add_argument("--model", type=str, default="unet", choices=["baseline", "unet", "gan"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--data_path", type=str, default="./data/coco/val2017", help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints", help="Where to save models")
    return parser.parse_args()

def train(args):
    # 1. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | Model: {args.model}")

    # 2. Data
    os.makedirs(args.save_dir, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((256, 256))])
    dataset = ColorizationDataset(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Model & Loss Selection
    if args.model == "baseline":
        model = BaselineCNN().to(device)
        criterion = nn.MSELoss()
    elif args.model == "unet":
        model = UNet().to(device)
        criterion = nn.L1Loss()
    # Тут потім додамо GAN
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0

        for batch in loop:
            L = batch['L'].to(device)
            ab = batch['ab'].to(device)

            optimizer.zero_grad()
            outputs = model(L)
            loss = criterion(outputs, ab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Save Checkpoint (тільки останній та кожні 5)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_epoch_{epoch+1}.pth"))

    # Final Save
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_final.pth"))
    print("Done!")

if __name__ == "__main__":
    args = get_args()
    train(args)