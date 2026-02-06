import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Додаємо шлях до src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.baseline_cnn import BaselineCNN
from src.dataset import ColorizationDataset

def get_args():
    parser = argparse.ArgumentParser(description="Train Baseline CNN")
    parser.add_argument("--epochs", type=int, default=20, help="Кількість епох")
    parser.add_argument("--batch_size", type=int, default=16, help="Розмір пакету")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate (для Baseline можна більше)")
    parser.add_argument("--data_path", type=str, default="./data/coco/val2017", help="Шлях до даних")
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints", help="Куди зберігати")
    return parser.parse_args()

def train_baseline(args):
    # 1. Пристрій
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"--- TRAINING BASELINE CNN on {device} ---")

    # 2. Дані
    os.makedirs(args.save_dir, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((256, 256))])
    
    try:
        dataset = ColorizationDataset(args.data_path, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Модель
    model = BaselineCNN().to(device)
    
    # 4. Loss & Optimizer (MSE для Baseline)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5. Цикл навчання
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

        # Збереження
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"baseline_epoch_{epoch+1}.pth"))

    # Фінальне збереження
    torch.save(model.state_dict(), os.path.join(args.save_dir, "baseline_final.pth"))
    print("Baseline Training Finished!")

if __name__ == "__main__":
    args = get_args()
    train_baseline(args)