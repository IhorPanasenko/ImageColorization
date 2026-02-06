import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Додаємо шлях до src, щоб бачити наші модулі
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.baseline_cnn import BaselineCNN
from src.models.u_net import UNet
from src.utils.dataset import ColorizationDataset

# --- НАЛАШТУВАННЯ ЗА ЗАМОВЧУВАННЯМ (DEFAULTS) ---
def get_args():
    parser = argparse.ArgumentParser(description="Train Colorization Models")
    
    # Тут ми прописуємо ідеальні параметри для U-Net, щоб не вводити їх вручну
    parser.add_argument("--model", type=str, default="unet", choices=["baseline", "unet"], 
                        help="Вибір архітектури: 'baseline' або 'unet'")
    
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Кількість епох навчання")
    
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Розмір пакету (зменш до 8, якщо не вистачає пам'яті)")
    
    parser.add_argument("--lr", type=float, default=2e-4, 
                        help="Швидкість навчання (Learning Rate)")
    
    parser.add_argument("--data_path", type=str, default="./data/coco/val2017", 
                        help="Шлях до зображень")
    
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints", 
                        help="Куди зберігати файли моделі")
    
    return parser.parse_args()

def train(args):
    # 1. Вибір пристрою (Mac MPS / CUDA / CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"--- ЗАПУСК НАВЧАННЯ ---")
    print(f"Пристрій: {device}")
    print(f"Модель:   {args.model.upper()}")
    print(f"Епохи:    {args.epochs}")
    print(f"Дані:     {args.data_path}")
    print(f"-----------------------")

    # 2. Підготовка даних
    os.makedirs(args.save_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.Resize((256, 256))])
    
    try:
        dataset = ColorizationDataset(args.data_path, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Завантажено {len(dataset)} зображень.")
    except FileNotFoundError:
        print(f"ПОМИЛКА: Не знайдено дані у {args.data_path}")
        return

    # 3. Ініціалізація моделі та функції втрат
    if args.model == "baseline":
        model = BaselineCNN().to(device)
        criterion = nn.MSELoss()
    elif args.model == "unet":
        model = UNet().to(device)
        criterion = nn.L1Loss() # L1 краще для чіткості
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Цикл навчання
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0

        for batch in loop:
            # Отримуємо дані
            L = batch['L'].to(device)
            ab = batch['ab'].to(device)

            # Крок навчання
            optimizer.zero_grad()
            outputs = model(L)
            loss = criterion(outputs, ab)
            loss.backward()
            optimizer.step()

            # Оновлення статистики
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Зберігаємо проміжні результати (кожні 5 епох)
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save_dir, f"{args.model}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
    
    # Фінальне збереження
    final_path = os.path.join(args.save_dir, f"{args.model}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Навчання завершено! Модель збережено у: {final_path}")

if __name__ == "__main__":
    args = get_args()
    train(args)