import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # Для красивої смуги завантаження

# Імпортуємо наші модулі
from models.baseline_cnn import BaselineCNN
from utils.dataset import ColorizationDataset

# --- КОНФІГУРАЦІЯ ---
dataset_path = "./data/coco/val2017" # Шлях до картинок
checkpoint_dir = "./checkpoints"      # Куди зберігати модель
batch_size = 16                       # Скільки картинок обробляти за раз
learning_rate = 1e-3                  # Швидкість навчання
num_epochs = 20                       # Скільки разів пройти весь датасет
# --------------------

def train():
    # 1. Визначення пристрою (Device Agnostic Code)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA acceleration!")
    else:
        device = torch.device("cpu")
        print("Using CPU (Warning: this will be slow).")

    # 2. Підготовка даних
    # Нам треба привести всі картинки до одного розміру (256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256) # Можна додати, якщо пропорції сильно різні
    ])

    train_dataset = ColorizationDataset(dataset_path, transform=transform)
    
    # DataLoader розбиває дані на "батчі" і перемішує їх
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded: {len(train_dataset)} images.")

    # 3. Ініціалізація моделі
    model = BaselineCNN().to(device)
    
    # 4. Функція втрат та Оптимізатор
    # Для Baseline моделі використовуємо MSE (Mean Squared Error)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Цикл навчання
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train() # Перемикаємо модель у режим навчання
        running_loss = 0.0
        
        # tqdm додає смугу прогресу
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in loop:
            # Отримуємо дані і кидаємо на GPU/MPS
            input_l = batch['L'].to(device)     # Вхід: (Batch, 1, 256, 256)
            target_ab = batch['ab'].to(device)  # Ціль: (Batch, 2, 256, 256)

            # Обнуляємо градієнти (щоб не накопичувались з минулого кроку)
            optimizer.zero_grad()

            # Forward pass (Прогноз)
            outputs = model(input_l)

            # Рахуємо помилку
            loss = criterion(outputs, target_ab)

            # Backward pass (Навчання)
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Середня помилка за епоху
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")

        # Зберігаємо чекпоінт кожні 5 епох або в кінці
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(checkpoint_dir, f"baseline_cnn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Фінальне збереження
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "baseline_cnn_final.pth"))
    print("Training finished!")

if __name__ == "__main__":
    # Створюємо папку для чекпоінтів, якщо її немає
    os.makedirs(checkpoint_dir, exist_ok=True)
    train()