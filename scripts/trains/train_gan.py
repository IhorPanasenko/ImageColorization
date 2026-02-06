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

# Імпортуємо наші моделі
# УВАГА: UNet виступає в ролі Генератора!
from src.models.u_net import UNet  
from src.models.gan.discriminator import PatchDiscriminator
from src.utils.dataset import ColorizationDataset

# --- НАЛАШТУВАННЯ ---
GAN_LAMBDA_L1 = 100.0 # Вага L1 loss (колір) відносно GAN loss (реалістичність)

class GANLoss(nn.Module):
    """
    Клас для розрахунку змагальної помилки (Adversarial Loss).
    Використовує BCEWithLogitsLoss, що є стандартом для стабільного навчання GAN.
    """
    def __init__(self, device):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device
        
        # Створюємо мітки "правда" (1.0) і "брехня" (0.0)
        # Ми будемо динамічно підганяти їх розмір під вихід дискримінатора
        self.real_label = 1.0
        self.fake_label = 0.0

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.full_like(prediction, self.real_label, device=self.device)
        else:
            return torch.full_like(prediction, self.fake_label, device=self.device)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

def get_args():
    parser = argparse.ArgumentParser(description="Train GAN Model")
    parser.add_argument("--epochs", type=int, default=20, help="Кількість епох")
    # Для GAN краще менший batch_size, бо в пам'яті 2 моделі
    parser.add_argument("--batch_size", type=int, default=8, help="Розмір пакету")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate")
    parser.add_argument("--data_path", type=str, default="./data/coco/val2017", help="Шлях до даних")
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints", help="Куди зберігати")
    return parser.parse_args()

def train_gan(args):
    # 1. Пристрій
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"--- STARTING GAN TRAINING on {device} ---")

    # 2. Дані
    os.makedirs(args.save_dir, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((256, 256))])
    
    dataset = ColorizationDataset(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Data loaded: {len(dataset)} images")

    # 3. Ініціалізація моделей
    print("Initializing models...")
    net_G = UNet().to(device)                 # Генератор (наш U-Net)
    net_D = PatchDiscriminator().to(device)   # Дискримінатор

    # 4. Функції втрат
    criterion_GAN = GANLoss(device)  # Для обману/виявлення (BCE)
    criterion_L1 = nn.L1Loss()       # Для точності кольору

    # 5. Оптимізатори
    # beta1=0.5 важливо для GAN (допомагає уникнути нестабільності)
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 6. Цикл навчання
    for epoch in range(args.epochs):
        net_G.train()
        net_D.train()
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in loop:
            # Дані
            real_L = batch['L'].to(device)      # Вхід (Ч/Б)
            real_ab = batch['ab'].to(device)    # Ціль (Колір)

            # ===============================
            #  КРОК 1: Тренування Дискримінатора (D)
            # ===============================
            # D має максимізувати log(D(x, y)) + log(1 - D(x, G(x)))
            optimizer_D.zero_grad()

            # 1.1. На реальних картинках
            # Подаємо пару (L, справжні ab)
            pred_real = net_D(real_L, real_ab)
            loss_D_real = criterion_GAN(pred_real, target_is_real=True)

            # 1.2. На фейкових картинках
            # Генеруємо фейк (detach, щоб не чіпати G)
            fake_ab = net_G(real_L)
            # Подаємо пару (L, згенеровані ab)
            pred_fake = net_D(real_L, fake_ab.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_is_real=False)

            # 1.3. Загальна помилка D
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ===============================
            #  КРОК 2: Тренування Генератора (G)
            # ===============================
            # G має максимізувати log(D(x, G(x))) -> обманути D
            optimizer_G.zero_grad()

            # 2.1. GAN Loss (обманюємо D)
            # Тепер ми НЕ робимо detach, бо хочемо, щоб градієнти пішли в G
            pred_fake_for_G = net_D(real_L, fake_ab)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, target_is_real=True)

            # 2.2. L1 Loss (щоб кольори були схожі на правду)
            loss_G_L1 = criterion_L1(fake_ab, real_ab) * GAN_LAMBDA_L1

            # 2.3. Загальна помилка G
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            # Оновлення прогрес-бару
            loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

        # --- ЗБЕРЕЖЕННЯ ---
        # Зберігаємо кожні 5 епох
        if (epoch + 1) % 5 == 0:
            torch.save(net_G.state_dict(), os.path.join(args.save_dir, f"gan_generator_epoch_{epoch+1}.pth"))
            torch.save(net_D.state_dict(), os.path.join(args.save_dir, f"gan_discriminator_epoch_{epoch+1}.pth"))

    # Фінальне збереження
    torch.save(net_G.state_dict(), os.path.join(args.save_dir, "gan_generator_final.pth"))
    torch.save(net_D.state_dict(), os.path.join(args.save_dir, "gan_discriminator_final.pth"))
    print("GAN Training Finished!")

if __name__ == "__main__":
    args = get_args()
    train_gan(args)