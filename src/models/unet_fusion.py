import torch
import torch.nn as nn
from src.models.unet import UNetDown, UNetUp # Використовуємо блоки зі старого файлу

class UNetFusion(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, global_features_dim=512):
        super(UNetFusion, self).__init__()

        # --- ENCODER (Такий самий) ---
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        
        # Bottleneck (Найглибше місце)
        # Вхід: 512. Вихід: 512. Розмір зображення тут 1x1 піксель (при вході 256)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # --- FUSION LAYER ---
        # Тут магія. Ми об'єднуємо вихід Bottleneck (512) і Глобальну підказку (512)
        # Разом вийде 1024 канали.
        # Ми хочемо зменшити їх назад до 512 перед подачею в декодер.
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(inplace=True)
        )

        # --- DECODER (Зміни на початку) ---
        
        # up1 приймає:
        # 1. Вихід з fusion (ми його зробили 512)
        # 2. Skip connection з down7 (512)
        # Разом вхід = 1024.
        self.up1 = UNetUp(512, 512, dropout=0.5) 
        
        # Далі все стандартно
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
            nn.Tanh()
        )

    def forward(self, x, global_hint):
        # 1. Прохід вниз
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7) # Тут розмір (Batch, 512, 1, 1)

        # 2. FUSION (Злиття)
        # Вирівнюємо d8 у вектор
        d8_flat = d8.view(d8.size(0), -1) # (Batch, 512)
        
        # Об'єднуємо з підказкою ResNet
        combined = torch.cat([d8_flat, global_hint], dim=1) # (Batch, 1024)
        
        # Змішуємо їх через Linear шар
        fused = self.fusion_layer(combined) # (Batch, 512)
        
        # Повертаємо форму тензора (Batch, 512, 1, 1) для декодера
        fused = fused.view(fused.size(0), 512, 1, 1)

        # 3. Прохід вгору
        u1 = self.up1(fused, d7) # fused замість d8
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)