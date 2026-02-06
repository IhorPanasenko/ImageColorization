import torch
import torch.nn as nn

class Block(nn.Module):
    """
    Базовий будівельний блок Дискримінатора:
    Convolution -> Batch Normalization -> Leaky ReLU
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        """
        in_channels: 3 (тому що ми подаємо L + ab, тобто все зображення разом)
        features: кількість каналів у прихованих шарах
        """
        super().__init__()
        
        # Початковий шар (без BatchNorm)
        # 3 -> 64
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_c = features[0]
        
        # Створюємо каскад шарів (Downsampling)
        # 64 -> 128 -> 256 -> 512
        for feature in features[1:]:
            layers.append(Block(in_c, feature, stride=2)) # stride=2 зменшує розмір картинки вдвічі
            in_c = feature

        # Останній шар перед виходом (stride=1, не зменшує розмір)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=4, stride=1, padding=1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(in_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Фінальний шар, який перетворює все в 1 канал (карту передбачень)
        self.final = nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        """
        x: Чорно-білий канал L (1 канал)
        y: Кольорові канали ab (2 канали) - або справжні, або згенеровані
        """
        # Ми "склеюємо" вхід і ціль, щоб Дискримінатор бачив повну картинку
        input_image = torch.cat([x, y], dim=1) # (Batch, 3, 256, 256)
        
        x = self.initial(input_image)
        for layer in layers: # Ця змінна layers має бути частиною self, виправимо нижче
            pass 
        
        # --- ВИПРАВЛЕНА ЛОГІКА ---
        # У конструкторі ми створили список layers, але не додали його в self.model
        # Давайте перепишемо це чистіше прямо тут.
        return self.model(input_image)

    # --- ПЕРЕПИСАНИЙ __INIT__ ДЛЯ ЧИСТОТИ ---
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # 1. Перший шар (Input -> 64)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Основні блоки (64 -> 128 -> 256 -> 512)
        layers = []
        in_c = features[0]
        for feature in features[1:]:
            layers.append(Block(in_c, feature, stride=2))
            in_c = feature
        
        # 3. Передостанній шар (stride=1)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=4, stride=1, padding=1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(in_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        self.intermediate = nn.Sequential(*layers)

        # 4. Фінальний шар (512 -> 1)
        # Вихід: Матриця чисел (наприклад 30x30), де кожне число - це "реалістичність" патчу
        self.final = nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        # x: L канал (Batch, 1, 256, 256)
        # y: ab канали (Batch, 2, 256, 256)
        
        # Об'єднуємо в повноколірне зображення (Lab)
        input_concat = torch.cat([x, y], dim=1) # -> (Batch, 3, 256, 256)
        
        x = self.initial(input_concat)
        x = self.intermediate(x)
        return self.final(x)