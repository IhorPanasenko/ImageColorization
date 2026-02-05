import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        
        # Encoder (Стиснення)
        # Вхід: 1 канал (L - яскравість)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), # -> зменшує розмір в 2 рази
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> ще в 2 рази
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> ще в 2 рази
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        
        # Decoder (Відновлення)
        # Вихід: 2 канали (a, b - колір)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # збільшує
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            # На виході Tanh, щоб загнати значення в діапазон [-1, 1], як ми нормалізували в dataset.py
            nn.Tanh() 
        )

    def forward(self, input_l):
        # Прохід через мережу
        features = self.encoder(input_l)
        output_ab = self.decoder(features)
        return output_ab

# Невеликий тест, щоб перевірити, чи не падає код
if __name__ == "__main__":
    # Створюємо фейковий тензор (1 картинка, 1 канал L, 256x256 пікселів)
    dummy_input = torch.randn(1, 1, 256, 256)
    model = BaselineCNN()
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Має бути (1, 2, 256, 256)