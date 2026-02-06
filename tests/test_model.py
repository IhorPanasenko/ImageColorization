import torch
import pytest
import sys
import os

# Додаємо шлях до src, щоб імпортувати моделі
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.baseline_cnn import BaselineCNN
from src.models.unet import UNet

@pytest.mark.parametrize("model_class", [BaselineCNN, UNet])
def test_model_shapes(model_class):
    """
    Перевіряємо, що якщо на вхід подати (Batch, 1, 256, 256),
    то на виході отримаємо (Batch, 2, 256, 256).
    """
    model = model_class()
    batch_size = 2
    
    # Створюємо фейковий вхідний тензор (випадкові числа)
    # Формат: (Batch, Channel=1, Height=256, Width=256)
    dummy_input = torch.randn(batch_size, 1, 256, 256)
    
    # Проганяємо через модель
    output = model(dummy_input)
    
    # ПЕРЕВІРКИ:
    # 1. Вихід не має бути пустим
    assert output is not None
    # 2. Розмірність має бути (Batch, 2, 256, 256)
    assert output.shape == (batch_size, 2, 256, 256)
    print(f"Tested {model_class.__name__}: Output shape OK.")

def test_unet_skip_connections():
    """Специфічний тест для U-Net, щоб переконатися, що не виникає помилок при конкатенації"""
    model = UNet()
    x = torch.randn(1, 1, 256, 256)
    try:
        y = model(x)
    except RuntimeError as e:
        pytest.fail(f"U-Net forward pass failed (likely skip connection mismatch): {e}")