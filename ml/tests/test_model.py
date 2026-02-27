import torch
import pytest
import sys
import os

# Додаємо шлях до src, щоб імпортувати моделі
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.baseline_cnn import BaselineCNN
from src.models.u_net import UNet
from src.models.unet_fusion import UNetFusion
from src.models.global_hints import GlobalHintNet

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


def test_global_hint_net_output_shape():
    """GlobalHintNet must compress (B,1,256,256) → (B,512) semantic feature vector."""
    model = GlobalHintNet()
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"


def test_unet_fusion_output_shape():
    """UNetFusion must accept (B,1,H,W) + (B,512) hint and return (B,2,H,W)."""
    model = UNetFusion()
    hint = torch.randn(2, 512)
    x = torch.randn(2, 1, 256, 256)
    out = model(x, hint)
    assert out.shape == (2, 2, 256, 256), f"Expected (2, 2, 256, 256), got {out.shape}"


def test_unet_fusion_with_global_hint_net():
    """End-to-end: L channel → GlobalHintNet → UNetFusion → ab prediction."""
    hint_net = GlobalHintNet()
    hint_net.eval()
    gen = UNetFusion()
    gen.eval()

    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        hint = hint_net(x)
        out = gen(x, hint)

    assert hint.shape == (2, 512), f"GlobalHintNet output: expected (2,512), got {hint.shape}"
    assert out.shape == (2, 2, 256, 256), f"UNetFusion output: expected (2,2,256,256), got {out.shape}"