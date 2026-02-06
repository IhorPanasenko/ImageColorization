import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.gan.discriminator import PatchDiscriminator

def test_discriminator_shape():
    # Вхід: (Batch, 1, 256, 256) для L і (Batch, 2, 256, 256) для ab
    x = torch.randn(1, 1, 256, 256)
    y = torch.randn(1, 2, 256, 256)
    
    model = PatchDiscriminator()
    output = model(x, y)
    
    print(f"Discriminator Output Shape: {output.shape}")
    
    # PatchGAN зменшує розмірність. 
    # Для входу 256x256 і 3 шарів даунсемплінгу вихід буде приблизно 30x30 або 26x26
    assert output.shape[1] == 1 # Має бути 1 канал (карта ймовірностей)
    assert output.shape[2] < 256 # Ширина має зменшитися