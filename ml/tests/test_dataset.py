import torch
import pytest
import sys
import os
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.dataset import ColorizationDataset

# Шлях до реальних даних (або можна створити фейкову картинку для тесту)
DATA_PATH = "./data/coco/val2017"

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Датасет COCO не завантажено")
def test_dataset_loading():
    transform = transforms.Resize((256, 256))
    dataset = ColorizationDataset(DATA_PATH, transform=transform)
    
    assert len(dataset) > 0, "Датасет пустий!"
    
    # Беремо перший елемент
    sample = dataset[0]
    L = sample['L']
    ab = sample['ab']
    
    # ПЕРЕВІРКИ:
    # 1. Перевіряємо типи даних
    assert isinstance(L, torch.Tensor)
    assert isinstance(ab, torch.Tensor)
    
    # 2. Перевіряємо розмірності
    assert L.shape == (1, 256, 256)
    assert ab.shape == (2, 256, 256)
    
    # 3. Перевіряємо діапазони значень (нормалізацію)
    # L має бути [0, 1]
    assert L.min() >= 0.0 and L.max() <= 1.0
    
    # ab має бути в розумних межах (теоретично [-1, 1], але іноді бувають викиди, тому ставимо з запасом)
    assert ab.min() >= -1.5 and ab.max() <= 1.5