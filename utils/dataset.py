import os
import numpy as np
from torch.utils.data import Dataset
from skimage import color, io, transform
import torch
from torchvision import transforms
from PIL import Image

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        root_dir: шлях до папки з картинками (наприклад, COCO)
        mode: 'train' або 'val'
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Зчитуємо всі файли з папки
        self.image_paths = [os.path.join(root_dir, x) for x in os.listdir(root_dir) 
                            if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. Відкриваємо зображення
        img = Image.open(img_path).convert("RGB")
        
        # 2. Застосовуємо трансформації (зменшення розміру)
        if self.transform:
            img = self.transform(img)
        
        img = np.array(img)

        # 3. Конвертація в LAB
        # skimage повертає L [0, 100], a [-128, 127], b [-128, 127]
        img_lab = color.rgb2lab(img).astype("float32")
        
        # 4. Нормалізація (приводимо дані до діапазону для нейромережі)
        # Канал L приводимо до [0, 1]
        img_l = img_lab[:, :, 0] / 100.0 
        # Канали ab приводимо до діапазону приблизно [-1, 1]
        img_ab = img_lab[:, :, 1:] / 128.0 

        # 5. Перетворення в тензори PyTorch
        # PyTorch любить формат (Channels, Height, Width), а не (H, W, C)
        img_l = torch.from_numpy(img_l).unsqueeze(0) # Додаємо канал: (1, H, W)
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))) # (2, H, W)

        return {'L': img_l, 'ab': img_ab}