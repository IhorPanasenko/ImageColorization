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
        root_dir: path to folder with images (e.g. COCO val2017)
        mode: 'train' or 'val'
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, x) for x in os.listdir(root_dir)
                            if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        img = np.array(img)

        # Convert to LAB; skimage returns L in [0, 100], a and b in [-128, 127]
        img_lab = color.rgb2lab(img).astype("float32")

        # Normalize to neural-network-friendly ranges
        img_l  = img_lab[:, :, 0] / 100.0   # L  -> [0, 1]
        img_ab = img_lab[:, :, 1:] / 128.0  # ab -> approximately [-1, 1]

        # Convert to PyTorch tensors in (C, H, W) format
        img_l  = torch.from_numpy(img_l).unsqueeze(0)           # (1, H, W)
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))  # (2, H, W)

        return {'L': img_l, 'ab': img_ab}