import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage import color

# Додаємо шлях до src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.baseline_cnn import BaselineCNN
from src.models.unet import UNet
# from src.models.unet_fusion import UNetFusion
# from src.models.global_hints import GlobalHintNet

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Colorization Models")
    
    # --- DEFAULTS ---
    # За замовчуванням беремо GAN (U-Net Generator)
    parser.add_argument("--model", type=str, default="gan", choices=["baseline", "unet", "gan", "fusion"], 
                        help="Тип моделі (default: gan)")
    
    # За замовчуванням шукаємо останній чекпоінт генератора
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/gan_generator_final.pth", 
                        help="Шлях до файлу ваг .pth")
    
    # За замовчуванням беремо папку для тестів
    parser.add_argument("--img_path", type=str, default="./data/test_samples", 
                        help="Шлях до картинки або папки з картинками")
    
    parser.add_argument("--save_dir", type=str, default="./outputs/images", 
                        help="Куди зберігати результат")
    
    parser.add_argument("--device", type=str, default="auto", 
                        help="cuda, mps, або cpu")
    
    return parser.parse_args()

def load_model(model_type, checkpoint_path, device):
    print(f"--- Loading Model: {model_type.upper()} ---")
    print(f"Path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Checkpoint not found at {checkpoint_path}.\n"
                                f"Did you run training? Try: python scripts/train_gan.py")

    if model_type == "baseline":
        model = BaselineCNN().to(device)
    elif model_type == "unet" or model_type == "gan":
        # У GAN генератором є U-Net
        model = UNet().to(device)
    elif model_type == "fusion":
        # model = UNetFusion().to(device)
        # hint_net = GlobalHintNet().to(device)
        # hint_net.eval()
        # return model, hint_net
        pass
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Завантаження ваг
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Іноді чекпоінт зберігається як словник {'model_state_dict': ...}
    # А іноді просто як state_dict. Додамо перевірку:
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("Model loaded successfully!")
    
    if model_type == "fusion":
        return model, None # hint_net
    return model, None

def process_image(img_path):
    img = Image.open(img_path).convert("RGB")
    original_size = img.size # (W, H)

    