"""
Shared utility functions used across training scripts and evaluation.

All functions respect the project's Lab color space convention:
  - L channel normalized to [0, 1]  (original range [0, 100]  / 100.0)
  - ab channels normalized to [-1, 1] (original range [-128, 127] / 128.0)
"""

import os
import numpy as np
import torch
from PIL import Image
from skimage import color
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
from torchvision import transforms


def get_device() -> torch.device:
    """
    Return the best available device: MPS (Apple Silicon) > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lab_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """
    Convert normalized L and ab tensors back to an RGB numpy image.

    Args:
        L:  Tensor of shape (1, H, W) or (H, W), normalized in [0, 1].
        ab: Tensor of shape (2, H, W), normalized in [-1, 1].

    Returns:
        RGB image as float32 numpy array of shape (H, W, 3), clipped to [0, 1].
    """
    # Move to CPU and remove batch/channel dims as needed
    L_np = L.detach().cpu().squeeze().numpy()          # (H, W)
    ab_np = ab.detach().cpu().numpy()                  # (2, H, W)

    # Denormalize: undo /100 and /128 applied during dataset loading
    L_np = L_np * 100.0                                # [0, 1]   -> [0, 100]
    ab_np = ab_np * 128.0                              # [-1, 1]  -> [-128, 128]

    # Assemble Lab image: (H, W, 3)
    ab_np = ab_np.transpose(1, 2, 0)                   # (H, W, 2)
    lab = np.concatenate([L_np[:, :, np.newaxis], ab_np], axis=2).astype(np.float32)

    rgb = color.lab2rgb(lab)                           # skimage returns float64 in [0, 1]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def prepare_grayscale_input(
    img_path: str,
    target_size: int = 256,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load an image, extract and normalize the L channel for model input,
    and return the ground-truth RGB for comparison.

    Args:
        img_path:    Path to the input image file.
        target_size: Resize the shorter edge to this value (default 256).

    Returns:
        L_tensor:     Float tensor of shape (1, 1, H, W), L channel in [0, 1].
        original_rgb: Ground-truth RGB as float32 numpy (H, W, 3) in [0, 1].
    """
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Resize((target_size, target_size))
    img = transform(img)

    img_np = np.array(img)
    lab = color.rgb2lab(img_np).astype(np.float32)

    L = lab[:, :, 0] / 100.0                           # (H, W) in [0, 1]
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    original_rgb = (img_np / 255.0).astype(np.float32) # (H, W, 3) in [0, 1]
    return L_tensor, original_rgb


def save_comparison_strip(
    grayscale: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str,
    title: str = "",
) -> None:
    """
    Save a side-by-side comparison strip: [Grayscale | Prediction | Ground Truth].

    Args:
        grayscale:    Grayscale image as (H, W, 3) or (H, W) float32 in [0, 1].
        prediction:   Model output RGB as (H, W, 3) float32 in [0, 1].
        ground_truth: Ground-truth RGB as (H, W, 3) float32 in [0, 1].
        save_path:    Full path (including filename) where the PNG will be saved.
        title:        Optional suptitle for the figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Convert grayscale to 3-channel if needed
    if grayscale.ndim == 2 or (grayscale.ndim == 3 and grayscale.shape[2] == 1):
        gray3 = np.stack([grayscale.squeeze()] * 3, axis=2)
    else:
        gray3 = grayscale

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (gray3,       "Grayscale Input"),
        (prediction,  "Model Prediction"),
        (ground_truth, "Ground Truth"),
    ]
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
