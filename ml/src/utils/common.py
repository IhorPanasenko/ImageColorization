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
from typing import Union

_RESAMPLE = (
    Image.Resampling.BICUBIC
    if hasattr(Image, "Resampling")
    else getattr(Image, "BICUBIC", 3)
)


def get_device() -> torch.device:
    """
    Return the best available torch device for inference.

    Priority: CUDA > CPU  (MPS disabled by default — see below).

    MPS (Apple Silicon Metal) is disabled by default because PyTorch's MPS
    backend can trigger non-catchable native crashes (SIGBUS / EXC_BAD_ACCESS)
    on certain macOS / PyTorch / Python version combinations — for example
    PyTorch 2.10 with Python 3.14 on Mac16,12 hardware running macOS 26.
    These crashes occur inside the Metal runtime during complex conv/matmul
    operations and kill the entire Python process; a simple smoke-test with
    tiny tensors is not sufficient to detect the problem.

    To re-enable MPS (e.g. after upgrading PyTorch to a version that
    officially supports your Python / macOS release), set the environment
    variable ``COLORIZE_DEVICE=mps``.

    You can also force any device explicitly::

        COLORIZE_DEVICE=cpu   python run.py
        COLORIZE_DEVICE=mps   python run.py
        COLORIZE_DEVICE=cuda  python run.py
    """
    forced = os.environ.get("COLORIZE_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)

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

    # Clamp ab to the valid CIE-Lab range to prevent impossible sRGB colors.
    # Without clamping, GAN outputs occasionally exceed Lab gamut limits,
    # producing negative XYZ values that get hard-clipped by lab2rgb and
    # degrade both visual quality and PSNR/SSIM.
    ab_np = np.clip(ab_np, -128.0, 127.0)

    # Assemble Lab image: (H, W, 3)
    ab_np = ab_np.transpose(1, 2, 0)                   # (H, W, 2)
    lab = np.concatenate([L_np[:, :, np.newaxis], ab_np], axis=2).astype(np.float32)

    rgb = color.lab2rgb(lab)                           # skimage returns float64 in [0, 1]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def prepare_grayscale_input(
    img_path: str,
    target_size: int = 256,
    return_meta: bool = False,
) -> Union[
    tuple[torch.Tensor, np.ndarray],
    tuple[torch.Tensor, np.ndarray, dict[str, int]],
]:
    """
    Load an image, extract and normalize the L channel for model input,
    and return the ground-truth RGB for comparison.

    The image is squash-resized to (target_size × target_size), matching the
    transforms.Resize((256, 256)) used in every training script.  Using a
    different resize strategy (e.g. aspect-preserving + black padding) would
    introduce a train/inference distribution mismatch that degrades PSNR.

    Args:
        img_path:    Path to the input image file.
        target_size: Square size the image is resized to (default 256).

    Returns:
        L_tensor:     Float tensor of shape (1, 1, H, W), L channel in [0, 1].
        original_rgb: Ground-truth RGB as float32 numpy (H, W, 3) in [0, 1].
        meta (opt):   Resize metadata for downstream display / aspect-ratio
                      recovery.  pad_* fields are always 0 (no padding applied).
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Squash-resize to match training transforms.Resize((target_size, target_size))
    img_resized = img.resize((target_size, target_size), _RESAMPLE)

    img_np = np.array(img_resized)
    lab = color.rgb2lab(img_np).astype(np.float32)

    L = lab[:, :, 0] / 100.0                           # (H, W) in [0, 1]
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    original_rgb = (img_np / 255.0).astype(np.float32) # (H, W, 3) in [0, 1]

    if return_meta:
        meta = {
            "orig_w": orig_w,
            "orig_h": orig_h,
            "resized_w": target_size,
            "resized_h": target_size,
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 0,
            "pad_bottom": 0,
        }
        return L_tensor, original_rgb, meta

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
