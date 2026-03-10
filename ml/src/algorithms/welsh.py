"""Welsh 2002 — Texture-Based Color Transfer.

Reference: Welsh, T., Ashikhmin, M., & Mueller, K. (2002).
           "Transferring Color to Greyscale Images."
           ACM Transactions on Graphics (SIGGRAPH), 21(3), 277-280.

Algorithm:
    For each pixel in the grayscale target, find the best-matching pixel in the
    colour reference image by comparing:
        1. Luminance value (L)
        2. Standard deviation of luminance in a local 5×5 neighbourhood (texture)
    Transfer the ab chrominance channels from the matched reference pixel.

Usage:
    result_rgb = colorize_welsh("gray.jpg", "reference.jpg")  # → float32 H×W×3 in [0,1]
"""

import numpy as np
from PIL import Image
from skimage import color
from skimage.transform import resize
from scipy.ndimage import uniform_filter
from scipy.spatial import KDTree


def _load_as_lab(path: str, target_size: int = 256) -> np.ndarray:
    """Load an image (any format/mode) and return a normalised float32 Lab array."""
    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    if img_np.shape[0] != target_size or img_np.shape[1] != target_size:
        img_np = resize(img_np, (target_size, target_size), anti_aliasing=True)
    return color.rgb2lab(img_np).astype(np.float32)  # L∈[0,100], ab∈[-128,127]


def _local_std(L: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute per-pixel standard deviation of L in a local window.

    Uses the identity: std² = E[x²] - E[x]² for fast computation via
    uniform (box) filtering — O(N) complexity regardless of window size.
    """
    mean   = uniform_filter(L, size=window, mode='reflect')
    mean_sq = uniform_filter(L ** 2, size=window, mode='reflect')
    var    = np.maximum(mean_sq - mean ** 2, 0.0)
    return np.sqrt(var)


def colorize_welsh(
    target_path: str,
    reference_path: str,
    target_size: int = 256,
    window: int = 5,
    jitter: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Colorize a greyscale image using Welsh et al. 2002.

    Args:
        target_path:    Path to the greyscale (or colour-stripped) target image.
        reference_path: Path to the colour reference image.
        target_size:    Both images are resized to this square size before
                        processing (default 256).
        window:         Neighbourhood window for local std computation (default 5).
        jitter:         If > 0, add Gaussian noise to feature vectors before
                        KDTree lookup to introduce colour diversity (default 0).
        seed:           Random seed for jitter reproducibility.

    Returns:
        Colorized image as float32 numpy array of shape (H, W, 3) in [0, 1].
    """
    # ── Load & convert both images to Lab ──────────────────────────────────────
    target_lab    = _load_as_lab(target_path,    target_size)
    reference_lab = _load_as_lab(reference_path, target_size)

    H, W = target_lab.shape[:2]

    # ── Compute local luminance texture (std in window) ────────────────────────
    target_std = _local_std(target_lab[:, :, 0], window=window)    # (H, W)
    ref_std    = _local_std(reference_lab[:, :, 0], window=window)  # (H, W)

    # ── Build 2D feature vectors: [L, local_std] ──────────────────────────────
    # Normalise L to [0,1] and std to reasonable range so both dimensions are
    # comparable (both naturally lie in similar numeric ranges after /100).
    ref_L   = reference_lab[:, :, 0].ravel() / 100.0     # (N,)
    ref_sig = ref_std.ravel() / 100.0                      # (N,)
    ref_ab  = reference_lab[:, :, 1:].reshape(-1, 2)      # (N, 2)

    ref_features = np.stack([ref_L, ref_sig], axis=1)     # (N, 2)

    tgt_L   = target_lab[:, :, 0].ravel() / 100.0
    tgt_sig = target_std.ravel() / 100.0
    tgt_features = np.stack([tgt_L, tgt_sig], axis=1)     # (N, 2)

    # Optional jitter for colour diversity
    if jitter > 0.0:
        rng = np.random.RandomState(seed)
        tgt_features = tgt_features + rng.normal(0, jitter, tgt_features.shape)

    # ── KDTree nearest-neighbour lookup (O(N log N)) ───────────────────────────
    tree = KDTree(ref_features)
    _, indices = tree.query(tgt_features, k=1, workers=-1)   # workers=-1 uses all CPUs

    # ── Transfer chrominance ───────────────────────────────────────────────────
    transferred_ab = ref_ab[indices]                          # (N, 2)

    # ── Reconstruct Lab image ──────────────────────────────────────────────────
    result_lab = np.zeros((H, W, 3), dtype=np.float32)
    result_lab[:, :, 0]  = target_lab[:, :, 0]               # keep original L
    result_lab[:, :, 1:] = transferred_ab.reshape(H, W, 2)

    # ── Convert back to RGB ────────────────────────────────────────────────────
    result_rgb = color.lab2rgb(result_lab).astype(np.float32)  # [0, 1]
    return np.clip(result_rgb, 0.0, 1.0)
