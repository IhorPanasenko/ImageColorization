"""Levin 2004 — Optimization-Based Colorization.

Reference: Levin, A., Lischinski, D., & Weiss, Y. (2004).
           "Colorization using Optimization."
           ACM Transactions on Graphics (SIGGRAPH), 23(3), 689-694.

Algorithm:
    Given sparse colour hints (sampled from the reference image), solve two
    independent sparse linear systems — one for the a-channel and one for the
    b-channel — that enforce the constraint:
        "adjacent pixels with similar luminance should have similar colour."

    The cost function to minimise is:
        J(U) = Σ_r ( U(r) − Σ_{s∈N(r)} w_rs · U(s) )²
    subject to U(r) = c_r for all hint pixels r.

    The weight w_rs between neighbouring pixels r and s is proportional to
        exp(−(L_r − L_s)² / (2σ_r²))
    where σ_r is the local luminance variance in r's neighbourhood.

Usage:
    result_rgb = colorize_levin("gray.jpg", "reference.jpg")  # → float32 H×W×3
"""

import numpy as np
from PIL import Image
from skimage import color
from skimage.transform import resize
import scipy.sparse
import scipy.sparse.linalg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_lab(path: str, target_size: int = 256) -> np.ndarray:
    """Load image from path and return float32 Lab array (H,W,3)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        arr = resize(arr, (target_size, target_size), anti_aliasing=True)
    return color.rgb2lab(arr).astype(np.float32)   # L∈[0,100], ab∈[-128,127]


def _build_weight_matrix(L: np.ndarray, window: int = 3) -> scipy.sparse.csr_matrix:
    """Build the sparse luminance-similarity weight matrix W.

    For each pixel r and each neighbour s in its (window×window) neighbourhood,
    compute affinity:
        w_rs ∝ exp(−(L_r − L_s)² / (2σ_r²))
    where σ_r² is the variance of L in the neighbourhood of r.

    Weights for each row are normalised to sum to 1.

    Args:
        L:      Luminance channel, shape (H, W), values in [0, 100].
        window: Side length of the local neighbourhood (default 3 → 3×3 = 8 neighbours).

    Returns:
        Sparse CSR matrix of shape (N, N) where N = H*W.
    """
    H, W = L.shape
    N = H * W
    half = window // 2

    rows, cols, vals = [], [], []

    # Pre-pad luminance for boundary handling
    L_norm = L / 100.0   # normalise to [0,1] for numerical stability
    L_pad  = np.pad(L_norm, half, mode='reflect')

    for r in range(H):
        for c in range(W):
            pixel_idx = r * W + c

            # Extract local neighbourhood
            patch = L_pad[r: r + window, c: c + window]  # (window, window)

            # Local variance determines sensitivity: small variance → tight coupling
            sigma2 = np.var(patch)
            if sigma2 < 1e-6:
                sigma2 = 1e-6   # avoid division by zero in flat regions

            center_L  = L_norm[r, c]

            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    if dr == 0 and dc == 0:
                        continue  # skip self-connection; diagonal is handled separately
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        neighbour_idx = nr * W + nc
                        diff = L_norm[nr, nc] - center_L
                        w    = np.exp(-(diff ** 2) / (2.0 * sigma2))
                        rows.append(pixel_idx)
                        cols.append(neighbour_idx)
                        vals.append(w)

    W_sparse = scipy.sparse.csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(N, N),
    )

    # Row-normalise so each row sums to 1 (stochastic matrix)
    row_sums = np.array(W_sparse.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0   # avoid divide-by-zero for isolated pixels
    D_inv = scipy.sparse.diags(1.0 / row_sums, 0, format='csr')
    return D_inv.dot(W_sparse)


def _solve_channel(
    W: scipy.sparse.csr_matrix,
    N: int,
    hint_indices: np.ndarray,
    hint_values: np.ndarray,
) -> np.ndarray:
    """Solve (I - W) u = b for one chrominance channel.

    At hint pixels, we pin the value by modifying the row of (I-W):
        - Set row to identity (already 0 off-diagonal from I-W structure)
        - Set b[hint] = hint_value

    Args:
        W:             Normalised weight matrix (N×N sparse).
        N:             Number of pixels.
        hint_indices:  1-D int array of hint pixel indices.
        hint_values:   Corresponding chrominance values (normalised to [−1, 1]).

    Returns:
        Solution vector u of length N.
    """
    I_minus_W = scipy.sparse.eye(N, format='csr') - W

    # Build RHS
    b = np.zeros(N, dtype=np.float64)
    b[hint_indices] = hint_values

    # Pin hint rows: set row to identity row (zero off-diag column, 1 on diag)
    # Efficient approach: zero out the rows, then set diagonal back to 1
    I_minus_W = I_minus_W.tolil()
    for idx in hint_indices:
        I_minus_W[idx, :] = 0.0
        I_minus_W[idx, idx] = 1.0
    I_minus_W = I_minus_W.tocsr()

    # Solve sparse linear system using Conjugate Gradient (fast iterative solver)
    u, info = scipy.sparse.linalg.cg(
        I_minus_W, b,
        maxiter=2000,
        rtol=1e-4,
    )

    if info != 0:
        # Fall back to direct solver for small images or if CG diverges
        try:
            u = scipy.sparse.linalg.spsolve(I_minus_W, b)
        except Exception:
            pass  # return whatever CG converged to

    return u


# ── Public API ─────────────────────────────────────────────────────────────────

def colorize_levin(
    target_path: str,
    reference_path: str,
    n_hints: int = 50,
    target_size: int = 256,
    window: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Colorize a greyscale image using Levin et al. 2004.

    Colour hints are automatically generated by randomly sampling pixels from
    the reference image at the same spatial positions in the target image.
    This simulates a user painting sparse coloured strokes on the greyscale input.

    Args:
        target_path:    Path to the greyscale (or colour-stripped) target image.
        reference_path: Path to the colour reference image.
        n_hints:        Number of random hint pixels to sample (default 50).
                        More hints → better quality but slower convergence.
        target_size:    Resize both images to this square size (default 256).
        window:         Local neighbourhood window for weight computation (default 3).
        seed:           Random seed for reproducible hint sampling.

    Returns:
        Colorized image as float32 numpy array of shape (H, W, 3) in [0, 1].
    """
    # ── Load images ────────────────────────────────────────────────────────────
    target_lab    = _load_lab(target_path,    target_size)
    reference_lab = _load_lab(reference_path, target_size)

    H, W = target_lab.shape[:2]
    N    = H * W

    L_channel = target_lab[:, :, 0]  # (H, W), values in [0, 100]

    # ── Sample random hint pixel positions ────────────────────────────────────
    rng           = np.random.RandomState(seed)
    hint_flat_idx = rng.choice(N, size=n_hints, replace=False)
    hint_rows     = hint_flat_idx // W
    hint_cols     = hint_flat_idx %  W

    # Extract ab values from reference at hint positions (normalise to [−1, 1])
    hint_a = reference_lab[hint_rows, hint_cols, 1] / 128.0
    hint_b = reference_lab[hint_rows, hint_cols, 2] / 128.0

    # ── Build sparse weight matrix ─────────────────────────────────────────────
    W_mat = _build_weight_matrix(L_channel, window=window)

    # ── Solve for a and b channels independently ───────────────────────────────
    u_a = _solve_channel(W_mat, N, hint_flat_idx, hint_a)
    u_b = _solve_channel(W_mat, N, hint_flat_idx, hint_b)

    # ── Reconstruct result ─────────────────────────────────────────────────────
    result_lab = np.zeros((H, W, 3), dtype=np.float32)
    result_lab[:, :, 0]  = L_channel                        # original luminance
    result_lab[:, :, 1]  = np.clip(u_a.reshape(H, W), -1.0, 1.0) * 128.0
    result_lab[:, :, 2]  = np.clip(u_b.reshape(H, W), -1.0, 1.0) * 128.0

    result_rgb = color.lab2rgb(result_lab).astype(np.float32)
    return np.clip(result_rgb, 0.0, 1.0)
