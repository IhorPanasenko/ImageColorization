"""
Reusable loss functions shared across GAN and Fusion training scripts.
"""

import torch
import torch.nn as nn
from torchvision import models


class GANLoss(nn.Module):
    """
    Adversarial loss using BCEWithLogitsLoss.

    Dynamically creates real/fake label tensors that match the discriminator's
    output shape and device automatically, so it works with any output spatial
    resolution on any device (CPU, CUDA, MPS) without manual device tracking.

    Args:
        label_smoothing: If > 0, real labels become (1 - label_smoothing)
                         to prevent the discriminator from becoming
                         over-confident.  Standard value: 0.1.
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.real_label = 1.0 - label_smoothing
        self.fake_label = 0.0

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        value = self.real_label if target_is_real else self.fake_label
        # torch.full_like mirrors prediction's device, dtype and shape automatically.
        return torch.full_like(prediction, value)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class PerceptualLoss(nn.Module):
    """
    VGG-16 based perceptual loss (feature matching loss).

    Compares activations at selected VGG layers between predicted and
    target images.  This bridges pixel-level accuracy (L1) and perceptual
    quality — critical for GAN-based colorization where L1 alone causes
    desaturated colors and pure GAN loss causes instability.

    The input to this loss should be 3-channel RGB-ish tensors in [0, 1].
    Since our models output 2-channel Lab *ab*, callers must convert to
    a pseudo-RGB representation first (see helper ``ab_to_pseudo_rgb``).
    """

    def __init__(self, layer_ids: tuple[int, ...] = (3, 8, 15, 22)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        prev = 0
        for lid in layer_ids:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:lid + 1]))
            prev = lid + 1
        # Freeze all VGG weights
        for p in self.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) predicted pseudo-RGB in [0, 1].
            target: (B, 3, H, W) ground-truth pseudo-RGB in [0, 1].
        Returns:
            Scalar perceptual loss (sum over selected layers).
        """
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for s in self.slices:
            x = s(x)
            with torch.no_grad():
                y = s(y)
            loss = loss + self.criterion(x, y)
        return loss


def ab_to_pseudo_rgb(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """
    Approximate Lab→RGB conversion *on the GPU* for perceptual loss.

    This creates a 3-channel tensor by stacking the L channel with the ab
    channels, normalized to roughly [0, 1].  It is NOT a true Lab→sRGB
    conversion (that requires non-differentiable clipping), but it gives
    VGG features that are correlated with color perception.

    Args:
        L:  (B, 1, H, W) in [0, 1].
        ab: (B, 2, H, W) in [-1, 1].
    Returns:
        (B, 3, H, W) pseudo-RGB in approximately [0, 1].
    """
    ab_scaled = (ab + 1.0) / 2.0  # [-1,1] -> [0,1]
    return torch.cat([L, ab_scaled], dim=1)  # (B, 3, H, W)


class HistogramLoss(nn.Module):
    """
    Differentiable colour-histogram matching loss for the *ab* channels.

    PSNR and L1 penalise per-pixel deviations regardless of whether the
    chosen colour is semantically plausible.  A model that regresses to grey
    (a=0, b=0 everywhere) minimises pixel-wise error but produces visually
    wrong results.  Matching the *global* ab distribution forces the model to
    pick colours with the right overall statistics even when per-pixel
    assignment is ambiguous.

    Implementation: soft histogram via Gaussian kernel (differentiable).
    The loss is the mean squared difference between the normalised histograms
    of pred_ab and real_ab over each of the two ab channels.

    Args:
        bins:      Number of histogram bins (default 32).
        bandwidth: Gaussian kernel bandwidth in normalised ab units (default 0.1).
    """

    def __init__(self, bins: int = 32, bandwidth: float = 0.1):
        super().__init__()
        self.bins = bins
        self.bandwidth = bandwidth
        # Fixed bin centres in [-1, 1]
        edges = torch.linspace(-1.0, 1.0, bins)
        self.register_buffer('edges', edges)

    def _soft_hist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a soft (differentiable) normalised histogram for a flat tensor x.
        Returns a (bins,) tensor that sums to 1.
        """
        # x: (N,)  edges: (bins,)
        diff = x.unsqueeze(1) - self.edges.unsqueeze(0)  # (N, bins)
        weights = torch.exp(-0.5 * (diff / self.bandwidth) ** 2)
        hist = weights.sum(dim=0)                         # (bins,)
        return hist / (hist.sum() + 1e-8)

    def forward(self, pred_ab: torch.Tensor, real_ab: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_ab: (B, 2, H, W) predicted ab in [-1, 1].
            real_ab: (B, 2, H, W) ground-truth ab in [-1, 1].
        Returns:
            Scalar loss.
        """
        loss = torch.tensor(0.0, device=pred_ab.device)
        for c in range(2):
            pred_flat = pred_ab[:, c].reshape(-1)
            real_flat = real_ab[:, c].reshape(-1)
            pred_hist = self._soft_hist(pred_flat)
            with torch.no_grad():
                real_hist = self._soft_hist(real_flat)
            loss = loss + torch.mean((pred_hist - real_hist) ** 2)
        return loss
