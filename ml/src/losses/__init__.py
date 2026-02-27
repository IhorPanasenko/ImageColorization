"""
Reusable loss functions shared across GAN and Fusion training scripts.
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Adversarial loss using BCEWithLogitsLoss.

    Dynamically creates real/fake label tensors that match the discriminator's
    output shape and device automatically, so it works with any output spatial
    resolution on any device (CPU, CUDA, MPS) without manual device tracking.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.real_label = 1.0
        self.fake_label = 0.0

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        value = self.real_label if target_is_real else self.fake_label
        # torch.full_like mirrors prediction's device, dtype and shape automatically.
        return torch.full_like(prediction, value)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
