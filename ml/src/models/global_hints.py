import torch
import torch.nn as nn
from torchvision import models


class GlobalHintNet(nn.Module):
    """ResNet18-based global colour-hint extractor.

    Args:
        freeze: If True (legacy behaviour), backbone weights are frozen and
                the module always runs in eval mode — suitable when used as a
                fixed feature extractor.  If False, all weights are trainable
                so the network can be jointly fine-tuned with the generator to
                learn colour-relevant features from the L channel.
    """

    def __init__(self, freeze: bool = False):
        super(GlobalHintNet, self).__init__()

        # Load ResNet18 pretrained on ImageNet; strip the final classification
        # head to obtain a 512-dim feature vector.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace the first conv layer: original expects 3-channel RGB input,
        # but we feed a 1-channel L image (replicated to 3 channels at runtime).
        # Re-initialise with the mean of the three colour filters so the
        # pre-trained feature detectors still fire correctly on luminance input.
        original_conv = resnet.conv1  # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            # Average the three colour-channel weights into the new layer.
            new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1))
        resnet.conv1 = new_conv

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) — L channel normalised to [0, 1]
        # ResNet expects 3-channel input: replicate L across all 3.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # -> (B, 3, H, W)

        features = self.backbone(x)              # (B, 512, 1, 1)
        return features.view(features.size(0), -1)  # (B, 512)