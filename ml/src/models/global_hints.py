import torch
import torch.nn as nn
from torchvision import models

class GlobalHintNet(nn.Module):
    def __init__(self):
        super(GlobalHintNet, self).__init__()

        # Load ResNet18 pretrained on ImageNet; strip the final classification
        # head to obtain a 512-dim feature vector.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone — weights are not updated during training.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (Batch, 1, H, W) — L channel
        # ResNet expects 3 channels: replicate L across all 3.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # -> (Batch, 3, H, W)

        features = self.backbone(x)  # (Batch, 512, 1, 1)
        return features.view(features.size(0), -1)  # (Batch, 512)