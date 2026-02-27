import torch
import torch.nn as nn

class Block(nn.Module):
    """
    Базовий будівельний блок Дискримінатора:
    Convolution -> Batch Normalization -> Leaky ReLU
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Classifies overlapping NxN patches of the image as real or fake.
    Input: L channel (1 ch) concatenated with ab channels (2 ch) = 3 channels total.
    Output: spatial patch-score map; each value scores the realism of one patch.
    """
    def __init__(self, in_channels=3, features=None):
        super().__init__()
        features = features if features is not None else [64, 128, 256, 512]

        # 1. First layer — no BatchNorm (standard PatchGAN design)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Intermediate downsampling blocks: 64 -> 128 -> 256 -> 512
        layers = []
        in_c = features[0]
        for feature in features[1:]:
            layers.append(Block(in_c, feature, stride=2))
            in_c = feature

        # 3. Stride-1 block — refines features without changing spatial size
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=4, stride=1, padding=1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(in_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.intermediate = nn.Sequential(*layers)

        # 4. Final layer: 512 -> 1 (one realism score per patch)
        self.final = nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        """
        x: L channel   (Batch, 1, H, W)
        y: ab channels (Batch, 2, H, W) — real or generated
        """
        input_concat = torch.cat([x, y], dim=1)  # (Batch, 3, H, W)
        x = self.initial(input_concat)
        x = self.intermediate(x)
        return self.final(x)