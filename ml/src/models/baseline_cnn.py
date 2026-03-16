import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Encoder-decoder colorization network with residual skip connections.

    Architecture: 3-level encoder → 3-level decoder, each encoder level is
    concatenated to the corresponding decoder level (skip connections).
    Input:  (B, 1, H, W)  — L channel normalised to [0, 1]
    Output: (B, 2, H, W)  — ab channels normalised to [-1, 1]
    """

    def __init__(self):
        super(BaselineCNN, self).__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )  # -> (B, 64, H/2, W/2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )  # -> (B, 128, H/4, W/4)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )  # -> (B, 256, H/8, W/8)

        # ── Decoder with skip connections ─────────────────────────────────────
        # dec1 receives enc3 output (256 ch)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )  # -> (B, 128, H/4, W/4)

        # dec2 receives cat(dec1, enc2) = 128+128 = 256 ch
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )  # -> (B, 64, H/2, W/2)

        # dec3 receives cat(dec2, enc1) = 64+64 = 128 ch
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),  # output range [-1, 1], matching dataset.py normalisation
        )  # -> (B, 2, H, W)

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(input_l)            # (B, 64,  H/2, W/2)
        e2 = self.enc2(e1)                 # (B, 128, H/4, W/4)
        e3 = self.enc3(e2)                 # (B, 256, H/8, W/8)

        d1 = self.dec1(e3)                 # (B, 128, H/4, W/4)
        d2 = self.dec2(torch.cat([d1, e2], dim=1))  # (B, 64,  H/2, W/2)
        d3 = self.dec3(torch.cat([d2, e1], dim=1))  # (B, 2,   H,   W)
        return d3


if __name__ == "__main__":
    dummy_input = torch.randn(1, 1, 256, 256)
    model = BaselineCNN()
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # expected: (1, 2, 256, 256)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
