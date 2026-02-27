import torch
import torch.nn as nn

class UNetDown(nn.Module):
    """Encoder block: Conv -> BatchNorm -> LeakyReLU."""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """Decoder block: ConvTranspose -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  # concatenate skip connection
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()

        # --- ENCODER ---
        self.down1 = UNetDown(in_channels, 64, normalize=False)   # 256 -> 128
        self.down2 = UNetDown(64, 128)                             # 128 -> 64
        self.down3 = UNetDown(128, 256)                            # 64  -> 32
        self.down4 = UNetDown(256, 512, dropout=0.5)               # 32  -> 16
        self.down5 = UNetDown(512, 512, dropout=0.5)               # 16  -> 8
        self.down6 = UNetDown(512, 512, dropout=0.5)               # 8   -> 4
        self.down7 = UNetDown(512, 512, dropout=0.5)               # 4   -> 2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)  # 2 -> 1 (bottleneck)

        # --- DECODER (with skip connections) ---
        # Each up block receives: its input + skip connection from the matching down block
        self.up1 = UNetUp(512, 512, dropout=0.5)   # bottleneck (512)  + skip down7 (512)
        self.up2 = UNetUp(1024, 512, dropout=0.5)  # 1024 (512 + 512)  + skip down6 (512)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        # Final layer: 128 (64 from up7 + 64 from down1) -> 2 channels (ab)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
            nn.Tanh()  # output range [-1, 1]
        )

    def forward(self, x):
        # Encoder — save outputs for skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)  # bottleneck

        # Decoder — merge with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)