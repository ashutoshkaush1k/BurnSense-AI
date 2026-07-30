from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers, the basic U-Net building block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """U-Net contracting path. Returns the bottleneck feature map plus the per-stage
    skip connections (shallowest first) consumed by the segmentation decoder."""

    def __init__(self, in_channels=3, base_channels=64, depth=4):
        super().__init__()
        stage_channels = [base_channels * (2 ** i) for i in range(depth)]

        self.stages = nn.ModuleList()
        prev_channels = in_channels
        for channels in stage_channels:
            self.stages.append(ConvBlock(prev_channels, channels))
            prev_channels = channels

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ConvBlock(stage_channels[-1], stage_channels[-1] * 2)

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
            x = self.pool(x)
        bottleneck = self.bottleneck(x)
        return bottleneck, skips
