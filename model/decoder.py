import torch
from torch import nn

from model.encoder import ConvBlock


class Decoder(nn.Module):
    """Path A: U-Net expansive path. Upsamples the bottleneck features, concatenates
    the matching encoder skip connection at each stage, and emits per-pixel burn mask
    logits via a final 1x1 conv (pair with BCEWithLogitsLoss; apply sigmoid at
    inference time)."""

    def __init__(self, base_channels=64, depth=4, out_channels=1):
        super().__init__()
        stage_channels = [base_channels * (2 ** i) for i in range(depth)]
        bottleneck_channels = stage_channels[-1] * 2

        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        in_channels = bottleneck_channels
        for channels in reversed(stage_channels):
            self.up_convs.append(nn.ConvTranspose2d(in_channels, channels, kernel_size=2, stride=2))
            self.conv_blocks.append(ConvBlock(channels * 2, channels))
            in_channels = channels

        self.final_conv = nn.Conv2d(stage_channels[0], out_channels, kernel_size=1)

    def forward(self, bottleneck, skips):
        x = bottleneck
        for up_conv, conv_block, skip in zip(self.up_convs, self.conv_blocks, reversed(skips)):
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = conv_block(x)
        return self.final_conv(x)
