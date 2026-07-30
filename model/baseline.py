from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder


class StandardBurnUNet(nn.Module):
    """Ablation baseline: the same shared encoder/decoder as DebiasedBurnUNet, but with
    no adversarial head, GRL, or Path B — a plain segmentation U-Net outputting only
    mask logits."""

    def __init__(self, in_channels=3, base_channels=64, depth=4):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels, depth=depth)
        self.decoder = Decoder(base_channels=base_channels, depth=depth, out_channels=1)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        mask_logits = self.decoder(bottleneck, skips)
        return mask_logits
