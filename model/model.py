from torch import nn

from model.adversarial_head import AdversarialHead
from model.decoder import Decoder
from model.encoder import Encoder


class DebiasedBurnUNet(nn.Module):
    """Dual-Head Adversarial U-Net: a shared encoder feeds a segmentation decoder
    (Path A) and, through a Gradient Reversal Layer, a Fitzpatrick skin-tone
    classifier (Path B) used to adversarially debias the shared bottleneck features."""

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        depth=4,
        num_tone_classes=6,
        grl_lambda=1.0,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels, depth=depth)
        self.decoder = Decoder(base_channels=base_channels, depth=depth, out_channels=1)

        bottleneck_channels = base_channels * (2 ** depth)
        self.adversarial_head = AdversarialHead(
            in_channels=bottleneck_channels,
            num_classes=num_tone_classes,
            lambda_=grl_lambda,
        )

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        mask = self.decoder(bottleneck, skips)
        tone_logits = self.adversarial_head(bottleneck)
        return mask, tone_logits
