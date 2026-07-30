from torch import nn

from model.layers import GradientReversalLayer


class AdversarialHead(nn.Module):
    """Path B: routes bottleneck features through the GRL so that training this head
    to classify Fitzpatrick skin tone pushes the shared encoder to become invariant
    to it, rather than predictive of it."""

    def __init__(self, in_channels, num_classes=6, lambda_=1.0, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bottleneck):
        x = self.grl(bottleneck)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
