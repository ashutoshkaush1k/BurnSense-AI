import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.layers import GradientReversalLayer


def test_forward_is_identity():
    grl = GradientReversalLayer(lambda_=1.0)
    x = torch.randn(4, 8, requires_grad=True)

    out = grl(x)

    assert torch.equal(out, x)


def test_backward_negates_and_scales_gradient():
    lambda_ = 0.7
    grl = GradientReversalLayer(lambda_=lambda_)
    x = torch.randn(4, 8, requires_grad=True)

    out = grl(x)
    upstream_grad = torch.randn_like(out)
    out.backward(upstream_grad)

    expected_grad = -lambda_ * upstream_grad

    assert torch.allclose(x.grad, expected_grad)
