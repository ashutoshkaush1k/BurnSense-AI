import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity on the forward pass; negates and scales the gradient on the backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction so it can be used as a standard nn.Module in a forward()."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
