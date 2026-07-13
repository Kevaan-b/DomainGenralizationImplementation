"""DGER gradient-reversal and auxiliary heads."""
from __future__ import annotations

from torch import Tensor, nn
from torch.autograd import Function


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, coefficient: float) -> Tensor:  # type: ignore[no-untyped-def]
        ctx.coefficient = coefficient
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:  # type: ignore[no-untyped-def]
        return gradient.neg().mul(ctx.coefficient), None


def gradient_reverse(tensor: Tensor, coefficient: float = 1.0) -> Tensor:
    """Identity forward pass whose backward pass reverses encoder gradients."""
    return _GradientReverse.apply(tensor, coefficient)


class DGERModules(nn.Module):
    """All DGER auxiliaries; source-domain positions are local to each fold."""

    def __init__(self, num_domains: int, latent_size: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.discriminator = nn.Linear(latent_size, num_domains)
        self.stabilizers = nn.ModuleList(nn.Linear(latent_size, num_classes) for _ in range(num_domains))
        self.entropy_heads = nn.ModuleList(nn.Linear(latent_size, num_classes) for _ in range(num_domains))
