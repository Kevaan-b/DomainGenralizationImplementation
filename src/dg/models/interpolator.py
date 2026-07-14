"""Latent displacement transformer used by DNT and DGNT."""
from __future__ import annotations

from torch import nn


class LatentInterpolator(nn.Module):
    """Three-layer convolutional interpolator reported for RotatedMNIST.

    The paper does not publish channel or kernel sizes.  We treat each latent
    vector as a one-channel length-``latent_size`` signal and preserve its
    shape through three convolutional layers.
    """

    def __init__(self, latent_size: int = 64, hidden_channels: int = 64) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.network = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, displacement):  # type: ignore[no-untyped-def]
        if displacement.ndim != 2 or displacement.shape[1] != self.latent_size:
            raise ValueError(
                f"Expected displacement shape [batch, {self.latent_size}], "
                f"got {tuple(displacement.shape)}."
            )
        return self.network(displacement.unsqueeze(1)).squeeze(1)
