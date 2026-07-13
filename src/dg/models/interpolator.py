"""Latent displacement transformer used by DNT and DGNT."""
from __future__ import annotations

from torch import nn


class LatentInterpolator(nn.Module):
    """Three-layer vector-compatible nonlinear interpolator from the protocol."""

    def __init__(self, latent_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size, latent_size), nn.ReLU(),
            nn.Linear(latent_size, latent_size), nn.ReLU(),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, displacement):  # type: ignore[no-untyped-def]
        return self.network(displacement)
