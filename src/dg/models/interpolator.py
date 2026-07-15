"""Latent displacement transformer used by DNT and DGNT."""
from __future__ import annotations

from torch import nn


class LatentInterpolator(nn.Module):
    """Three-layer convolutional interpolator reported for RotatedMNIST.

    The paper does not publish channel or kernel sizes.  We treat each latent
    vector as a one-channel length-``latent_size`` signal and preserve its
    shape through three convolutional layers.
    """

    def __init__(
        self, latent_size: int = 64, hidden_channels: int = 64,
        mode: str = "learned",
    ) -> None:
        super().__init__()
        if mode not in {
            "learned", "conv1d_3layer", "mlp_3x64", "identity", "residual",
        }:
            raise ValueError(
                "Interpolator mode must be learned, conv1d_3layer, mlp_3x64, "
                "identity, or residual."
            )
        self.latent_size = latent_size
        self.mode = mode
        if mode == "identity":
            self.network = nn.Identity()
        elif mode == "mlp_3x64":
            self.network = nn.Sequential(
                nn.Linear(latent_size, latent_size), nn.ReLU(),
                nn.Linear(latent_size, latent_size), nn.ReLU(),
                nn.Linear(latent_size, latent_size),
            )
        else:
            self.network = nn.Sequential(
                nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
            )
            if mode == "residual":
                final = self.network[-1]
                if not isinstance(final, nn.Conv1d):
                    raise TypeError("Residual interpolator requires a convolutional final layer.")
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(self, displacement):  # type: ignore[no-untyped-def]
        if displacement.ndim != 2 or displacement.shape[1] != self.latent_size:
            raise ValueError(
                f"Expected displacement shape [batch, {self.latent_size}], "
                f"got {tuple(displacement.shape)}."
            )
        if self.mode == "identity":
            return displacement
        if self.mode == "mlp_3x64":
            return self.network(displacement)
        transformed = self.network(displacement.unsqueeze(1)).squeeze(1)
        return displacement + transformed if self.mode == "residual" else transformed
