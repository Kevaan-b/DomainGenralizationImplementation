"""The shared, deliberately small RotatedMNIST classifier."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelOutput:
    """Classifier output with the latent representation retained for DG losses."""

    features: Tensor
    logits: Tensor


class MNISTCNN(nn.Module):
    """Two-convolution MNIST CNN specified by the experiment protocol."""

    def __init__(self, latent_size: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 4 * 4, latent_size), nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_size, num_classes)

    def forward(self, images: Tensor) -> ModelOutput:
        features = self.encoder(images)
        return ModelOutput(features=features, logits=self.classifier(features))
