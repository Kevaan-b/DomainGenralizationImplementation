"""Small deterministic fixtures that avoid MNIST downloads in unit tests."""
from __future__ import annotations

import torch

from dg.data.rotated_mnist import ANGLES, RotatedMNISTCache


def synthetic_cache(examples_per_class: int = 10) -> RotatedMNISTCache:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(examples_per_class)
    count = len(labels)
    images = torch.linspace(0, 1, count * 28 * 28, dtype=torch.float32).reshape(count, 1, 28, 28)
    return RotatedMNISTCache(
        images=images.unsqueeze(0).repeat(len(ANGLES), 1, 1, 1, 1),
        labels=labels.unsqueeze(0).repeat(len(ANGLES), 1),
        angles=torch.tensor(ANGLES, dtype=torch.long),
        mnist_indices=torch.arange(count, dtype=torch.long).unsqueeze(0).repeat(len(ANGLES), 1),
        dataset_seed=7,
    )
