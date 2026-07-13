"""Deterministically cached six-domain RotatedMNIST data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform

ANGLES = (0, 15, 30, 45, 60, 75)
ROTATION_POLICY = {"direction": "clockwise", "interpolation": "bilinear", "fill": 0, "range": "[0, 1]"}


@dataclass(frozen=True)
class RotatedMNISTCache:
    images: Tensor
    labels: Tensor
    angles: Tensor
    mnist_indices: Tensor
    dataset_seed: int


def build_or_load_cache(root: str | Path, dataset_seed: int, angles: tuple[int, ...] = ANGLES) -> RotatedMNISTCache:
    """Cache a fixed balanced MNIST subset rotated once with a documented policy."""
    root = Path(root)
    angle_key = "-".join(str(angle) for angle in angles)
    cache_path = root / f"rotated_mnist_seed_{dataset_seed}_angles_{angle_key}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return RotatedMNISTCache(**payload)
    root.mkdir(parents=True, exist_ok=True)
    base = MNIST(root=str(root), train=True, download=True)
    labels = base.targets.numpy()
    rng = np.random.default_rng(dataset_seed)
    selected = np.concatenate([rng.choice(np.flatnonzero(labels == label), 100, replace=False) for label in range(10)])
    selected.sort()
    base_images = base.data[selected].unsqueeze(1).float().div(255.0)
    base_labels = torch.as_tensor(labels[selected], dtype=torch.long)
    rotated = []
    for angle in angles:
        # torchvision angles are counter-clockwise, hence negative is clockwise.
        rotated.append(base_images.clone() if angle == 0 else transform.rotate(base_images, -float(angle), interpolation=InterpolationMode.BILINEAR, fill=0.0))
    payload: dict[str, Any] = {
        "images": torch.stack(rotated).to(torch.float32),
        "labels": base_labels.repeat(len(angles), 1),
        "angles": torch.tensor(angles, dtype=torch.int64),
        "mnist_indices": torch.as_tensor(selected, dtype=torch.int64).repeat(len(angles), 1),
        "dataset_seed": dataset_seed,
    }
    torch.save(payload, cache_path)
    cache_path.with_suffix(".json").write_text(json.dumps({"rotation_policy": ROTATION_POLICY, "angles": angles, "selected_mnist_indices": selected.tolist()}, indent=2))
    return RotatedMNISTCache(**payload)


class RotatedMNISTSubset(Dataset[dict[str, Any]]):
    """A fold subset retaining all domain and original-index metadata."""

    def __init__(self, cache: RotatedMNISTCache, pairs: list[tuple[int, int]]) -> None:
        self.cache, self.pairs = cache, pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        domain, index = self.pairs[item]
        return {"image": (self.cache.images[domain, index] - 0.1307) / 0.3081,
                "label": self.cache.labels[domain, index], "domain": torch.tensor(domain, dtype=torch.long),
                "angle": int(self.cache.angles[domain]), "mnist_index": int(self.cache.mnist_indices[domain, index])}
