"""Loss primitives with explicit gradient routing for all methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class InterpolationLoss:
    total: Tensor
    path: Tensor
    endpoint: Tensor


def interpolation_path(start: Tensor, displacement: Tensor, weights: Iterable[float]) -> tuple[Tensor, ...]:
    return tuple(start + float(weight) * displacement for weight in weights)


def endpoint_loss(
    displacement: Tensor, expected_displacement: Tensor,
    normalization: str = "none",
) -> Tensor:
    """Equation 4's expected Euclidean endpoint error."""
    error = displacement - expected_displacement
    if error.ndim < 2:
        raise ValueError("Endpoint displacements must include batch and feature dimensions.")
    if normalization not in {"none", "sqrt_latent"}:
        raise ValueError("Endpoint normalization must be none or sqrt_latent.")
    flattened = error.flatten(start_dim=1)
    loss = torch.linalg.vector_norm(flattened, ord=2, dim=1).mean()
    if normalization == "sqrt_latent":
        loss = loss / flattened.shape[1] ** .5
    return loss


def interpolation_loss(
    classifier: nn.Module, start: Tensor, end: Tensor, labels: Tensor,
    interpolator: nn.Module, weights: Iterable[float],
    endpoint_normalization: str = "none",
) -> InterpolationLoss:
    """Classify every prescribed path point and enforce endpoint consistency."""
    displacement = interpolator(end - start)
    points = interpolation_path(start, displacement, weights)
    path = torch.stack([functional.cross_entropy(classifier(point), labels) for point in points]).mean()
    endpoint = endpoint_loss(displacement, end - start, endpoint_normalization)
    return InterpolationLoss(total=path + endpoint, path=path, endpoint=endpoint)


def class_balanced_cross_entropy(logits: Tensor, labels: Tensor, num_classes: int = 10) -> Tensor:
    """Mean per-present-class CE, avoiding majority classes dominating auxiliaries."""
    losses = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            losses.append(functional.cross_entropy(logits[mask], labels[mask]))
    if not losses:
        raise ValueError("Class-balanced CE requires at least one example.")
    return torch.stack(losses).mean()
