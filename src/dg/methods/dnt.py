"""DeepAll with same-class cross-domain latent interpolation robustness."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as functional

from ..models.interpolator import LatentInterpolator
from ..models.mnist_cnn import MNISTCNN
from ..training.losses import interpolation_loss
from .base import DomainGeneralizationMethod


class DNT(DomainGeneralizationMethod):
    def __init__(self, optimizer_kwargs: dict[str, object], loss_weight: float = 1.0, weights: Iterable[float] = (0., .25, .5, .75, 1.), interpolation_mode: str = "learned", endpoint_normalization: str = "none", endpoint_loss_mode: str = "mean_sample_l2", endpoint_weight: float = 1.0) -> None:
        super().__init__()
        self.network, self.interpolator = MNISTCNN(), LatentInterpolator(mode=interpolation_mode)
        self.optimizer = torch.optim.SGD(self.parameters(), **optimizer_kwargs)
        self.loss_weight, self.weights = loss_weight, tuple(weights)
        self.interpolation_mode = interpolation_mode
        self.endpoint_normalization = endpoint_normalization
        self.endpoint_loss_mode = endpoint_loss_mode
        self.endpoint_weight = endpoint_weight

    def train_step(self, batch: Mapping[str, Tensor], pair_batch: Mapping[str, Tensor] | None = None) -> dict[str, float]:
        if pair_batch is None:
            raise ValueError("DNT needs a same-class cross-domain pair batch.")
        del batch  # Algorithm 1 defines both losses on the paired minibatches.
        left_output = self.network(pair_batch["left_image"])
        start = left_output.features
        end = self.network(pair_batch["right_image"]).features
        classification = functional.cross_entropy(left_output.logits, pair_batch["label"])
        interpolated = interpolation_loss(
            self.network.classifier, start, end, pair_batch["label"],
            self.interpolator, self.weights, self.endpoint_normalization,
            self.endpoint_loss_mode, self.endpoint_weight,
        )
        total = classification + self.loss_weight * interpolated.total
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        return {"loss": total.item(), "classification_loss": classification.item(), "interpolation_loss": interpolated.total.item(), "path_loss": interpolated.path.item(), "endpoint_loss": interpolated.endpoint.item(), "weighted_endpoint_loss": interpolated.weighted_endpoint.item(), "accuracy": (left_output.logits.argmax(1) == pair_batch["label"]).float().mean().item()}

    def predict(self, images: Tensor) -> Tensor:
        return self.network(images).logits
