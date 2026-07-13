"""Pooled, domain-balanced ERM baseline."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as functional

from ..models.mnist_cnn import MNISTCNN
from .base import DomainGeneralizationMethod


class DeepAll(DomainGeneralizationMethod):
    def __init__(self, optimizer_kwargs: dict[str, object]) -> None:
        super().__init__()
        self.network = MNISTCNN()
        self.optimizer = torch.optim.SGD(self.network.parameters(), **optimizer_kwargs)

    @classmethod
    def create(cls, **optimizer_kwargs: object) -> "DeepAll":
        return cls(dict(optimizer_kwargs))

    def train_step(self, batch: Mapping[str, Tensor], pair_batch: Mapping[str, Tensor] | None = None) -> dict[str, float]:
        output = self.network(batch["image"])
        loss = functional.cross_entropy(output.logits, batch["label"])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "classification_loss": loss.item(), "accuracy": (output.logits.argmax(1) == batch["label"]).float().mean().item()}

    def predict(self, images: Tensor) -> Tensor:
        return self.network(images).logits
