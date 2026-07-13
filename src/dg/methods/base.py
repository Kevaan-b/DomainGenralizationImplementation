"""Common interface for all benchmark methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import Tensor, nn


class DomainGeneralizationMethod(nn.Module, ABC):
    @abstractmethod
    def train_step(self, batch: Mapping[str, Tensor], pair_batch: Mapping[str, Tensor] | None = None) -> dict[str, float]:
        """Perform one complete optimizer update and return scalar train metrics."""

    @abstractmethod
    def predict(self, images: Tensor) -> Tensor:
        """Return main-classifier logits; never uses training-only auxiliaries."""
