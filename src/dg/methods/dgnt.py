"""DGER plus DNT interpolation, with auxiliaries excluded from interpolation gradients."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import torch
from torch import Tensor

from ..models.interpolator import LatentInterpolator
from ..training.losses import interpolation_loss
from .dger import DGER, TensorBatch, _set_grad


class DGNT(DGER):
    def __init__(self, num_domains: int, optimizer_kwargs: dict[str, object], alpha_1: float = .5, alpha_2: float = .005, alpha_3: float = .01, interpolation_lambda: float = 1., weights: Iterable[float] = (0., .25, .5, .75, 1.), auxiliary_lr: float | None = None, domain_reduction: str = "sum", interpolation_mode: str = "learned", endpoint_normalization: str = "none", endpoint_loss_mode: str = "mean_sample_l2") -> None:
        super().__init__(num_domains, optimizer_kwargs, alpha_1, alpha_2, alpha_3, auxiliary_lr, domain_reduction)
        self.interpolator = LatentInterpolator(mode=interpolation_mode)
        primary_parameters = list(self.network.parameters()) + list(self.auxiliaries.discriminator.parameters()) + list(self.interpolator.parameters())
        entropy_parameters = list(self.auxiliaries.entropy_heads.parameters())
        main_groups: list[dict[str, object]] = [{"params": primary_parameters}, {"params": entropy_parameters}]
        if auxiliary_lr is not None:
            main_groups[1]["lr"] = auxiliary_lr
        self.main_optimizer = torch.optim.SGD(main_groups, **optimizer_kwargs)
        self.interpolation_lambda, self.weights = interpolation_lambda, tuple(weights)
        self.interpolation_mode = interpolation_mode
        self.endpoint_normalization = endpoint_normalization
        self.endpoint_loss_mode = endpoint_loss_mode

    def _primary_additional_loss(
        self, pair_batch: TensorBatch | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Add DNT to DGER's joint F/T/D phase without touching auxiliaries."""
        if pair_batch is None:
            raise ValueError("DGNT needs a same-class cross-domain pair batch.")
        _set_grad(self.interpolator, True)
        start = self.network.encoder(pair_batch["left_image"])
        end = self.network.encoder(pair_batch["right_image"])
        interpolation = interpolation_loss(
            self.network.classifier, start, end, pair_batch["label"],
            self.interpolator, self.weights, self.endpoint_normalization,
            self.endpoint_loss_mode,
        )
        weighted = self.interpolation_lambda * interpolation.total
        return weighted, {
            "weighted_interpolation_loss": weighted,
            "interpolation_loss": interpolation.total,
            "path_loss": interpolation.path,
            "endpoint_loss": interpolation.endpoint,
        }

    def train_step(self, batch: Mapping[str, Tensor], pair_batch: Mapping[str, Tensor] | None = None) -> dict[str, float]:
        if pair_batch is None:
            raise ValueError("DGNT needs a same-class cross-domain pair batch.")
        stabilizer = self._train_stabilizers(batch["image"], batch["label"], batch["domain"])
        dger_total, components, logits = self._main_loss(batch["image"], batch["label"], batch["domain"])
        _set_grad(self.interpolator, True)
        start = self.network(pair_batch["left_image"]).features
        end = self.network(pair_batch["right_image"]).features
        interpolation = interpolation_loss(
            self.network.classifier, start, end, pair_batch["label"],
            self.interpolator, self.weights, self.endpoint_normalization,
            self.endpoint_loss_mode,
        )
        total = dger_total + self.interpolation_lambda * interpolation.total
        self.main_optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.main_optimizer.step()
        return {"loss": total.item(), "stabilizer_fit_loss": stabilizer.item(), **{key: value.item() for key, value in components.items()}, "interpolation_loss": interpolation.total.item(), "path_loss": interpolation.path.item(), "endpoint_loss": interpolation.endpoint.item(), "accuracy": (logits.argmax(1) == batch["label"]).float().mean().item()}
