"""DGER's two-phase auxiliary update with explicit frozen-parameter boundaries."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as functional

from ..models.dger_modules import DGERModules, gradient_reverse
from ..models.mnist_cnn import MNISTCNN
from ..training.losses import class_balanced_cross_entropy
from .base import DomainGeneralizationMethod


def _set_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class DGER(DomainGeneralizationMethod):
    def __init__(self, num_domains: int, optimizer_kwargs: dict[str, object], alpha_1: float = .5, alpha_2: float = .005, alpha_3: float = .01, auxiliary_lr: float | None = None) -> None:
        super().__init__()
        self.network, self.auxiliaries = MNISTCNN(), DGERModules(num_domains)
        self.alpha_1, self.alpha_2, self.alpha_3 = alpha_1, alpha_2, alpha_3
        primary_parameters = list(self.network.parameters()) + list(self.auxiliaries.discriminator.parameters())
        entropy_parameters = list(self.auxiliaries.entropy_heads.parameters())
        aux_parameters = self.auxiliaries.stabilizers.parameters()
        main_groups: list[dict[str, object]] = [{"params": primary_parameters}, {"params": entropy_parameters}]
        if auxiliary_lr is not None:
            main_groups[1]["lr"] = auxiliary_lr
        self.main_optimizer = torch.optim.SGD(main_groups, **optimizer_kwargs)
        aux_kwargs = dict(optimizer_kwargs)
        if auxiliary_lr is not None:
            aux_kwargs["lr"] = auxiliary_lr
        self.stabilizer_optimizer = torch.optim.SGD(aux_parameters, **aux_kwargs)

    def _train_stabilizers(self, images: Tensor, labels: Tensor, domains: Tensor) -> Tensor:
        _set_grad(self.network, False)
        _set_grad(self.auxiliaries, False)
        _set_grad(self.auxiliaries.stabilizers, True)
        with torch.no_grad():
            features = self.network(images).features
        losses = []
        for domain_id, head in enumerate(self.auxiliaries.stabilizers):
            mask = domains == domain_id
            if mask.any():
                losses.append(class_balanced_cross_entropy(head(features[mask]), labels[mask]))
        # Each head has its own optimizer slot; summing gives every head the
        # same effective learning rate as a separate own-domain update.
        loss = torch.stack(losses).sum()
        self.stabilizer_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.stabilizer_optimizer.step()
        return loss.detach()

    def _main_loss(self, images: Tensor, labels: Tensor, domains: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        _set_grad(self.network, True)
        _set_grad(self.auxiliaries.discriminator, True)
        _set_grad(self.auxiliaries.entropy_heads, True)
        _set_grad(self.auxiliaries.stabilizers, False)
        output = self.network(images)
        classification = functional.cross_entropy(output.logits, labels)
        counts = torch.bincount(domains, minlength=len(self.auxiliaries.stabilizers)).float()
        sample_weights = counts.sum() / counts.clamp_min(1)[domains]
        adversarial = functional.cross_entropy(self.auxiliaries.discriminator(gradient_reverse(output.features)), domains, reduction="none")
        adversarial = (adversarial * sample_weights).sum() / sample_weights.sum()
        entropy_losses, stabilizer_losses = [], []
        for domain_id, (entropy_head, stabilizer) in enumerate(zip(self.auxiliaries.entropy_heads, self.auxiliaries.stabilizers)):
            own, other = domains == domain_id, domains != domain_id
            if own.any():
                entropy_losses.append(class_balanced_cross_entropy(entropy_head(gradient_reverse(output.features[own])), labels[own]))
            if other.any():
                stabilizer_losses.append(class_balanced_cross_entropy(stabilizer(output.features[other]), labels[other]))
        # Algorithm 1 and the implementation specification define both
        # auxiliaries as sums across source domains, not a domain average.
        entropy = torch.stack(entropy_losses).sum()
        stabilizing = torch.stack(stabilizer_losses).sum()
        total = classification + self.alpha_1 * adversarial + self.alpha_2 * entropy + self.alpha_3 * stabilizing
        return total, {"classification_loss": classification, "adversarial_loss": adversarial, "entropy_loss": entropy, "stabilizing_loss": stabilizing}, output.logits

    def train_step(self, batch: Mapping[str, Tensor], pair_batch: Mapping[str, Tensor] | None = None) -> dict[str, float]:
        stabilizer = self._train_stabilizers(batch["image"], batch["label"], batch["domain"])
        total, components, logits = self._main_loss(batch["image"], batch["label"], batch["domain"])
        self.main_optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.main_optimizer.step()
        return {"loss": total.item(), "stabilizer_fit_loss": stabilizer.item(), **{key: value.item() for key, value in components.items()}, "accuracy": (logits.argmax(1) == batch["label"]).float().mean().item()}

    def predict(self, images: Tensor) -> Tensor:
        return self.network(images).logits
