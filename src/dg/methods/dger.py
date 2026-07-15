"""DGER objectives, including the paper's alternating Algorithm 1 update."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from ..models.dger_modules import DGERModules, gradient_reverse
from ..models.mnist_cnn import MNISTCNN
from ..training.losses import class_balanced_cross_entropy
from .base import DomainGeneralizationMethod


TensorBatch = Mapping[str, Tensor]


@dataclass(frozen=True)
class DGERDomainEpisode:
    """Fresh own-domain and other-domain samples for one Algorithm 1 branch."""

    own: TensorBatch
    others: tuple[TensorBatch, ...]


@dataclass(frozen=True)
class DGERIteration:
    """All independently sampled data consumed by one outer Algorithm 1 iteration."""

    main: TensorBatch
    episodes: tuple[DGERDomainEpisode, ...]


def sum_domain_cross_entropy(
    logits: Tensor, labels: Tensor, domains: Tensor, num_domains: int,
) -> Tensor:
    """Sum one mean cross-entropy expectation for every source domain."""
    losses = []
    for domain_id in range(num_domains):
        mask = domains == domain_id
        if not mask.any():
            raise ValueError(f"Algorithm 1 requires samples from source domain {domain_id}.")
        losses.append(functional.cross_entropy(logits[mask], labels[mask]))
    return torch.stack(losses).sum()


def _set_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class DGER(DomainGeneralizationMethod):
    def __init__(self, num_domains: int, optimizer_kwargs: dict[str, object], alpha_1: float = .5, alpha_2: float = .005, alpha_3: float = .01, auxiliary_lr: float | None = None, domain_reduction: str = "sum") -> None:
        super().__init__()
        if domain_reduction not in {"sum", "mean"}:
            raise ValueError("domain_reduction must be 'sum' or 'mean'.")
        self.network, self.auxiliaries = MNISTCNN(), DGERModules(num_domains)
        self.alpha_1, self.alpha_2, self.alpha_3 = alpha_1, alpha_2, alpha_3
        self.domain_reduction = domain_reduction
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

    def _freeze_all(self) -> None:
        # Subclasses such as DGNT own additional modules. Freezing the complete
        # method makes every Algorithm 1 phase boundary explicit.
        _set_grad(self, False)

    def _zero_all_optimizers(self) -> None:
        self.main_optimizer.zero_grad(set_to_none=True)
        self.stabilizer_optimizer.zero_grad(set_to_none=True)

    def _classification_adversarial_step(
        self, batch: TensorBatch, pair_batch: TensorBatch | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Algorithm 1: update F, T, and D using the first two Eq. 10 terms."""
        self._freeze_all()
        _set_grad(self.network, True)
        _set_grad(self.auxiliaries.discriminator, True)
        output = self.network(batch["image"])
        num_domains = len(self.auxiliaries.stabilizers)
        classification = sum_domain_cross_entropy(
            output.logits, batch["label"], batch["domain"], num_domains,
        )
        domain_logits = self.auxiliaries.discriminator(gradient_reverse(output.features))
        adversarial = sum_domain_cross_entropy(
            domain_logits, batch["domain"], batch["domain"], num_domains,
        )
        if self.domain_reduction == "mean":
            classification = classification / num_domains
            adversarial = adversarial / num_domains
        additional, additional_metrics = self._primary_additional_loss(pair_batch)
        weighted = classification + self.alpha_1 * adversarial + additional
        self._zero_all_optimizers()
        weighted.backward()
        self.main_optimizer.step()
        return (
            classification.detach(), adversarial.detach(), output.logits.detach(),
            {name: value.detach() for name, value in additional_metrics.items()},
        )

    def _primary_additional_loss(
        self, pair_batch: TensorBatch | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Extension point for objectives sharing Algorithm 1's primary step."""
        if pair_batch is not None:
            raise ValueError("Plain DGER does not accept an interpolation pair batch.")
        return next(self.parameters()).new_zeros(()), {}

    def _stabilizer_step(
        self, images: Tensor, labels: Tensor, domains: Tensor, domain_id: int,
    ) -> Tensor:
        """Algorithm 1: fit T_i on a fresh own-domain batch with F frozen."""
        del domains  # The episode already identifies its source domain.
        self._freeze_all()
        stabilizer = self.auxiliaries.stabilizers[domain_id]
        _set_grad(stabilizer, True)
        with torch.no_grad():
            features = self.network.encoder(images)
        raw_loss = functional.cross_entropy(stabilizer(features), labels)
        weighted = self.alpha_3 * raw_loss
        self._zero_all_optimizers()
        weighted.backward()
        self.stabilizer_optimizer.step()
        return raw_loss.detach()

    def _entropy_step(self, batch: TensorBatch, domain_id: int) -> Tensor:
        """Algorithm 1: update F and T'_i adversarially on domain i."""
        self._freeze_all()
        _set_grad(self.network.encoder, True)
        entropy_head = self.auxiliaries.entropy_heads[domain_id]
        _set_grad(entropy_head, True)
        features = self.network.encoder(batch["image"])
        raw_loss = functional.cross_entropy(
            entropy_head(gradient_reverse(features)), batch["label"],
        )
        weighted = self.alpha_2 * raw_loss
        self._zero_all_optimizers()
        weighted.backward()
        self.main_optimizer.step()
        return raw_loss.detach()

    def _cross_domain_step(
        self, other_batches: tuple[TensorBatch, ...], domain_id: int,
    ) -> Tensor:
        """Algorithm 1: update only F so frozen T_i classifies every j != i."""
        if not other_batches:
            raise ValueError("DGER requires at least two source domains.")
        self._freeze_all()
        _set_grad(self.network.encoder, True)
        stabilizer = self.auxiliaries.stabilizers[domain_id]
        losses = []
        for batch in other_batches:
            features = self.network.encoder(batch["image"])
            losses.append(functional.cross_entropy(stabilizer(features), batch["label"]))
        raw_loss = torch.stack(losses).sum()
        weighted = self.alpha_3 * raw_loss
        self._zero_all_optimizers()
        weighted.backward()
        self.main_optimizer.step()
        return raw_loss.detach()

    @staticmethod
    def _domain_ids(batch: TensorBatch) -> set[int]:
        if batch["domain"].numel() == 0:
            raise ValueError("Algorithm 1 batches must not be empty.")
        return {int(domain_id) for domain_id in batch["domain"].unique().tolist()}

    def _validate_paper_iteration(self, iteration: DGERIteration) -> None:
        num_domains = len(self.auxiliaries.stabilizers)
        expected_domains = set(range(num_domains))
        if self._domain_ids(iteration.main) != expected_domains:
            raise ValueError("The main Algorithm 1 batch must cover every source domain exactly.")
        if len(iteration.episodes) != num_domains:
            raise ValueError(f"Expected {num_domains} domain episodes, got {len(iteration.episodes)}.")
        for domain_id, episode in enumerate(iteration.episodes):
            if self._domain_ids(episode.own) != {domain_id}:
                raise ValueError(f"The own batch for domain {domain_id} is misrouted.")
            if len(episode.others) != num_domains - 1:
                raise ValueError(f"Domain {domain_id} requires one batch from every other source domain.")
            other_domains = [self._domain_ids(batch) for batch in episode.others]
            if any(len(domains) != 1 for domains in other_domains):
                raise ValueError("Each cross-domain batch must contain exactly one source domain.")
            routed_domains = {next(iter(domains)) for domains in other_domains}
            if routed_domains != expected_domains - {domain_id}:
                raise ValueError(f"Cross-domain batches for domain {domain_id} are misrouted.")

    def paper_train_step(
        self, iteration: DGERIteration, pair_batch: TensorBatch | None = None,
    ) -> dict[str, float]:
        """Perform one complete outer iteration from paper Algorithm 1."""
        num_domains = len(self.auxiliaries.stabilizers)
        self._validate_paper_iteration(iteration)
        classification, adversarial, logits, additional_metrics = self._classification_adversarial_step(
            iteration.main, pair_batch,
        )
        stabilizer_losses, entropy_losses, cross_domain_losses = [], [], []
        for domain_id, episode in enumerate(iteration.episodes):
            stabilizer_losses.append(self._stabilizer_step(
                episode.own["image"], episode.own["label"],
                episode.own["domain"], domain_id,
            ))
            entropy_losses.append(self._entropy_step(episode.own, domain_id))
            cross_domain_losses.append(self._cross_domain_step(episode.others, domain_id))
        stabilizer = torch.stack(stabilizer_losses).sum()
        entropy = torch.stack(entropy_losses).sum()
        cross_domain = torch.stack(cross_domain_losses).sum()
        additional_total = additional_metrics.get(
            "weighted_interpolation_loss", classification.new_zeros(()),
        )
        weighted_total = (
            classification + self.alpha_1 * adversarial
            + self.alpha_2 * entropy + self.alpha_3 * (stabilizer + cross_domain)
            + additional_total
        )
        return {
            # These phase losses were observed at successive parameter states;
            # their weighted sum is a diagnostic, not one jointly optimized loss.
            "diagnostic_weighted_loss_sum": weighted_total.item(),
            "classification_loss": classification.item(),
            "adversarial_loss": adversarial.item(),
            "stabilizer_fit_loss": stabilizer.item(),
            "entropy_loss": entropy.item(),
            "stabilizing_loss": cross_domain.item(),
            "optimizer_steps": float(1 + 3 * num_domains),
            "accuracy": (logits.argmax(1) == iteration.main["label"]).float().mean().item(),
            **{name: value.item() for name, value in additional_metrics.items()},
        }

    def two_step_train_step(
        self, iteration: DGERIteration, pair_batch: TensorBatch | None = None,
    ) -> dict[str, float]:
        """Group the same Algorithm 1 objectives into two optimizer updates.

        This is a diagnostic schedule ablation: it consumes the identical main
        batch and per-domain episodes as :meth:`paper_train_step`, but combines
        all stabilizer fits into one update and all feature-side objectives into
        a second update.
        """
        self._validate_paper_iteration(iteration)
        num_domains = len(self.auxiliaries.stabilizers)

        self._freeze_all()
        _set_grad(self.auxiliaries.stabilizers, True)
        stabilizer_losses = []
        for domain_id, episode in enumerate(iteration.episodes):
            with torch.no_grad():
                features = self.network.encoder(episode.own["image"])
            stabilizer_losses.append(functional.cross_entropy(
                self.auxiliaries.stabilizers[domain_id](features),
                episode.own["label"],
            ))
        stabilizer = torch.stack(stabilizer_losses).sum()
        self._zero_all_optimizers()
        (self.alpha_3 * stabilizer).backward()
        self.stabilizer_optimizer.step()

        self._freeze_all()
        _set_grad(self.network, True)
        _set_grad(self.auxiliaries.discriminator, True)
        _set_grad(self.auxiliaries.entropy_heads, True)
        output = self.network(iteration.main["image"])
        classification = sum_domain_cross_entropy(
            output.logits, iteration.main["label"], iteration.main["domain"],
            num_domains,
        )
        adversarial = sum_domain_cross_entropy(
            self.auxiliaries.discriminator(gradient_reverse(output.features)),
            iteration.main["domain"], iteration.main["domain"], num_domains,
        )
        if self.domain_reduction == "mean":
            classification = classification / num_domains
            adversarial = adversarial / num_domains

        additional, additional_metrics = self._primary_additional_loss(pair_batch)
        entropy_losses, cross_domain_losses = [], []
        for domain_id, episode in enumerate(iteration.episodes):
            own_features = self.network.encoder(episode.own["image"])
            entropy_losses.append(functional.cross_entropy(
                self.auxiliaries.entropy_heads[domain_id](
                    gradient_reverse(own_features),
                ),
                episode.own["label"],
            ))
            stabilizer_head = self.auxiliaries.stabilizers[domain_id]
            cross_domain_losses.append(torch.stack([
                functional.cross_entropy(
                    stabilizer_head(self.network.encoder(batch["image"])),
                    batch["label"],
                )
                for batch in episode.others
            ]).sum())
        entropy = torch.stack(entropy_losses).sum()
        cross_domain = torch.stack(cross_domain_losses).sum()
        total = (
            classification + self.alpha_1 * adversarial
            + self.alpha_2 * entropy + self.alpha_3 * cross_domain + additional
        )
        self._zero_all_optimizers()
        total.backward()
        self.main_optimizer.step()

        diagnostic_total = total.detach() + self.alpha_3 * stabilizer.detach()
        return {
            "loss": total.item(),
            "diagnostic_weighted_loss_sum": diagnostic_total.item(),
            "classification_loss": classification.item(),
            "adversarial_loss": adversarial.item(),
            "stabilizer_fit_loss": stabilizer.item(),
            "entropy_loss": entropy.item(),
            "stabilizing_loss": cross_domain.item(),
            "optimizer_steps": 2.0,
            "accuracy": (
                output.logits.detach().argmax(1) == iteration.main["label"]
            ).float().mean().item(),
            **{name: value.detach().item() for name, value in additional_metrics.items()},
        }

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
