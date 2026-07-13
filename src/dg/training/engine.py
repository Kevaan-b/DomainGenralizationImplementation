"""Shared train/validate/checkpoint loop for all methods and both tracks."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.rotated_mnist import RotatedMNISTCache, RotatedMNISTSubset
from ..data.samplers import BalancedBatchIterator, SameClassCrossDomainPairSampler
from ..data.splits import stratified_source_split
from .checkpointing import save_checkpoint, write_json
from .metrics import evaluate


@dataclass(frozen=True)
class FoldData:
    train: RotatedMNISTSubset
    validation: RotatedMNISTSubset
    target: RotatedMNISTSubset
    domain_positions: dict[int, np.ndarray]
    pair_sampler: SameClassCrossDomainPairSampler
    source_angles: tuple[int, ...]
    local_domains: dict[int, int]


def make_fold(cache: RotatedMNISTCache, target_angle: int, seed: int, budget: float) -> FoldData:
    target_domain = int((cache.angles == target_angle).nonzero(as_tuple=True)[0].item())
    source_domains = tuple(index for index in range(len(cache.angles)) if index != target_domain)
    train_pairs, validation_pairs = [], []
    for domain in source_domains:
        split = stratified_source_split(cache.labels[domain].numpy(), seed + domain, budget)
        train_pairs.extend((domain, int(index)) for index in split.train)
        validation_pairs.extend((domain, int(index)) for index in split.validation)
    train = RotatedMNISTSubset(cache, train_pairs)
    positions: dict[int, list[int]] = {local_domain: [] for local_domain in range(len(source_domains))}
    local_lookup = {source: local for local, source in enumerate(source_domains)}
    for position, (domain, _) in enumerate(train.pairs):
        positions[local_lookup[domain]].append(position)
    labels = np.array([int(train[position]["label"]) for position in range(len(train))])
    domains = np.array([local_lookup[domain] for domain, _ in train.pairs])
    sampler = SameClassCrossDomainPairSampler(labels, domains, seed + 10_000)
    target_size = int(cache.images.shape[1])
    return FoldData(train, RotatedMNISTSubset(cache, validation_pairs), RotatedMNISTSubset(cache, [(target_domain, index) for index in range(target_size)]), {key: np.asarray(value) for key, value in positions.items()}, sampler, tuple(int(cache.angles[index]) for index in source_domains), local_lookup)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate(dataset: RotatedMNISTSubset, positions: np.ndarray, device: torch.device, local_domains: dict[int, int]) -> dict[str, torch.Tensor]:
    examples = [dataset[int(position)] for position in positions]
    non_blocking = device.type == "cuda"
    return {"image": torch.stack([example["image"] for example in examples]).to(device, non_blocking=non_blocking),
            "label": torch.stack([example["label"] for example in examples]).to(device, non_blocking=non_blocking),
            "domain": torch.tensor([local_domains[int(example["domain"])] for example in examples], dtype=torch.long).to(device, non_blocking=non_blocking)}


def _pair_batch(dataset: RotatedMNISTSubset, left: np.ndarray, right: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    left_examples, right_examples = [dataset[int(index)] for index in left], [dataset[int(index)] for index in right]
    labels = torch.stack([example["label"] for example in left_examples])
    if not torch.equal(labels, torch.stack([example["label"] for example in right_examples])):
        raise RuntimeError("Pair sampler violated same-class constraint.")
    non_blocking = device.type == "cuda"
    return {"left_image": torch.stack([example["image"] for example in left_examples]).to(device, non_blocking=non_blocking),
            "right_image": torch.stack([example["image"] for example in right_examples]).to(device, non_blocking=non_blocking), "label": labels.to(device, non_blocking=non_blocking)}


class TrainingEngine:
    def __init__(self, method, fold: FoldData, configuration: dict[str, Any], run_dir: Path, device: torch.device) -> None:  # type: ignore[no-untyped-def]
        self.method, self.fold, self.configuration, self.run_dir, self.device = method.to(device), fold, configuration, run_dir, device
        workers = int(configuration.get("num_workers", 0))
        loader_options = {"num_workers": workers, "worker_init_fn": _seed_worker, "generator": torch.Generator().manual_seed(int(configuration["seed"])), "pin_memory": device.type == "cuda"}
        self.validation_loader = DataLoader(fold.validation, batch_size=256, shuffle=False, **loader_options)
        self.target_loader = DataLoader(fold.target, batch_size=256, shuffle=False, **loader_options)
        self.local_domains = fold.local_domains
        self.best_accuracy, self.global_step = float("-inf"), 0

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        optimizers = {name: value.state_dict() for name, value in vars(self.method).items() if isinstance(value, torch.optim.Optimizer)}
        return {"model": self.method.state_dict(), "optimizers": optimizers, "epoch": epoch, "global_step": self.global_step,
                "best_source_validation_accuracy": self.best_accuracy, "configuration": self.configuration,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}

    def run(self) -> dict[str, Any]:
        iteration_limit = self.configuration.get("iterations")
        batch_per_domain = int(self.configuration["batch_per_domain"])
        steps_per_epoch = math.ceil(max(len(indices) for indices in self.fold.domain_positions.values()) / batch_per_domain)
        epochs = int(self.configuration.get("epochs", math.ceil(int(iteration_limit) / steps_per_epoch) if iteration_limit is not None else 1))
        metrics_path = self.run_dir / "metrics.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as metrics_file:
            for epoch in range(1, epochs + 1):
                self.method.train()
                epoch_metrics: list[dict[str, float]] = []
                iterator = BalancedBatchIterator(self.fold.domain_positions, batch_per_domain, int(self.configuration["seed"]) + epoch)
                for positions in iterator:
                    batch = _collate(self.fold.train, positions, self.device, self.local_domains)
                    pair = None
                    if self.configuration["method"] in {"dnt", "dgnt"}:
                        left, right = self.fold.pair_sampler.sample(len(positions))
                        pair = _pair_batch(self.fold.train, left, right, self.device)
                    epoch_metrics.append(self.method.train_step(batch, pair))
                    self.global_step += 1
                    if iteration_limit is not None and self.global_step >= int(iteration_limit):
                        break
                validation = evaluate(self.method, self.validation_loader, self.device)
                validation_by_angle = {}
                for domain, angle in enumerate(self.fold.train.cache.angles.tolist()):
                    if int(angle) in self.fold.source_angles:
                        subset = RotatedMNISTSubset(self.fold.train.cache, [pair for pair in self.fold.validation.pairs if pair[0] == domain])
                        validation_by_angle[str(angle)] = evaluate(self.method, DataLoader(subset, batch_size=256, shuffle=False, num_workers=0, pin_memory=self.device.type == "cuda"), self.device)
                target = evaluate(self.method, self.target_loader, self.device)
                mean_metrics = {key: sum(metric[key] for metric in epoch_metrics) / len(epoch_metrics) for key in epoch_metrics[0]}
                macro_accuracy = sum(metric["accuracy"] for metric in validation_by_angle.values()) / len(validation_by_angle)
                record = {"epoch": epoch, "global_step": self.global_step, "train": mean_metrics, "source_validation": validation, "source_validation_by_angle": validation_by_angle, "source_validation_macro_accuracy": macro_accuracy, "target_analysis_only": target, "best_epoch_so_far": self.best_accuracy}
                if macro_accuracy > self.best_accuracy:
                    self.best_accuracy = macro_accuracy
                    record["best_epoch_so_far"] = epoch
                    save_checkpoint(self.run_dir / "best_source_val.pt", self._checkpoint_payload(epoch))
                    write_json(self.run_dir / "best_source_val.json", {"epoch": epoch, "source_validation": validation, "source_validation_by_angle": validation_by_angle, "source_validation_macro_accuracy": macro_accuracy})
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                save_checkpoint(self.run_dir / "last.pt", self._checkpoint_payload(epoch))
                if iteration_limit is not None and self.global_step >= int(iteration_limit):
                    break
        # Report the held-out target only for the source-selected checkpoint.
        best_payload = torch.load(self.run_dir / "best_source_val.pt", map_location=self.device, weights_only=False)
        self.method.load_state_dict(best_payload["model"])
        return {"best_source_validation_accuracy": self.best_accuracy, "target": evaluate(self.method, self.target_loader, self.device)}
