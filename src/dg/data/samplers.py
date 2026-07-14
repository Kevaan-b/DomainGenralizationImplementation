"""Exact-size source batches and same-class cross-domain pair construction."""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np


class SameClassCrossDomainPairSampler:
    """Sample same-label, different-domain index pairs from source training data."""

    def __init__(self, labels: Sequence[int], domains: Sequence[int], seed: int) -> None:
        self.labels = np.asarray(labels)
        self.domains = np.asarray(domains)
        self.rng = np.random.default_rng(seed)
        self.buckets: dict[tuple[int, int], np.ndarray] = {}
        for domain in np.unique(self.domains):
            for label in np.unique(self.labels):
                indices = np.flatnonzero((self.domains == domain) & (self.labels == label))
                if len(indices):
                    self.buckets[(int(domain), int(label))] = indices
        self.valid_labels = tuple(label for label in np.unique(self.labels) if sum((domain, int(label)) in self.buckets for domain in np.unique(self.domains)) >= 2)
        if not self.valid_labels:
            raise ValueError("No class is available in at least two source domains.")

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size < 1:
            raise ValueError("Pair batch size must be positive.")
        left, right = [], []
        for _ in range(batch_size):
            label = int(self.rng.choice(self.valid_labels))
            domains = [domain for domain in np.unique(self.domains) if (int(domain), label) in self.buckets]
            source, destination = self.rng.choice(domains, size=2, replace=False)
            left.append(self.rng.choice(self.buckets[(int(source), label)]))
            right.append(self.rng.choice(self.buckets[(int(destination), label)]))
        return np.asarray(left), np.asarray(right)

    def state_dict(self) -> dict[str, object]:
        return {"rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "rng" not in state:
            raise ValueError("Pair sampler state is missing its RNG state.")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def pair_left(self, left_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Pair prescribed left endpoints with same-class other-domain samples."""
        left = np.asarray(left_indices, dtype=np.int64)
        if left.ndim != 1 or left.size == 0:
            raise ValueError("Left endpoint indices must be a non-empty vector.")
        if left.min() < 0 or left.max() >= len(self.labels):
            raise ValueError("Left endpoint index is outside the source training set.")
        right = []
        unique_domains = np.unique(self.domains)
        for index in left:
            label, source = int(self.labels[index]), int(self.domains[index])
            destinations = [
                int(domain) for domain in unique_domains
                if int(domain) != source and (int(domain), label) in self.buckets
            ]
            if not destinations:
                raise ValueError(
                    f"No other source domain contains label {label} for left index {index}."
                )
            destination = int(self.rng.choice(destinations))
            right.append(int(self.rng.choice(self.buckets[(destination, label)])))
        return left.copy(), np.asarray(right, dtype=np.int64)


class NearBalancedBatchIterator:
    """Exact-size source batches with the remainder rotated across domains."""

    def __init__(
        self, domain_indices: dict[int, np.ndarray], batch_size: int, seed: int,
        steps: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not domain_indices or any(len(indices) == 0 for indices in domain_indices.values()):
            raise ValueError("Every source domain needs training data.")
        if batch_size < len(domain_indices):
            raise ValueError("batch_size must cover every source domain.")
        self.domain_indices = {
            int(domain): np.asarray(indices, dtype=np.int64) for domain, indices in domain_indices.items()
        }
        self.domains = tuple(sorted(self.domain_indices))
        self.batch_size = batch_size
        self.steps = steps if steps is not None else int(np.ceil(
            sum(len(indices) for indices in self.domain_indices.values()) / batch_size
        ))
        if self.steps < 1:
            raise ValueError("steps must be positive")
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[np.ndarray]:
        pools = {domain: self.rng.permutation(indices) for domain, indices in self.domain_indices.items()}
        offsets = {domain: 0 for domain in self.domains}
        base, remainder = divmod(self.batch_size, len(self.domains))
        for step in range(self.steps):
            extra_domains = {
                self.domains[(step + offset) % len(self.domains)] for offset in range(remainder)
            }
            batch: list[int] = []
            for domain in self.domains:
                count = base + int(domain in extra_domains)
                while count:
                    pool = pools[domain]
                    available = len(pool) - offsets[domain]
                    take = min(count, available)
                    batch.extend(int(index) for index in pool[offsets[domain]:offsets[domain] + take])
                    offsets[domain] += take
                    count -= take
                    if offsets[domain] == len(pool):
                        pools[domain] = self.rng.permutation(self.domain_indices[domain])
                        offsets[domain] = 0
            yield self.rng.permutation(np.asarray(batch, dtype=np.int64))


class BalancedBatchIterator:
    """Finite epoch iterator: equal samples per source domain, cycling short sets."""

    def __init__(self, domain_indices: dict[int, np.ndarray], batch_per_domain: int, seed: int) -> None:
        if batch_per_domain < 1:
            raise ValueError("batch_per_domain must be positive")
        self.domain_indices = {key: np.asarray(value) for key, value in domain_indices.items()}
        if any(len(indices) == 0 for indices in self.domain_indices.values()):
            raise ValueError("Every source domain needs training data.")
        self.batch_per_domain = batch_per_domain
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[np.ndarray]:
        longest = max(len(indices) for indices in self.domain_indices.values())
        steps = int(np.ceil(longest / self.batch_per_domain))
        pools = {domain: self.rng.permutation(indices) for domain, indices in self.domain_indices.items()}
        offsets = defaultdict(int)
        for _ in range(steps):
            batch = []
            for domain, indices in self.domain_indices.items():
                chosen = []
                while len(chosen) < self.batch_per_domain:
                    pool = pools[domain]
                    available = len(pool) - offsets[domain]
                    take = min(self.batch_per_domain - len(chosen), available)
                    chosen.extend(pool[offsets[domain]: offsets[domain] + take])
                    offsets[domain] += take
                    if offsets[domain] == len(pool):
                        pools[domain] = self.rng.permutation(indices)
                        offsets[domain] = 0
                batch.extend(chosen)
            yield self.rng.permutation(np.asarray(batch, dtype=np.int64))


class CyclingDomainSampler:
    """Draw fresh fixed-size domain batches, reshuffling only at pool boundaries."""

    def __init__(self, domain_indices: dict[int, np.ndarray], batch_size: int, seed: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if any(len(indices) == 0 for indices in domain_indices.values()):
            raise ValueError("Every source domain needs training data.")
        self.domain_indices = {key: np.asarray(value) for key, value in domain_indices.items()}
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.pools = {domain: self.rng.permutation(indices) for domain, indices in self.domain_indices.items()}
        self.offsets = {domain: 0 for domain in self.domain_indices}

    def state_dict(self) -> dict[str, object]:
        """Return all cursor and RNG state needed for an exact stream resume."""
        return {
            "batch_size": self.batch_size,
            "domain_indices": {domain: indices.copy() for domain, indices in self.domain_indices.items()},
            "pools": {domain: pool.copy() for domain, pool in self.pools.items()},
            "offsets": dict(self.offsets),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("Sampler state batch size does not match this sampler.")
        saved_indices = state["domain_indices"]
        if not isinstance(saved_indices, dict) or set(saved_indices) != set(self.domain_indices):
            raise ValueError("Sampler state source domains do not match this sampler.")
        if any(not np.array_equal(saved_indices[domain], indices) for domain, indices in self.domain_indices.items()):
            raise ValueError("Sampler state indices do not match this sampler.")
        pools, offsets = state["pools"], state["offsets"]
        if not isinstance(pools, dict) or not isinstance(offsets, dict):
            raise ValueError("Sampler state pools and offsets must be mappings.")
        self.pools = {int(domain): np.asarray(pool).copy() for domain, pool in pools.items()}
        self.offsets = {int(domain): int(offset) for domain, offset in offsets.items()}
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def sample(self, domain: int) -> np.ndarray:
        if domain not in self.domain_indices:
            raise ValueError(f"Unknown source domain: {domain}")
        chosen: list[int] = []
        while len(chosen) < self.batch_size:
            pool = self.pools[domain]
            available = len(pool) - self.offsets[domain]
            take = min(self.batch_size - len(chosen), available)
            chosen.extend(pool[self.offsets[domain]: self.offsets[domain] + take])
            self.offsets[domain] += take
            if self.offsets[domain] == len(pool):
                self.pools[domain] = self.rng.permutation(self.domain_indices[domain])
                self.offsets[domain] = 0
        return np.asarray(chosen, dtype=np.int64)
