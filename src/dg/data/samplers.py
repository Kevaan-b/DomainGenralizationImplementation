"""Domain-balanced ordinary batches and independent DNT pair sampling."""
from __future__ import annotations

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
        left, right = [], []
        for _ in range(batch_size):
            label = int(self.rng.choice(self.valid_labels))
            domains = [domain for domain in np.unique(self.domains) if (int(domain), label) in self.buckets]
            source, destination = self.rng.choice(domains, size=2, replace=False)
            left.append(self.rng.choice(self.buckets[(int(source), label)]))
            right.append(self.rng.choice(self.buckets[(int(destination), label)]))
        return np.asarray(left), np.asarray(right)


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
