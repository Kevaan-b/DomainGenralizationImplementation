"""Leakage-free class-stratified source-domain partitions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray


def stratified_source_split(
    labels: np.ndarray, seed: int, budget: float = 1.0,
    validation_fraction: float = 0.1,
) -> SplitIndices:
    """Create a 90/10 split after a proportional per-class source budget.

    Budgeting before splitting ensures train and validation both represent the
    same requested data fraction.  At least one validation sample is retained
    whenever a selected class has two or more observations.
    """
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1].")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    generator = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label).copy()
        generator.shuffle(candidates)
        selected_count = max(1, int(round(len(candidates) * budget)))
        selected = candidates[:selected_count]
        validation_count = (
            max(1, int(round(selected_count * validation_fraction)))
            if validation_fraction > 0 and selected_count > 1 else 0
        )
        validation.extend(selected[:validation_count])
        train.extend(selected[validation_count:])
    return SplitIndices(np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64))
