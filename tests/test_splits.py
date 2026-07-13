import numpy as np

from dg.data.splits import stratified_source_split
from dg.training.engine import make_fold

from .helpers import synthetic_cache


def test_stratified_split_and_budget_preserve_classes():
    labels = np.repeat(np.arange(10), 100)
    split = stratified_source_split(labels, seed=2, budget=0.2)
    assert len(split.train) == 180
    assert len(split.validation) == 20
    assert np.array_equal(np.bincount(labels[split.train], minlength=10), np.full(10, 18))
    assert np.array_equal(np.bincount(labels[split.validation], minlength=10), np.full(10, 2))


def test_fold_excludes_held_out_domain_from_all_source_partitions():
    fold = make_fold(synthetic_cache(), target_angle=30, seed=5, budget=1.0)
    target_domain = 2
    source_pairs = set(fold.train.pairs) | set(fold.validation.pairs)
    assert all(domain != target_domain for domain, _ in source_pairs)
    assert not set(fold.train.pairs) & set(fold.validation.pairs)
    assert all(domain == target_domain for domain, _ in fold.target.pairs)
