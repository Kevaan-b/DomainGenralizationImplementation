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


def test_paper_fold_trains_on_every_source_example_without_validation_holdout():
    cache = synthetic_cache(examples_per_class=100)
    fold = make_fold(cache, target_angle=30, seed=5, budget=1.0, validation_fraction=0.0)

    assert len(fold.train) == 5 * cache.images.shape[1]
    assert len(fold.validation) == 0
    target_domain = 2
    for positions in fold.domain_positions.values():
        assert len(positions) == cache.images.shape[1]
        pairs = [fold.train.pairs[int(position)] for position in positions]
        assert len(set(pairs)) == cache.images.shape[1]
        assert all(domain != target_domain for domain, _ in pairs)
        labels = np.array([int(fold.train[int(position)]["label"]) for position in positions])
        assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, 100))
