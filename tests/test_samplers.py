import copy

import numpy as np

from dg.data.samplers import CyclingDomainSampler, SameClassCrossDomainPairSampler


def test_pair_sampler_returns_same_class_different_domains():
    labels = np.tile(np.arange(10), 6)
    domains = np.repeat(np.arange(6), 10)
    sampler = SameClassCrossDomainPairSampler(labels, domains, seed=7)
    left, right = sampler.sample(32)
    assert np.all(labels[left] == labels[right])
    assert np.all(domains[left] != domains[right])


def test_near_balanced_batch_has_exact_reported_size_and_rotates_short_domain():
    from dg.data.samplers import NearBalancedBatchIterator

    domains = {domain: np.arange(domain * 100, (domain + 1) * 100) for domain in range(5)}
    batches = list(NearBalancedBatchIterator(domains, batch_size=64, seed=4, steps=5))

    assert all(len(batch) == 64 for batch in batches)
    counts = [
        [int(((batch >= domain * 100) & (batch < (domain + 1) * 100)).sum()) for domain in range(5)]
        for batch in batches
    ]
    assert all(sorted(per_domain) == [12, 13, 13, 13, 13] for per_domain in counts)
    assert [per_domain.index(12) for per_domain in counts] == [4, 0, 1, 2, 3]


def test_pair_sampler_can_pair_prescribed_left_examples():
    labels = np.tile(np.arange(10), 3)
    domains = np.repeat(np.arange(3), 10)
    sampler = SameClassCrossDomainPairSampler(labels, domains, seed=9)
    prescribed_left = np.array([0, 11, 22])

    left, right = sampler.pair_left(prescribed_left)

    assert np.array_equal(left, prescribed_left)
    assert np.array_equal(labels[left], labels[right])
    assert np.all(domains[left] != domains[right])


def test_pair_sampler_state_restores_the_exact_next_pair():
    labels = np.tile(np.arange(10), 3)
    domains = np.repeat(np.arange(3), 10)
    sampler = SameClassCrossDomainPairSampler(labels, domains, seed=9)
    state = copy.deepcopy(sampler.state_dict())
    expected = sampler.sample(8)
    restored = SameClassCrossDomainPairSampler(labels, domains, seed=999)

    restored.load_state_dict(state)
    actual = restored.sample(8)

    assert all(np.array_equal(left, right) for left, right in zip(actual, expected))


def test_cycling_domain_sampler_state_restores_the_exact_next_draw():
    indices = {0: np.arange(7), 1: np.arange(7, 14)}
    sampler = CyclingDomainSampler(indices, batch_size=3, seed=8)
    sampler.sample(0)
    state = copy.deepcopy(sampler.state_dict())
    expected = (sampler.sample(0), sampler.sample(1))
    restored = CyclingDomainSampler(indices, batch_size=3, seed=999)

    restored.load_state_dict(state)
    actual = (restored.sample(0), restored.sample(1))

    assert all(np.array_equal(left, right) for left, right in zip(actual, expected))
