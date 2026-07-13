import numpy as np

from dg.data.samplers import SameClassCrossDomainPairSampler


def test_pair_sampler_returns_same_class_different_domains():
    labels = np.tile(np.arange(10), 6)
    domains = np.repeat(np.arange(6), 10)
    sampler = SameClassCrossDomainPairSampler(labels, domains, seed=7)
    left, right = sampler.sample(32)
    assert np.all(labels[left] == labels[right])
    assert np.all(domains[left] != domains[right])
