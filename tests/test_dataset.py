import torch

from dg.data import rotated_mnist
from dg.data.rotated_mnist import RotatedMNISTSubset

from .helpers import synthetic_cache


def test_dataset_item_contract_and_shared_indices():
    cache = synthetic_cache(examples_per_class=100)
    assert cache.images.shape == (6, 1000, 1, 28, 28)
    assert cache.labels.shape == (6, 1000)
    assert torch.equal(cache.mnist_indices[0], cache.mnist_indices[-1])
    assert torch.equal(torch.bincount(cache.labels[0], minlength=10), torch.full((10,), 100))
    item = RotatedMNISTSubset(cache, [(0, 0)])[0]
    assert set(item) == {"image", "label", "domain", "angle", "mnist_index"}
    assert item["image"].shape == (1, 28, 28)
    assert item["label"].dtype == torch.long


def test_zero_angle_subset_recovers_cached_base_after_normalization():
    cache = synthetic_cache()
    item = RotatedMNISTSubset(cache, [(0, 12)])[0]
    recovered = item["image"] * 0.3081 + 0.1307
    assert torch.allclose(recovered, cache.images[0, 12])


def test_cache_builder_rotates_once_and_reuses_a_saved_cache(tmp_path, monkeypatch):
    class FakeMNIST:
        def __init__(self, *args, **kwargs):
            self.targets = torch.arange(10).repeat_interleave(100)
            self.data = torch.arange(1000 * 28 * 28, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(1000, 28, 28)

    monkeypatch.setattr(rotated_mnist, "MNIST", FakeMNIST)
    cache = rotated_mnist.build_or_load_cache(tmp_path, dataset_seed=9)
    assert cache.images.shape == (6, 1000, 1, 28, 28)
    assert cache.images.dtype == torch.float32
    assert cache.labels.dtype == torch.long
    assert torch.equal(cache.images[0], FakeMNIST().data.unsqueeze(1).float() / 255.0)
    monkeypatch.setattr(rotated_mnist, "MNIST", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache was not reused")))
    loaded = rotated_mnist.build_or_load_cache(tmp_path, dataset_seed=9)
    assert torch.equal(loaded.images, cache.images)
