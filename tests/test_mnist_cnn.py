import torch

from dg.models.mnist_cnn import MNISTCNN


def test_mnist_cnn_returns_required_shapes():
    model = MNISTCNN()
    output = model(torch.randn(3, 1, 28, 28))
    assert output.features.shape == (3, 64)
    assert output.logits.shape == (3, 10)
