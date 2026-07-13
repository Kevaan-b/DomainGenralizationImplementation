import torch

from dg.models.interpolator import LatentInterpolator
from dg.training.losses import interpolation_loss, interpolation_path


def test_interpolation_endpoints_and_gradients():
    encoder = torch.nn.Linear(4, 64)
    classifier = torch.nn.Linear(64, 10)
    interpolator = LatentInterpolator()
    x, x_prime = torch.randn(4, 4), torch.randn(4, 4)
    z, z_prime = encoder(x), encoder(x_prime)
    delta = interpolator(z_prime - z)
    path = interpolation_path(z, delta, (0.0, 1.0))
    assert torch.allclose(path[0], z)
    loss = interpolation_loss(classifier, z, z_prime, torch.tensor([1, 2, 3, 4]), interpolator, (0., .5, 1.))
    loss.total.backward()
    assert encoder.weight.grad is not None
    assert classifier.weight.grad is not None
    assert next(interpolator.parameters()).grad is not None


def test_endpoint_loss_is_zero_for_identity_displacement():
    displacement = torch.randn(2, 64)
    from dg.training.losses import endpoint_loss
    assert endpoint_loss(displacement, displacement).item() == 0.0


def test_identity_displacement_reaches_the_second_endpoint():
    start, end = torch.randn(3, 64), torch.randn(3, 64)
    endpoint = interpolation_path(start, end - start, (0.0, 1.0))[-1]
    assert torch.allclose(endpoint, end)
