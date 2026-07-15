import pytest
import torch
from torch.nn import functional as functional

from dg.models.interpolator import LatentInterpolator
from dg.training.losses import endpoint_loss, interpolation_loss, interpolation_path


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
    assert endpoint_loss(displacement, displacement).item() == 0.0


def test_identity_displacement_reaches_the_second_endpoint():
    start, end = torch.randn(3, 64), torch.randn(3, 64)
    endpoint = interpolation_path(start, end - start, (0.0, 1.0))[-1]
    # start + (end - start) can accumulate one float32 rounding step near zero.
    assert torch.allclose(endpoint, end, atol=1e-6, rtol=1e-5)


def test_endpoint_loss_is_mean_per_example_l2_norm_not_elementwise_mse():
    displacement = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    expected_displacement = torch.zeros_like(displacement)

    loss = endpoint_loss(displacement, expected_displacement)

    # The paper writes an L2 norm inside an expectation. The first example has
    # norm 5 and the second norm 0, hence the minibatch expectation is 2.5.
    assert torch.allclose(loss, torch.tensor(2.5))


def test_reported_interpolator_has_three_convolutional_layers_and_preserves_vectors():
    interpolator = LatentInterpolator(latent_size=64)
    convolution_types = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    convolutions = [
        module for module in interpolator.modules()
        if isinstance(module, convolution_types)
    ]
    displacement = torch.randn(4, 64)

    transformed = interpolator(displacement)

    assert len(convolutions) == 3
    assert transformed.shape == displacement.shape


def test_historical_mlp_interpolator_has_exact_three_by_64_architecture():
    interpolator = LatentInterpolator(latent_size=64, mode="mlp_3x64")
    displacement = torch.randn(4, 64)

    transformed = interpolator(displacement)

    assert [type(module) for module in interpolator.network] == [
        torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear,
        torch.nn.ReLU, torch.nn.Linear,
    ]
    linear_layers = [
        module for module in interpolator.network
        if isinstance(module, torch.nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (64, 64), (64, 64), (64, 64),
    ]
    assert sum(parameter.numel() for parameter in interpolator.parameters()) == 12_480
    assert transformed.shape == displacement.shape


def test_historical_endpoint_mse_is_exact_elementwise_mean_squared_error():
    displacement = torch.tensor([[3.0, 4.0], [0.0, 12.0]])
    expected_displacement = torch.zeros_like(displacement)

    loss = endpoint_loss(
        displacement, expected_displacement, mode="mse_mean_all",
    )

    assert torch.equal(loss, functional.mse_loss(displacement, expected_displacement))
    assert loss.item() == pytest.approx(42.25)


def test_dnt_classification_loss_uses_left_endpoints_from_the_paired_batch():
    from dg.methods.dnt import DNT

    torch.manual_seed(17)
    method = DNT({"lr": 0.0, "momentum": 0.0, "weight_decay": 0.0})
    ordinary_batch = {
        "image": torch.zeros(2, 1, 28, 28),
        "label": torch.tensor([0, 0]),
        "domain": torch.tensor([0, 1]),
    }
    pair_batch = {
        "left_image": torch.ones(2, 1, 28, 28),
        "right_image": -torch.ones(2, 1, 28, 28),
        "label": torch.tensor([8, 9]),
    }
    with torch.no_grad():
        expected = functional.cross_entropy(
            method.network(pair_batch["left_image"]).logits,
            pair_batch["label"],
        )

    metrics = method.train_step(ordinary_batch, pair_batch)

    assert metrics["classification_loss"] == pytest.approx(expected.item())


def test_sqrt_latent_endpoint_normalization_removes_dimension_scaling():
    displacement = torch.ones(2, 64)
    expected_displacement = torch.zeros_like(displacement)

    loss = endpoint_loss(
        displacement, expected_displacement, normalization="sqrt_latent",
    )

    assert torch.allclose(loss, torch.tensor(1.0))


def test_identity_interpolator_returns_displacement_without_trainable_branch():
    interpolator = LatentInterpolator(mode="identity")
    displacement = torch.randn(4, 64)

    transformed = interpolator(displacement)

    assert torch.equal(transformed, displacement)
    assert not any(parameter.requires_grad for parameter in interpolator.parameters())


def test_residual_interpolator_is_identity_initialized_but_trainable():
    interpolator = LatentInterpolator(mode="residual")
    displacement = torch.randn(4, 64)

    transformed = interpolator(displacement)

    assert torch.equal(transformed, displacement)
    assert any(parameter.requires_grad for parameter in interpolator.parameters())


def test_lambda_zero_ablation_reduces_dnt_objective_to_left_classification():
    from dg.methods.dnt import DNT

    torch.manual_seed(4)
    method = DNT(
        {"lr": 0.0, "momentum": 0.0, "weight_decay": 0.0}, loss_weight=0.0,
    )
    pair_batch = {
        "left_image": torch.randn(4, 1, 28, 28),
        "right_image": torch.randn(4, 1, 28, 28),
        "label": torch.tensor([0, 1, 2, 3]),
    }

    metrics = method.train_step({}, pair_batch)

    assert metrics["loss"] == pytest.approx(metrics["classification_loss"])
