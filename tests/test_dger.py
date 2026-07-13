import torch

from dg.models.dger_modules import DGERModules, gradient_reverse


def test_dger_auxiliary_shapes_and_gradient_reversal():
    modules = DGERModules(num_domains=5)
    features = torch.randn(4, 64, requires_grad=True)
    assert modules.discriminator(features).shape == (4, 5)
    assert modules.stabilizers[0](features).shape == (4, 10)
    assert modules.entropy_heads[0](features).shape == (4, 10)
    gradient_reverse(features).sum().backward()
    assert torch.allclose(features.grad, -torch.ones_like(features))


def test_original_dger_track_assigns_auxiliary_learning_rate_to_both_head_types():
    from dg.methods.dger import DGER
    method = DGER(5, {"lr": 1e-4, "momentum": .9, "weight_decay": .001}, auxiliary_lr=1e-5)
    assert method.main_optimizer.param_groups[0]["lr"] == 1e-4
    assert method.main_optimizer.param_groups[1]["lr"] == 1e-5
    assert method.stabilizer_optimizer.param_groups[0]["lr"] == 1e-5


def test_dger_two_phase_updates_respect_frozen_parameter_boundaries():
    from dg.methods.dger import DGER

    method = DGER(5, {"lr": 1e-3, "momentum": .9, "weight_decay": 0.0})
    images = torch.randn(10, 1, 28, 28)
    labels = torch.arange(10) % 10
    domains = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    encoder_before = next(method.network.encoder.parameters()).detach().clone()
    stabilizer_before = next(method.auxiliaries.stabilizers.parameters()).detach().clone()
    method._train_stabilizers(images, labels, domains)
    assert torch.equal(encoder_before, next(method.network.encoder.parameters()))
    assert not torch.equal(stabilizer_before, next(method.auxiliaries.stabilizers.parameters()))
    stabilizer_before = next(method.auxiliaries.stabilizers.parameters()).detach().clone()
    total, _, _ = method._main_loss(images, labels, domains)
    method.main_optimizer.zero_grad(set_to_none=True)
    total.backward()
    assert next(method.auxiliaries.entropy_heads.parameters()).grad is not None
    method.main_optimizer.step()
    assert not torch.equal(encoder_before, next(method.network.encoder.parameters()))
    assert torch.equal(stabilizer_before, next(method.auxiliaries.stabilizers.parameters()))


def test_dgnt_interpolation_does_not_touch_dger_auxiliaries():
    from dg.methods.dgnt import DGNT
    from dg.training.losses import interpolation_loss

    method = DGNT(5, {"lr": 1e-3, "momentum": .9, "weight_decay": 0.0})
    for parameter in method.auxiliaries.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    start = method.network(torch.randn(2, 1, 28, 28)).features
    end = method.network(torch.randn(2, 1, 28, 28)).features
    interpolation_loss(method.network.classifier, start, end, torch.tensor([0, 1]), method.interpolator, (0.0, 1.0)).total.backward()
    assert all(parameter.grad is None for parameter in method.auxiliaries.parameters())
