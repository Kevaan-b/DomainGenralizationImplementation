import pytest
import torch
from torch.nn import functional as functional

from dg.models.dger_modules import DGERModules, gradient_reverse


def _two_domain_iteration():
    from dg.methods.dger import DGERDomainEpisode, DGERIteration

    main = {
        "image": torch.randn(6, 1, 28, 28),
        "label": torch.tensor([0, 1, 2, 0, 1, 2]),
        "domain": torch.tensor([0, 0, 0, 1, 1, 1]),
    }
    own = tuple({key: value[main["domain"] == domain_id] for key, value in main.items()} for domain_id in range(2))
    return DGERIteration(
        main=main,
        episodes=tuple(DGERDomainEpisode(own=own[domain_id], others=(own[1 - domain_id],)) for domain_id in range(2)),
    )


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


def test_dger_uses_algorithm_one_optimizer_step_order(monkeypatch):
    from dg.methods.dger import DGER

    method = DGER(2, {"lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0})
    counts = {"main": 0, "stabilizer": 0}
    original_main_step = method.main_optimizer.step
    original_stabilizer_step = method.stabilizer_optimizer.step

    def main_step(*args, **kwargs):
        counts["main"] += 1
        return original_main_step(*args, **kwargs)

    def stabilizer_step(*args, **kwargs):
        counts["stabilizer"] += 1
        return original_stabilizer_step(*args, **kwargs)

    monkeypatch.setattr(method.main_optimizer, "step", main_step)
    monkeypatch.setattr(method.stabilizer_optimizer, "step", stabilizer_step)
    method.paper_train_step(_two_domain_iteration())

    assert counts == {"main": 1 + 2 * 2, "stabilizer": 2}


def test_dger_algorithm_one_has_exact_gradient_owner_order(monkeypatch):
    from dg.methods.dger import DGER

    method = DGER(2, {"lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0})
    owner_modules = {
        "F": method.network.encoder,
        "T": method.network.classifier,
        "D": method.auxiliaries.discriminator,
        "T0": method.auxiliaries.stabilizers[0],
        "T1": method.auxiliaries.stabilizers[1],
        "Tprime0": method.auxiliaries.entropy_heads[0],
        "Tprime1": method.auxiliaries.entropy_heads[1],
    }
    events = []

    def record_step(original_step):
        def wrapped(*args, **kwargs):
            gradient_owners = {
                name for name, module in owner_modules.items()
                if any(parameter.grad is not None for parameter in module.parameters())
            }
            enabled_owners = {
                name for name, module in owner_modules.items()
                if any(parameter.requires_grad for parameter in module.parameters())
            }
            events.append((gradient_owners, enabled_owners))
            return original_step(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(method.main_optimizer, "step", record_step(method.main_optimizer.step))
    monkeypatch.setattr(method.stabilizer_optimizer, "step", record_step(method.stabilizer_optimizer.step))
    method.paper_train_step(_two_domain_iteration())

    expected = [
        {"F", "T", "D"},
        {"T0"}, {"F", "Tprime0"}, {"F"},
        {"T1"}, {"F", "Tprime1"}, {"F"},
    ]
    assert [gradient_owners for gradient_owners, _ in events] == expected
    assert [enabled_owners for _, enabled_owners in events] == expected


def test_domain_loss_is_sum_of_domain_expectations():
    from dg.methods.dger import sum_domain_cross_entropy

    logits = torch.tensor([
        [2.0, -1.0], [0.0, 1.0],
        [1.0, 0.0], [-1.0, 2.0],
    ])
    labels = torch.tensor([0, 1, 0, 1])
    domains = torch.tensor([0, 0, 1, 1])
    expected = (
        functional.cross_entropy(logits[:2], labels[:2])
        + functional.cross_entropy(logits[2:], labels[2:])
    )

    assert torch.allclose(sum_domain_cross_entropy(logits, labels, domains, 2), expected)


def test_target_dgnt_uses_domain_mean_so_lambda_one_has_consistent_scale():
    from dg.methods.dgnt import DGNT
    from dg.methods.dger import sum_domain_cross_entropy

    method = DGNT(
        2, {"lr": 0.0, "momentum": 0.0, "weight_decay": 0.0},
        domain_reduction="mean", weights=(0.0, 1.0),
    )
    iteration = _two_domain_iteration()
    pair_batch = {
        "left_image": iteration.main["image"][:4],
        "right_image": torch.randn(4, 1, 28, 28),
        "label": iteration.main["label"][:4],
    }
    with torch.no_grad():
        output = method.network(iteration.main["image"])
        expected_classification = sum_domain_cross_entropy(
            output.logits, iteration.main["label"], iteration.main["domain"], 2,
        ) / 2

    metrics = method.paper_train_step(iteration, pair_batch)

    assert metrics["classification_loss"] == pytest.approx(expected_classification.item())
    assert metrics["weighted_interpolation_loss"] == pytest.approx(
        metrics["interpolation_loss"]
    )


def test_alpha_three_scales_each_stabilizer_fit_update():
    from dg.methods.dger import DGER

    torch.manual_seed(5)
    unscaled = DGER(2, {"lr": 1e-2, "momentum": 0.0, "weight_decay": 0.0}, alpha_3=1.0)
    scaled = DGER(2, {"lr": 1e-2, "momentum": 0.0, "weight_decay": 0.0}, alpha_3=0.25)
    scaled.load_state_dict(unscaled.state_dict())
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 0, 1])
    domains = torch.tensor([0, 0, 1, 1])
    before = next(unscaled.auxiliaries.stabilizers[0].parameters()).detach().clone()

    unscaled._stabilizer_step(images, labels, domains, domain_id=0)
    scaled._stabilizer_step(images, labels, domains, domain_id=0)
    full_delta = next(unscaled.auxiliaries.stabilizers[0].parameters()).detach() - before
    scaled_delta = next(scaled.auxiliaries.stabilizers[0].parameters()).detach() - before

    assert torch.allclose(scaled_delta, 0.25 * full_delta, atol=1e-7, rtol=1e-5)


def test_cross_domain_loss_sums_each_other_domain_expectation():
    from dg.methods.dger import DGER

    method = DGER(3, {"lr": 0.0, "momentum": 0.0, "weight_decay": 0.0})
    batches = tuple({
        "image": torch.randn(3, 1, 28, 28),
        "label": torch.tensor([0, 1, 2]),
        "domain": torch.full((3,), domain_id, dtype=torch.long),
    } for domain_id in (1, 2))
    with torch.no_grad():
        expected = sum(
            functional.cross_entropy(
                method.auxiliaries.stabilizers[0](method.network.encoder(batch["image"])),
                batch["label"],
            ) for batch in batches
        )

    actual = method._cross_domain_step(batches, domain_id=0)

    assert torch.allclose(actual, expected)


def test_algorithm_one_rejects_misrouted_domain_episodes():
    from dg.methods.dger import DGER, DGERDomainEpisode, DGERIteration

    method = DGER(2, {"lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0})
    valid = _two_domain_iteration()
    wrong_own = {**valid.episodes[0].own, "domain": torch.ones(3, dtype=torch.long)}
    malformed = DGERIteration(
        main=valid.main,
        episodes=(DGERDomainEpisode(wrong_own, valid.episodes[0].others), valid.episodes[1]),
    )

    with pytest.raises(ValueError, match="own batch for domain 0"):
        method.paper_train_step(malformed)


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


def test_dgnt_paper_step_preserves_dger_order_and_adds_interpolation_to_main_update(monkeypatch):
    from dg.methods.dgnt import DGNT

    method = DGNT(
        2, {"lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0},
        weights=(0.0, 0.5, 1.0),
    )
    pair_batch = {
        "left_image": torch.randn(4, 1, 28, 28),
        "right_image": torch.randn(4, 1, 28, 28),
        "label": torch.tensor([0, 1, 2, 3]),
    }
    owner_modules = {
        "F": method.network.encoder,
        "T": method.network.classifier,
        "D": method.auxiliaries.discriminator,
        "T0": method.auxiliaries.stabilizers[0],
        "T1": method.auxiliaries.stabilizers[1],
        "Tprime0": method.auxiliaries.entropy_heads[0],
        "Tprime1": method.auxiliaries.entropy_heads[1],
        "Tpsi": method.interpolator,
    }
    events = []

    def record_step(optimizer_name, original_step):
        def wrapped(*args, **kwargs):
            gradient_owners = {
                name for name, module in owner_modules.items()
                if any(parameter.grad is not None for parameter in module.parameters())
            }
            enabled_owners = {
                name for name, module in owner_modules.items()
                if any(parameter.requires_grad for parameter in module.parameters())
            }
            events.append((optimizer_name, gradient_owners, enabled_owners))
            return original_step(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(
        method.main_optimizer, "step",
        record_step("main", method.main_optimizer.step),
    )
    monkeypatch.setattr(
        method.stabilizer_optimizer, "step",
        record_step("stabilizer", method.stabilizer_optimizer.step),
    )

    metrics = method.paper_train_step(
        _two_domain_iteration(), pair_batch=pair_batch,
    )

    expected = [
        ("main", {"F", "T", "D", "Tpsi"}),
        ("stabilizer", {"T0"}),
        ("main", {"F", "Tprime0"}),
        ("main", {"F"}),
        ("stabilizer", {"T1"}),
        ("main", {"F", "Tprime1"}),
        ("main", {"F"}),
    ]
    assert [(name, gradients) for name, gradients, _ in events] == expected
    assert [(name, enabled) for name, _, enabled in events] == expected
    assert "interpolation_loss" in metrics
