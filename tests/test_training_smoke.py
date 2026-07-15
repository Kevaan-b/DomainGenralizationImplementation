from pathlib import Path

import torch

from dg.methods.deepall import DeepAll
from dg.methods.dger import DGER
from dg.methods.dgnt import DGNT
from dg.methods.dnt import DNT
from dg.training.engine import TrainingEngine, make_fold

from .helpers import synthetic_cache


def _config(method: str) -> dict:
    return {
        "method": method, "seed": 3, "batch_per_domain": 2, "epochs": 1,
        "num_workers": 0, "iterations": 2,
    }


def _method(method: str):
    kwargs = {"lr": 1e-3, "momentum": .9, "weight_decay": 0.0}
    if method == "deepall":
        return DeepAll.create(**kwargs)
    if method == "dnt":
        return DNT(kwargs)
    if method == "dger":
        return DGER(5, kwargs)
    return DGNT(5, kwargs)


def test_two_step_cpu_smoke_creates_source_selected_checkpoints_for_each_method(tmp_path: Path):
    fold = make_fold(synthetic_cache(), target_angle=0, seed=3, budget=1.0)
    for name in ("deepall", "dnt", "dger", "dgnt"):
        run_dir = tmp_path / name
        result = TrainingEngine(_method(name), fold, _config(name), run_dir, torch.device("cpu")).run()
        assert 0.0 <= result["target"]["accuracy"] <= 1.0
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "best_source_val.pt").exists()
        assert (run_dir / "best_source_val.json").exists()


def test_production_batch_path_uses_64_prescribed_left_pairs(tmp_path: Path):
    class SpyDNT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("steps", torch.tensor(0))

        def train_step(self, batch, pair_batch):
            assert len(batch["image"]) == 64
            assert len(pair_batch["left_image"]) == 64
            assert torch.equal(batch["image"], pair_batch["left_image"])
            assert torch.equal(batch["label"], pair_batch["label"])
            assert torch.all(pair_batch["left_domain"] != pair_batch["right_domain"])
            self.steps.add_(1)
            return {"loss": 1.0, "accuracy": 0.0}

        def predict(self, images):
            return torch.zeros(len(images), 10, device=images.device)

    fold = make_fold(synthetic_cache(), target_angle=0, seed=3, budget=1.0)
    configuration = {
        "method": "dnt", "seed": 3, "batch_size": 64,
        "pair_batch_size": 64, "dger_domain_batch_size": 2,
        "epochs": 1, "iterations": 1, "num_workers": 0,
    }
    method = SpyDNT()

    TrainingEngine(
        method, fold, configuration, tmp_path / "production_dnt", torch.device("cpu"),
    ).run()

    assert method.steps.item() == 1


def test_target_dger_and_dgnt_share_algorithm_one_routing(tmp_path: Path):
    class SpyAlternatingMethod(torch.nn.Module):
        def __init__(self, expects_pair: bool):
            super().__init__()
            self.expects_pair = expects_pair
            self.register_buffer("steps", torch.tensor(0))

        def paper_train_step(self, iteration, pair_batch=None):
            assert len(iteration.main["image"]) == 64
            assert (pair_batch is not None) is self.expects_pair
            self.steps.add_(1)
            return {"loss": 1.0, "accuracy": 0.0}

        def train_step(self, *args, **kwargs):
            raise AssertionError("Target DGER/DGNT must use alternating paper_train_step")

        def predict(self, images):
            return torch.zeros(len(images), 10, device=images.device)

    fold = make_fold(synthetic_cache(), target_angle=0, seed=3, budget=1.0)
    for name in ("dger", "dgnt"):
        method = SpyAlternatingMethod(expects_pair=name == "dgnt")
        configuration = {
            "method": name, "seed": 3, "batch_size": 64,
            "pair_batch_size": 64, "dger_domain_batch_size": 2,
            "epochs": 1, "iterations": 1, "num_workers": 0,
        }
        TrainingEngine(
            method, fold, configuration, tmp_path / name, torch.device("cpu"),
        ).run()
        assert method.steps.item() == 1


def test_original_dger_track_reports_the_exact_final_iteration(tmp_path: Path, monkeypatch):
    fold = make_fold(
        synthetic_cache(examples_per_class=2), target_angle=0, seed=3,
        budget=1.0, validation_fraction=0.0,
    )
    configuration = {
        "track": "dger_original", "method": "dger", "seed": 3,
        "batch_per_domain": 2, "num_workers": 0, "iterations": 2,
        "update_schedule": "algorithm_1", "checkpoint_selection": "final_iteration",
    }
    run_dir = tmp_path / "paper"
    evaluated_datasets = []

    def fake_evaluate(method, loader, device):
        del method, device
        evaluated_datasets.append(loader.dataset)
        return {"accuracy": 0.5, "mean_per_class_accuracy": 0.5, "cross_entropy": 1.0}

    monkeypatch.setattr("dg.training.engine.evaluate", fake_evaluate)

    result = TrainingEngine(
        DGER(5, {"lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0}),
        fold, configuration, run_dir, torch.device("cpu"),
    ).run()
    checkpoint = torch.load(run_dir / "paper_final.pt", map_location="cpu", weights_only=False)

    assert checkpoint["global_step"] == 2
    assert checkpoint["progress"] == {"unit": "outer_iteration", "value": 2}
    assert checkpoint["optimizer_step_count"] == 32
    assert checkpoint["sampler"] is not None
    assert "epoch" not in checkpoint
    assert result["global_step"] == 2
    assert "best_source_validation_accuracy" not in result
    assert not (run_dir / "best_source_val.pt").exists()
    assert evaluated_datasets == [fold.target]


def test_original_track_runs_3000_outer_iterations_and_evaluates_that_state(tmp_path: Path, monkeypatch):
    class SpyPaperMethod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("completed_iterations", torch.tensor(0, dtype=torch.long))

        def paper_train_step(self, iteration):
            del iteration
            self.completed_iterations.add_(1)
            return {"optimizer_steps": 16.0}

        def predict(self, images):
            return torch.zeros(len(images), 10, device=images.device)

    fold = make_fold(
        synthetic_cache(examples_per_class=2), target_angle=0, seed=3,
        budget=1.0, validation_fraction=0.0,
    )
    configuration = {
        "track": "dger_original", "method": "dger", "seed": 3,
        "batch_per_domain": 2, "num_workers": 0, "iterations": 3000,
        "checkpoint_interval": 3000, "update_schedule": "algorithm_1",
        "checkpoint_selection": "final_iteration",
    }
    method = SpyPaperMethod()
    engine = TrainingEngine(method, fold, configuration, tmp_path / "paper_3000", torch.device("cpu"))
    monkeypatch.setattr(engine, "_paper_iteration", lambda sampler: None)
    evaluated_at = []

    def fake_evaluate(current_method, loader, device):
        del loader, device
        evaluated_at.append(int(current_method.completed_iterations))
        return {"accuracy": 0.5}

    monkeypatch.setattr("dg.training.engine.evaluate", fake_evaluate)
    result = engine.run()
    checkpoint = torch.load(tmp_path / "paper_3000" / "paper_final.pt", weights_only=False)

    assert result["global_step"] == 3000
    assert int(checkpoint["model"]["completed_iterations"]) == 3000
    assert evaluated_at == [3000]


def test_paper_iteration_samples_fresh_batches_with_correct_domain_routing():
    fold = make_fold(
        synthetic_cache(examples_per_class=2), target_angle=0, seed=3,
        budget=1.0, validation_fraction=0.0,
    )
    engine = TrainingEngine(
        DGER(5, {"lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0}),
        fold, {"seed": 3, "num_workers": 0}, Path("unused"), torch.device("cpu"),
    )

    class RecordingSampler:
        def __init__(self):
            self.calls = []

        def sample(self, domain):
            self.calls.append(domain)
            return fold.domain_positions[domain][:1]

    sampler = RecordingSampler()
    iteration = engine._paper_iteration(sampler)

    assert sampler.calls[:5] == [0, 1, 2, 3, 4]
    assert {domain: sampler.calls.count(domain) for domain in range(5)} == {domain: 6 for domain in range(5)}
    for domain_id, episode in enumerate(iteration.episodes):
        assert set(episode.own["domain"].tolist()) == {domain_id}
        assert {int(batch["domain"].item()) for batch in episode.others} == set(range(5)) - {domain_id}


def test_two_step_ablation_routes_dger_family_to_episode_preserving_two_step(tmp_path: Path):
    class SpyTwoStepMethod(torch.nn.Module):
        def __init__(self, expects_pair: bool):
            super().__init__()
            self.expects_pair = expects_pair
            self.register_buffer("steps", torch.tensor(0))

        def two_step_train_step(self, iteration, pair_batch=None):
            assert len(iteration.main["image"]) == 64
            assert (pair_batch is not None) is self.expects_pair
            self.steps.add_(1)
            return {"loss": 1.0, "accuracy": 0.0}

        def train_step(self, *args, **kwargs):
            raise AssertionError("two_step must preserve Algorithm 1 episodes")

        def paper_train_step(self, *args, **kwargs):
            raise AssertionError("two_step must not use the alternating paper route")

        def predict(self, images):
            return torch.zeros(len(images), 10, device=images.device)

    fold = make_fold(synthetic_cache(), target_angle=0, seed=3, budget=1.0)
    for name in ("dger", "dgnt"):
        method = SpyTwoStepMethod(expects_pair=name == "dgnt")
        configuration = {
            "method": name, "seed": 3, "batch_size": 64,
            "pair_batch_size": 64, "dger_domain_batch_size": 2,
            "epochs": 1, "iterations": 1, "num_workers": 0,
            "update_schedule": "two_step", "ablation": {"name": "two_step"},
        }

        TrainingEngine(
            method, fold, configuration, tmp_path / name, torch.device("cpu"),
        ).run()

        assert method.steps.item() == 1
