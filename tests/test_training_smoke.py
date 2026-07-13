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
