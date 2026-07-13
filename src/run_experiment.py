"""CLI entry point for one reproducible RotatedMNIST run."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

from dg.config import validate_experiment_config
from dg.data.rotated_mnist import ROTATION_POLICY, build_or_load_cache
from dg.methods.deepall import DeepAll
from dg.methods.dger import DGER
from dg.methods.dgnt import DGNT
from dg.methods.dnt import DNT
from dg.training.checkpointing import write_json
from dg.training.engine import TrainingEngine, make_fold
from dg.training.reproducibility import environment_metadata, resolve_device, seed_everything


def _git_metadata() -> dict[str, Any]:
    try:
        return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _create_method(config: dict[str, Any], source_domains: int):
    optimizer = dict(config["optimizer"])
    optimizer.pop("name", None)
    loss = config["loss"]
    if config["method"] == "deepall":
        return DeepAll.create(**optimizer)
    if config["method"] == "dnt":
        return DNT(optimizer, loss["interpolation_lambda"], loss["interpolation_weights"])
    arguments = {"alpha_1": loss["dger_alpha_1"], "alpha_2": loss["dger_alpha_2"], "alpha_3": loss["dger_alpha_3"], "auxiliary_lr": optimizer.pop("auxiliary_lr", None)}
    if config["method"] == "dger":
        return DGER(source_domains, optimizer, **arguments)
    if config["method"] == "dgnt":
        return DGNT(source_domains, optimizer, interpolation_lambda=loss["interpolation_lambda"], weights=loss["interpolation_weights"], **arguments)
    raise ValueError(f"Unsupported method: {config['method']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("deepall", "dnt", "dger", "dgnt"))
    parser.add_argument("--target-angle", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-budget", type=float, choices=(1.0, 0.2, 0.1, 0.05))
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text())
    for name, value in (("method", arguments.method), ("target_angle", arguments.target_angle), ("seed", arguments.seed), ("data_budget", arguments.data_budget)):
        if value is not None:
            config[name] = value
    validate_experiment_config(config)
    device = resolve_device(config["device"])
    seed_everything(config["seed"], config["deterministic"])
    cache = build_or_load_cache(config["data_root"], config["dataset_seed"], tuple(config["angles"]))
    fold = make_fold(cache, config["target_angle"], config["seed"], config["data_budget"])
    run_dir = Path(config["results_root"]) / config["track"] / config["method"] / f"target_{config['target_angle']}_seed_{config['seed']}_budget_{config['data_budget']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    model_config = {"encoder": "Conv2d(1,32,5)-ReLU-MaxPool2d(2)-Conv2d(32,64,5)-ReLU-MaxPool2d(2)-Linear(1024,64)-ReLU", "classifier": "Linear(64,10)", "latent_size": 64, "interpolator": "Linear(64,64)-ReLU x2 then Linear(64,64)"}
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump({**config, "source_angles": list(fold.source_angles), "effective_batch": config["batch_per_domain"] * len(fold.source_angles), "model": model_config, "environment": environment_metadata(device, config["deterministic"]), "git": _git_metadata()}, sort_keys=True))
    write_json(run_dir / "indices.json", {
        "selected_mnist_indices": cache.mnist_indices[0].tolist(),
        "source_train": [{"domain": domain, "index": index} for domain, index in fold.train.pairs],
        "source_validation": [{"domain": domain, "index": index} for domain, index in fold.validation.pairs],
        "target_test": [{"domain": domain, "index": index} for domain, index in fold.target.pairs],
        "rotation_policy": ROTATION_POLICY,
    })
    method = _create_method(config, len(fold.source_angles))
    result = TrainingEngine(method, fold, config, run_dir, device).run()
    write_json(run_dir / "final_metrics.json", result)


if __name__ == "__main__":
    main()
