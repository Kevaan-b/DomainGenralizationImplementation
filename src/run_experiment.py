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
    arguments = {"alpha_1": loss["dger_alpha_1"], "alpha_2": loss["dger_alpha_2"], "alpha_3": loss["dger_alpha_3"], "auxiliary_lr": optimizer.pop("auxiliary_lr", None), "domain_reduction": "sum" if config["track"] == "dger_original" else "mean"}
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
    fold = make_fold(
        cache, config["target_angle"], config["seed"], config["data_budget"],
        validation_fraction=float(config.get("source_validation_fraction", 0.1)),
    )
    run_dir = Path(config["results_root"]) / config["track"] / config["method"] / f"target_{config['target_angle']}_seed_{config['seed']}_budget_{config['data_budget']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    model_config = {
        "encoder": "Conv2d(1,32,5)-ReLU-MaxPool2d(2)-Conv2d(32,64,5)-ReLU-MaxPool2d(2)-Linear(1024,64)-ReLU",
        "classifier": "Linear(64,10)",
        "latent_size": 64,
        "interpolator": "Conv1d(1,64,3,padding=1)-ReLU-Conv1d(64,64,3,padding=1)-ReLU-Conv1d(64,1,3,padding=1)",
    }
    source_count = len(fold.source_angles)
    paper_protocol = ({
        "outer_iterations": int(config["iterations"]),
        "optimizer_steps_per_iteration": 1 + 3 * source_count,
        "feature_updates_per_iteration": 1 + 2 * source_count,
        "sample_exposures_per_iteration": int(config["batch_per_domain"]) * source_count * (source_count + 1),
        "paper_ambiguities": [
            "optimizer, momentum, and weight decay", "batch size and sampling replacement",
            "loss normalization and isolated-step alpha placement", "CNN and auxiliary widths",
            "rotation and preprocessing details", "checkpoint selection", "exact MNIST indices",
            "whether the base subset is redrawn across the 10 runs",
        ],
    } if config["track"] == "dger_original" else None)
    target_protocol = ({
        "epochs": int(config["epochs"]),
        "checkpoint_selection": "best_macro_source_validation",
        "domain_loss_reduction": "mean",
        "dger_update_schedule": "alternating_algorithm_1_phases",
        "reconstruction_choices": [
            "five-point uniform interpolation grid",
            "Conv1d interpolator channels=64, kernel_size=3, padding=1",
            "near-balanced 13/13/13/13/12 allocation for batch size 64",
            "DNT/DGNT batch size 64 interpreted as 64 pairs",
            f"DGER per-domain auxiliary batch size {config['dger_domain_batch_size']}",
            "mean minibatch/domain loss reductions",
            "DGNT interpolation added to the first joint DGER phase only",
            "DGER alpha weights imported from Zhao et al.",
        ],
    } if config["track"] == "target_comparison" else None)
    batch_metadata = (
        {"main_step_batch_size": config["batch_per_domain"] * source_count}
        if paper_protocol is not None else {
            "batch_size": config["batch_size"],
            "pair_batch_size": config["pair_batch_size"],
            "batch_allocation": "near-balanced; the 12-sample source rotates across steps",
        }
    )
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump({**config, **batch_metadata, "source_angles": list(fold.source_angles), "paper_protocol": paper_protocol, "target_protocol": target_protocol, "model": model_config, "environment": environment_metadata(device, config["deterministic"]), "git": _git_metadata()}, sort_keys=True))
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
