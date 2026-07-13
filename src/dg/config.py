"""Validation for the explicit RotatedMNIST experiment contract."""
from __future__ import annotations

from typing import Any

from .data.rotated_mnist import ANGLES

METHODS = frozenset({"deepall", "dnt", "dger", "dgnt"})
DATA_BUDGETS = frozenset({1.0, 0.2, 0.1, 0.05})


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Reject silently incomparable or malformed benchmark runs at the CLI boundary."""
    required = {"track", "method", "seed", "target_angle", "data_budget", "data_root", "results_root", "dataset_seed", "angles", "batch_per_domain", "device", "deterministic", "optimizer", "loss"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Configuration is missing required keys: {', '.join(missing)}")
    if config["method"] not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}.")
    if tuple(config["angles"]) != ANGLES:
        raise ValueError(f"RotatedMNIST requires canonical angles {list(ANGLES)}.")
    if config["target_angle"] not in ANGLES:
        raise ValueError("target_angle must be one of the canonical angles.")
    if float(config["data_budget"]) not in DATA_BUDGETS:
        raise ValueError(f"data_budget must be one of {sorted(DATA_BUDGETS, reverse=True)}.")
    if int(config["batch_per_domain"]) != 12:
        raise ValueError("The primary reproduction requires 12 examples per source domain (effective batch 60).")
    if config["track"] == "dger_original":
        if config["method"] != "dger" or int(config.get("iterations", 0)) != 3000:
            raise ValueError("The dger_original track is DGER-only and requires exactly 3000 iterations.")
        if float(config["optimizer"].get("lr", 0)) != 1e-4 or float(config["optimizer"].get("auxiliary_lr", 0)) != 1e-5:
            raise ValueError("The dger_original track requires F/T/D lr=1e-4 and T_i/T'_i lr=1e-5.")
    elif config["track"] != "target_comparison":
        raise ValueError("track must be target_comparison or dger_original.")
    elif int(config.get("epochs", 0)) != 100:
        raise ValueError("The target_comparison track requires exactly 100 epochs.")
