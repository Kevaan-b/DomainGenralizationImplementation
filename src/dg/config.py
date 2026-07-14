"""Validation for the explicit RotatedMNIST experiment contract."""
from __future__ import annotations

import math
from typing import Any

from .data.rotated_mnist import ANGLES

METHODS = frozenset({"deepall", "dnt", "dger", "dgnt"})
DATA_BUDGETS = frozenset({1.0, 0.2, 0.1, 0.05})


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Reject silently incomparable or malformed benchmark runs at the CLI boundary."""
    required = {"track", "method", "seed", "target_angle", "data_budget", "data_root", "results_root", "dataset_seed", "angles", "device", "deterministic", "optimizer", "loss"}
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
    if config["track"] == "dger_original":
        if int(config.get("batch_per_domain", 0)) != 12:
            raise ValueError("The original-DGER reconstruction requires 12 examples per domain.")
        if config["method"] != "dger" or int(config.get("iterations", 0)) != 3000:
            raise ValueError("The dger_original track is DGER-only and requires exactly 3000 iterations.")
        if float(config["optimizer"].get("lr", 0)) != 1e-4 or float(config["optimizer"].get("auxiliary_lr", 0)) != 1e-5:
            raise ValueError("The dger_original track requires F/T/D lr=1e-4 and T_i/T'_i lr=1e-5.")
        if float(config["data_budget"]) != 1.0 or float(config.get("source_validation_fraction", -1)) != 0.0:
            raise ValueError("The dger_original track trains on every source example without a validation holdout.")
        if config.get("update_schedule") != "algorithm_1" or config.get("checkpoint_selection") != "final_iteration":
            raise ValueError("The dger_original track requires Algorithm 1 updates and final-iteration evaluation.")
        if int(config.get("checkpoint_interval", 0)) < 1:
            raise ValueError("The dger_original checkpoint interval must be positive.")
        alphas = (
            float(config["loss"].get("dger_alpha_1", -1)),
            float(config["loss"].get("dger_alpha_2", -1)),
            float(config["loss"].get("dger_alpha_3", -1)),
        )
        if alphas != (0.5, 0.005, 0.01):
            raise ValueError("The dger_original track requires paper loss weights (0.5, 0.005, 0.01).")
    elif config["track"] != "target_comparison":
        raise ValueError("track must be target_comparison or dger_original.")
    else:
        if int(config.get("batch_size", 0)) != 64:
            raise ValueError("The target paper reports minibatch size 64.")
        if int(config.get("pair_batch_size", 0)) != 64:
            raise ValueError("DNT/DGNT require 64 paired items per paper minibatch.")
        if int(config.get("dger_domain_batch_size", 0)) < 1:
            raise ValueError("A positive DGER per-domain batch size is required.")
        if int(config.get("epochs", 0)) != 100:
            raise ValueError("The target_comparison track requires exactly 100 epochs.")
        if config.get("update_schedule") != "method_faithful":
            raise ValueError("The target_comparison track requires method-faithful updates.")
        optimizer = config["optimizer"]
        optimizer_contract = (
            optimizer.get("name") == "sgd"
            and float(optimizer.get("lr", -1)) == .001
            and float(optimizer.get("momentum", -1)) == .9
            and float(optimizer.get("weight_decay", -1)) == .001
        )
        if not optimizer_contract:
            raise ValueError("The target paper requires SGD(lr=.001, momentum=.9, weight_decay=.001).")
        if config["method"] in {"dnt", "dgnt"}:
            loss = config["loss"]
            if float(loss.get("interpolation_lambda", -1)) != 1.0:
                raise ValueError("RotatedMNIST DNT/DGNT require interpolation lambda 1.")
            if loss.get("interpolation_policy") != "uniform_grid":
                raise ValueError("The primary DNT/DGNT reconstruction requires interpolation_policy=uniform_grid.")
            weights = tuple(float(weight) for weight in loss.get("interpolation_weights", ()))
            if not all(math.isfinite(weight) for weight in weights):
                raise ValueError("Interpolation weights must be finite.")
            if len(weights) < 2 or weights[0] != 0.0 or weights[-1] != 1.0:
                raise ValueError("The uniform interpolation grid must include endpoints 0 and 1.")
            differences = [right - left for left, right in zip(weights, weights[1:])]
            if any(difference <= 0 for difference in differences) or max(differences) - min(differences) > 1e-8:
                raise ValueError("Interpolation weights must be a strictly increasing uniform grid.")
        if config["method"] in {"dger", "dgnt"}:
            alpha_keys = ("dger_alpha_1", "dger_alpha_2", "dger_alpha_3")
            if any(key not in config["loss"] for key in alpha_keys):
                raise ValueError("DGER/DGNT require all three DGER alpha weights.")
            if any(
                not math.isfinite(float(config["loss"][key]))
                or float(config["loss"][key]) < 0
                for key in alpha_keys
            ):
                raise ValueError("DGER alpha weights must be finite and nonnegative.")
