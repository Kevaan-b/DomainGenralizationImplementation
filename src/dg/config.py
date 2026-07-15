"""Validation for the explicit RotatedMNIST experiment contract."""
from __future__ import annotations

import math
from typing import Any

from .data.rotated_mnist import ANGLES

METHODS = frozenset({"deepall", "dnt", "dger", "dgnt"})
DATA_BUDGETS = frozenset({1.0, 0.2, 0.1, 0.05})


def _validate_endpoint_history_contract(config: dict[str, Any]) -> None:
    ablation = config["ablation"]
    method = config["method"]
    name = ablation.get("name")
    if method in {"deepall", "dger"}:
        expected_name = f"{method}_shared_control"
        if name != expected_name or ablation.get("changed_knobs") != []:
            raise ValueError(
                f"Endpoint-history {method} requires the unchanged {expected_name}."
            )
        if config.get("update_schedule") != "method_faithful":
            raise ValueError("Endpoint-history controls require method_faithful updates.")
        if ablation.get("factorial_member") is not False:
            raise ValueError("Endpoint-history controls are not factorial members.")
        if method == "dger":
            alphas = tuple(float(config["loss"].get(key, -1)) for key in (
                "dger_alpha_1", "dger_alpha_2", "dger_alpha_3",
            ))
            if alphas != (0.5, 0.005, 0.01):
                raise ValueError("The DGER shared control requires fixed paper weights.")
        return

    cells = {
        "hist_mlp_mse": (
            "mlp_3x64", "mse_mean_all",
            ["loss.interpolation_mode", "loss.endpoint_loss"],
        ),
        "hist_mlp_l2": (
            "mlp_3x64", "mean_sample_l2", ["loss.interpolation_mode"],
        ),
        "hist_conv_mse": (
            "conv1d_3layer", "mse_mean_all", ["loss.endpoint_loss"],
        ),
        "hist_conv_l2": ("conv1d_3layer", "mean_sample_l2", []),
    }
    if name not in cells:
        raise ValueError(f"{name!r} is not a recognized endpoint-history cell.")
    architecture, endpoint_mode, changed_knobs = cells[name]
    loss = config["loss"]
    actual = (loss.get("interpolation_mode"), loss.get("endpoint_loss"))
    if actual != (architecture, endpoint_mode):
        raise ValueError(
            f"{name} requires interpolation_mode={architecture} and "
            f"endpoint_loss={endpoint_mode}."
        )
    if ablation.get("changed_knobs") != changed_knobs:
        raise ValueError(
            f"{name} requires factorial changed_knobs={changed_knobs}."
        )
    if ablation.get("factorial_member") is not True:
        raise ValueError("Endpoint-history cells must be marked as factorial members.")
    if float(loss.get("interpolation_lambda", -1)) != 1.0:
        raise ValueError("Endpoint-history cells require interpolation lambda 1.")
    if loss.get("endpoint_normalization", "none") != "none":
        raise ValueError("Endpoint-history cells do not normalize the endpoint loss.")
    if tuple(float(value) for value in loss.get("interpolation_weights", ())) != (
        0.0, 0.25, 0.5, 0.75, 1.0,
    ):
        raise ValueError("Endpoint-history cells require the fixed five-point grid.")
    if method == "dgnt":
        alphas = tuple(float(loss.get(key, -1)) for key in (
            "dger_alpha_1", "dger_alpha_2", "dger_alpha_3",
        ))
        if alphas != (0.5, 0.005, 0.01):
            raise ValueError("Endpoint-history DGNT cells require fixed DGER weights.")
    if config.get("update_schedule") != "method_faithful":
        raise ValueError("Endpoint-history cells require method_faithful updates.")


def _validate_ablation_contract(config: dict[str, Any]) -> None:
    """Lock diagnostic labels to the exact one-factor settings they claim."""
    ablation = config.get("ablation")
    if not isinstance(ablation, dict):
        return
    if config.get("paper_comparable") is not False:
        raise ValueError("Ablations must explicitly set paper_comparable=false.")
    if config.get("ablation_schema_version") != 1:
        raise ValueError("Ablations require ablation_schema_version=1.")
    question = ablation.get("scientific_question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Ablations require a non-empty scientific_question.")
    matrix = ablation.get("matrix")
    if matrix == "endpoint_history":
        _validate_endpoint_history_contract(config)
        return
    if matrix is not None:
        raise ValueError(f"Unknown ablation matrix: {matrix!r}.")

    method = config["method"]
    name = ablation.get("name")
    allowed = {
        "deepall": {"control"},
        "dnt": {
            "baseline", "lambda_0", "interpolator_identity",
            "interpolator_residual", "endpoint_sqrt",
        },
        "dger": {"alternating", "two_step"},
        "dgnt": {
            "baseline", "lambda_0", "interpolator_identity",
            "interpolator_residual", "endpoint_sqrt", "two_step",
        },
    }
    if name not in allowed[method]:
        raise ValueError(
            f"{name!r} is not a recognized ablation variant for {method}."
        )

    changed_knobs = {
        "control": [],
        "baseline": [],
        "alternating": [],
        "lambda_0": ["loss.interpolation_lambda"],
        "interpolator_identity": ["loss.interpolation_mode"],
        "interpolator_residual": ["loss.interpolation_mode"],
        "endpoint_sqrt": ["loss.endpoint_normalization"],
        "two_step": ["update_schedule"],
    }[name]
    if ablation.get("changed_knobs") != changed_knobs:
        raise ValueError(
            f"{name} is a one-factor ablation and requires changed_knobs={changed_knobs}."
        )

    loss = config["loss"]
    expected_schedule = "two_step" if name == "two_step" else "method_faithful"
    if config.get("update_schedule") != expected_schedule:
        raise ValueError(
            f"{name} requires update_schedule={expected_schedule}."
        )
    if method in {"dnt", "dgnt"}:
        expected_interpolation = {
            "interpolation_lambda": 0.0 if name == "lambda_0" else 1.0,
            "interpolation_mode": {
                "interpolator_identity": "identity",
                "interpolator_residual": "residual",
            }.get(name, "learned"),
            "endpoint_loss": "mean_sample_l2",
            "endpoint_normalization": (
                "sqrt_latent" if name == "endpoint_sqrt" else "none"
            ),
        }
        for key, expected in expected_interpolation.items():
            defaults = {
                "interpolation_mode": "learned",
                "endpoint_loss": "mean_sample_l2",
                "endpoint_normalization": "none",
            }
            default = defaults.get(key)
            actual = loss.get(key, default)
            if actual != expected:
                raise ValueError(
                    f"{name} one-factor contract requires {key}={expected!r}; "
                    f"got {actual!r}."
                )
        if tuple(float(value) for value in loss.get("interpolation_weights", ())) != (
            0.0, 0.25, 0.5, 0.75, 1.0,
        ):
            raise ValueError("Ablations require the fixed five-point interpolation grid.")
    if method in {"dger", "dgnt"}:
        alphas = tuple(float(loss.get(key, -1)) for key in (
            "dger_alpha_1", "dger_alpha_2", "dger_alpha_3",
        ))
        if alphas != (0.5, 0.005, 0.01):
            raise ValueError("Ablations require fixed DGER weights (0.5, 0.005, 0.01).")


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
        is_ablation = isinstance(config.get("ablation"), dict)
        if int(config.get("batch_size", 0)) != 64:
            raise ValueError("The target paper reports minibatch size 64.")
        if int(config.get("pair_batch_size", 0)) != 64:
            raise ValueError("DNT/DGNT require 64 paired items per paper minibatch.")
        if int(config.get("dger_domain_batch_size", 0)) < 1:
            raise ValueError("A positive DGER per-domain batch size is required.")
        if int(config.get("epochs", 0)) != 100:
            raise ValueError("The target_comparison track requires exactly 100 epochs.")
        update_schedule = config.get("update_schedule")
        if update_schedule != "method_faithful":
            allowed_two_step = (
                is_ablation and update_schedule == "two_step"
                and config["method"] in {"dger", "dgnt"}
            )
            if not allowed_two_step:
                raise ValueError(
                    "Nonstandard update schedules require an explicit ablation "
                    "for a DGER-family method."
                )
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
            interpolation_lambda = float(loss.get("interpolation_lambda", -1))
            if not math.isfinite(interpolation_lambda) or interpolation_lambda < 0:
                raise ValueError("Interpolation lambda must be finite and nonnegative.")
            if not is_ablation and interpolation_lambda != 1.0:
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
            interpolation_mode = loss.get("interpolation_mode", "learned")
            if interpolation_mode not in {
                "learned", "conv1d_3layer", "mlp_3x64", "identity", "residual",
            }:
                raise ValueError("Unsupported interpolation mode.")
            endpoint_mode = loss.get("endpoint_loss", "mean_sample_l2")
            if endpoint_mode not in {"mean_sample_l2", "mse_mean_all"}:
                raise ValueError("Unsupported endpoint loss mode.")
            endpoint_normalization = loss.get("endpoint_normalization", "none")
            if endpoint_normalization not in {"none", "sqrt_latent"}:
                raise ValueError("Endpoint normalization must be none or sqrt_latent.")
            if not is_ablation and (
                interpolation_mode not in {"learned", "conv1d_3layer"}
                or endpoint_mode != "mean_sample_l2"
                or endpoint_normalization != "none"
            ):
                raise ValueError("Nonstandard interpolation settings require an explicit ablation.")
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
        _validate_ablation_contract(config)
