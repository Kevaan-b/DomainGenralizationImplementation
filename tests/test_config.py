import pytest

from dg.config import validate_experiment_config


def _base_config() -> dict:
    return {"track": "target_comparison", "method": "deepall", "seed": 0, "target_angle": 0,
            "data_budget": 1.0, "data_root": "data", "results_root": "results", "dataset_seed": 1,
            "angles": [0, 15, 30, 45, 60, 75], "batch_size": 64,
            "pair_batch_size": 64, "dger_domain_batch_size": 12, "device": "auto",
            "deterministic": True, "epochs": 100, "update_schedule": "method_faithful",
            "optimizer": {"name": "sgd", "lr": .001, "momentum": .9, "weight_decay": .001},
            "loss": {"interpolation_lambda": 1.0,
                     "interpolation_weights": [0.0, .25, .5, .75, 1.0]}}


def test_config_rejects_noncanonical_benchmark_settings():
    config = _base_config()
    validate_experiment_config(config)
    config["batch_size"] = 60
    with pytest.raises(ValueError, match="batch size 64"):
        validate_experiment_config(config)


def test_dnt_config_rejects_undisclosed_interpolation_policy():
    config = _base_config()
    config["method"] = "dnt"
    config["loss"]["interpolation_policy"] = "iid_uniform"

    with pytest.raises(ValueError, match="uniform_grid"):
        validate_experiment_config(config)


def test_dgnt_config_requires_dger_weights_before_method_creation():
    config = _base_config()
    config["method"] = "dgnt"
    config["loss"]["interpolation_policy"] = "uniform_grid"

    with pytest.raises(ValueError, match="all three DGER alpha"):
        validate_experiment_config(config)


def test_target_dger_method_uses_domain_mean_reduction():
    from run_experiment import _create_method

    config = _base_config()
    config["method"] = "dger"
    config["loss"].update({
        "dger_alpha_1": .5, "dger_alpha_2": .005, "dger_alpha_3": .01,
    })

    validate_experiment_config(config)
    method = _create_method(config, source_domains=5)

    assert method.domain_reduction == "mean"


def test_original_dger_config_locks_the_paper_protocol():
    config = _base_config()
    config.update({
        "track": "dger_original", "method": "dger", "iterations": 3000,
        "batch_per_domain": 12,
        "data_budget": 1.0, "source_validation_fraction": 0.0,
        "update_schedule": "algorithm_1", "checkpoint_selection": "final_iteration",
        "checkpoint_interval": 100,
        "optimizer": {"lr": 1e-4, "auxiliary_lr": 1e-5},
        "loss": {"dger_alpha_1": 0.5, "dger_alpha_2": 0.005, "dger_alpha_3": 0.01},
    })
    config.pop("batch_size")
    config.pop("pair_batch_size")
    config.pop("dger_domain_batch_size")
    config.pop("epochs")

    validate_experiment_config(config)
    config["source_validation_fraction"] = 0.1
    with pytest.raises(ValueError, match="without a validation holdout"):
        validate_experiment_config(config)
