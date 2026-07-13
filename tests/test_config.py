import pytest

from dg.config import validate_experiment_config


def _base_config() -> dict:
    return {"track": "target_comparison", "method": "deepall", "seed": 0, "target_angle": 0,
            "data_budget": 1.0, "data_root": "data", "results_root": "results", "dataset_seed": 1,
            "angles": [0, 15, 30, 45, 60, 75], "batch_per_domain": 12, "device": "auto",
            "deterministic": True, "epochs": 100, "optimizer": {}, "loss": {}}


def test_config_rejects_noncanonical_benchmark_settings():
    config = _base_config()
    validate_experiment_config(config)
    config["batch_per_domain"] = 8
    with pytest.raises(ValueError, match="effective batch"):
        validate_experiment_config(config)
