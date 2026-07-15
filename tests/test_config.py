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


def _explicit_ablation_config(method: str, name: str) -> dict:
    config = _base_config()
    config["method"] = method
    changed_knobs = {
        "lambda_0": ["loss.interpolation_lambda"],
        "interpolator_identity": ["loss.interpolation_mode"],
        "interpolator_residual": ["loss.interpolation_mode"],
        "endpoint_sqrt": ["loss.endpoint_normalization"],
        "two_step": ["update_schedule"],
    }.get(name, [])
    config["ablation_schema_version"] = 1
    config["ablation"] = {
        "name": name,
        "scientific_question": "test contract",
        "changed_knobs": changed_knobs,
    }
    config["paper_comparable"] = False
    config["loss"].update({
        "interpolation_policy": "uniform_grid",
        "interpolation_mode": "learned",
        "endpoint_normalization": "none",
    })
    if method in {"dger", "dgnt"}:
        config["loss"].update({
            "dger_alpha_1": .5, "dger_alpha_2": .005, "dger_alpha_3": .01,
        })
    return config


def test_explicit_lambda_zero_ablation_is_valid_and_reaches_method_factory():
    from run_experiment import _create_method

    config = _explicit_ablation_config("dnt", "lambda_0")
    config["loss"]["interpolation_lambda"] = 0.0

    validate_experiment_config(config)
    method = _create_method(config, source_domains=5)

    assert method.loss_weight == 0.0


@pytest.mark.parametrize("mode", ["identity", "residual"])
def test_interpolator_mode_ablation_reaches_dnt_and_dgnt_factories(mode):
    from run_experiment import _create_method

    for name in ("dnt", "dgnt"):
        config = _explicit_ablation_config(name, f"interpolator_{mode}")
        config["loss"]["interpolation_mode"] = mode

        validate_experiment_config(config)
        method = _create_method(config, source_domains=5)

        assert method.interpolation_mode == mode


def test_endpoint_sqrt_normalization_ablation_reaches_method_factory():
    from run_experiment import _create_method

    config = _explicit_ablation_config("dnt", "endpoint_sqrt")
    config["loss"]["endpoint_normalization"] = "sqrt_latent"

    validate_experiment_config(config)
    method = _create_method(config, source_domains=5)

    assert method.endpoint_normalization == "sqrt_latent"


def test_two_step_is_allowed_only_as_an_explicit_dger_family_ablation():
    config = _explicit_ablation_config("dger", "two_step")
    config["update_schedule"] = "two_step"

    validate_experiment_config(config)

    config.pop("ablation")
    with pytest.raises(ValueError, match="explicit ablation"):
        validate_experiment_config(config)


def test_ablation_config_rejects_unknown_variant_names():
    config = _explicit_ablation_config("dnt", "invented_variant")

    with pytest.raises(ValueError, match="recognized ablation variant"):
        validate_experiment_config(config)


def test_ablation_name_must_match_its_exact_scientific_delta():
    config = _explicit_ablation_config("dnt", "lambda_0")
    # The label claims lambda=0, but the actual loss remains the baseline.
    config["loss"]["interpolation_lambda"] = 1.0

    with pytest.raises(ValueError, match="lambda_0.*interpolation_lambda"):
        validate_experiment_config(config)


def test_one_factor_ablation_rejects_accidental_compound_changes():
    config = _explicit_ablation_config("dnt", "interpolator_residual")
    config["loss"]["interpolation_mode"] = "residual"
    config["loss"]["endpoint_normalization"] = "sqrt_latent"

    with pytest.raises(ValueError, match="one-factor|changed_knobs"):
        validate_experiment_config(config)


@pytest.mark.parametrize("method", ["dnt", "dgnt"])
@pytest.mark.parametrize(
    ("name", "interpolation_mode", "endpoint_loss", "changed_knobs"),
    [
        (
            "hist_mlp_mse", "mlp_3x64", "mse_mean_all",
            ["loss.interpolation_mode", "loss.endpoint_loss"],
        ),
        (
            "hist_mlp_l2", "mlp_3x64", "mean_sample_l2",
            ["loss.interpolation_mode"],
        ),
        (
            "hist_conv_mse", "conv1d_3layer", "mse_mean_all",
            ["loss.endpoint_loss"],
        ),
        ("hist_conv_l2", "conv1d_3layer", "mean_sample_l2", []),
    ],
)
def test_endpoint_history_configs_validate_and_reach_method_factory(
    method, name, interpolation_mode, endpoint_loss, changed_knobs,
):
    from run_experiment import _create_method

    config = _explicit_ablation_config(method, name)
    config["ablation"].update({
        "matrix": "endpoint_history",
        "changed_knobs": changed_knobs,
        "factorial_member": True,
    })
    config["loss"].update({
        "interpolation_mode": interpolation_mode,
        "endpoint_loss": endpoint_loss,
    })

    validate_experiment_config(config)
    created = _create_method(config, source_domains=5)

    assert created.interpolation_mode == interpolation_mode
    assert created.endpoint_loss_mode == endpoint_loss


def test_endpoint_history_dger_reference_validates_as_unchanged_alternating_control():
    config = _explicit_ablation_config("dger", "dger_shared_control")
    config["ablation"].update({
        "matrix": "endpoint_history",
        "changed_knobs": [],
        "factorial_member": False,
    })

    validate_experiment_config(config)


@pytest.mark.parametrize("method", ["dnt", "dgnt"])
@pytest.mark.parametrize(
    ("name", "interpolation_mode", "endpoint_weight", "changed_knobs"),
    [
        ("scale_mlp_1", "mlp_3x64", 1.0, ["loss.interpolation_mode"]),
        (
            "scale_mlp_1_over_8", "mlp_3x64", 0.125,
            ["loss.interpolation_mode", "loss.endpoint_weight"],
        ),
        (
            "scale_mlp_1_over_64", "mlp_3x64", 0.015625,
            ["loss.interpolation_mode", "loss.endpoint_weight"],
        ),
        (
            "scale_mlp_0_01", "mlp_3x64", 0.01,
            ["loss.interpolation_mode", "loss.endpoint_weight"],
        ),
        ("scale_conv_1", "conv1d_3layer", 1.0, []),
        (
            "scale_conv_1_over_8", "conv1d_3layer", 0.125,
            ["loss.endpoint_weight"],
        ),
        (
            "scale_conv_1_over_64", "conv1d_3layer", 0.015625,
            ["loss.endpoint_weight"],
        ),
        (
            "scale_conv_0_01", "conv1d_3layer", 0.01,
            ["loss.endpoint_weight"],
        ),
    ],
)
def test_endpoint_scale_configs_validate_and_propagate_to_factory(
    method, name, interpolation_mode, endpoint_weight, changed_knobs,
):
    from run_experiment import _create_method

    config = _explicit_ablation_config(method, name)
    config["ablation"].update({
        "matrix": "endpoint_scale",
        "changed_knobs": changed_knobs,
        "factorial_member": True,
    })
    config["loss"].update({
        "interpolation_mode": interpolation_mode,
        "endpoint_loss": "mean_sample_l2",
        "endpoint_weight": endpoint_weight,
    })

    validate_experiment_config(config)
    created = _create_method(config, source_domains=5)

    assert created.interpolation_mode == interpolation_mode
    assert created.endpoint_loss_mode == "mean_sample_l2"
    assert created.endpoint_weight == endpoint_weight


@pytest.mark.parametrize(
    ("method", "name"),
    [
        ("deepall", "deepall_shared_control"),
        ("dger", "dger_shared_control"),
    ],
)
def test_endpoint_scale_shared_controls_validate_without_changed_knobs(method, name):
    config = _explicit_ablation_config(method, name)
    config["ablation"].update({
        "matrix": "endpoint_scale",
        "changed_knobs": [],
        "factorial_member": False,
    })

    validate_experiment_config(config)
