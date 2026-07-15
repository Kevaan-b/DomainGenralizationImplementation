from copy import deepcopy
from argparse import Namespace
from pathlib import Path

import yaml


def _base_config() -> dict:
    return {
        "track": "target_comparison",
        "method": "deepall",
        "seed": 0,
        "target_angle": 0,
        "data_budget": 1.0,
        "results_root": "results",
        "update_schedule": "method_faithful",
        "loss": {
            "interpolation_lambda": 1.0,
            "interpolation_mode": "learned",
            "endpoint_normalization": "none",
        },
    }


def test_dnt_ablation_matrix_is_one_factor_at_a_time_and_immutable():
    from ablation_sweep import build_ablation_configs

    base = _base_config()
    before = deepcopy(base)

    configurations = build_ablation_configs(base, method="dnt")

    assert base == before
    assert [config["ablation"]["name"] for config in configurations] == [
        "baseline",
        "lambda_0",
        "interpolator_identity",
        "interpolator_residual",
        "endpoint_sqrt",
    ]
    by_name = {config["ablation"]["name"]: config for config in configurations}
    assert by_name["baseline"]["loss"] == before["loss"]
    assert by_name["lambda_0"]["loss"]["interpolation_lambda"] == 0.0
    assert by_name["interpolator_identity"]["loss"]["interpolation_mode"] == "identity"
    assert by_name["interpolator_residual"]["loss"]["interpolation_mode"] == "residual"
    assert by_name["endpoint_sqrt"]["loss"]["endpoint_normalization"] == "sqrt_latent"
    for name, configuration in by_name.items():
        changed = {
            key for key in (
                "interpolation_lambda", "interpolation_mode", "endpoint_normalization",
            )
            if configuration["loss"][key] != before["loss"][key]
        }
        assert len(changed) == (0 if name == "baseline" else 1)


def test_dger_ablation_matrix_compares_alternating_and_two_step_schedules():
    from ablation_sweep import build_ablation_configs

    configurations = build_ablation_configs(_base_config(), method="dger")

    assert {
        config["ablation"]["name"]: config["update_schedule"]
        for config in configurations
    } == {
        "alternating": "method_faithful",
        "two_step": "two_step",
    }


def test_dgnt_ablation_matrix_includes_interpolation_and_schedule_controls():
    from ablation_sweep import build_ablation_configs

    configurations = build_ablation_configs(_base_config(), method="dgnt")
    names = {config["ablation"]["name"] for config in configurations}

    assert names == {
        "baseline",
        "lambda_0",
        "interpolator_identity",
        "interpolator_residual",
        "endpoint_sqrt",
        "two_step",
    }


def test_endpoint_history_matrix_is_exact_cartesian_product_for_dnt_and_dgnt():
    from ablation_sweep import build_ablation_configs

    expected = [
        ("hist_mlp_mse", "mlp_3x64", "mse_mean_all"),
        ("hist_mlp_l2", "mlp_3x64", "mean_sample_l2"),
        ("hist_conv_mse", "conv1d_3layer", "mse_mean_all"),
        ("hist_conv_l2", "conv1d_3layer", "mean_sample_l2"),
    ]

    for method in ("dnt", "dgnt"):
        configurations = build_ablation_configs(
            _base_config(), method=method, matrix="endpoint_history",
        )

        assert [
            (
                config["ablation"]["name"],
                config["loss"]["interpolation_mode"],
                config["loss"]["endpoint_loss"],
            )
            for config in configurations
        ] == expected
        assert all(
            config["ablation"].get("matrix") == "endpoint_history"
            for config in configurations
        )


def test_endpoint_history_matrix_gives_dger_one_unchanged_alternating_reference():
    from ablation_sweep import build_ablation_configs

    configurations = build_ablation_configs(
        _base_config(), method="dger", matrix="endpoint_history",
    )

    assert len(configurations) == 1
    reference = configurations[0]
    assert reference["ablation"]["name"] == "dger_shared_control"
    assert reference["ablation"].get("matrix") == "endpoint_history"
    assert reference["ablation"]["changed_knobs"] == []
    assert reference["update_schedule"] == "method_faithful"


def test_ablation_run_directory_prevents_variants_from_overwriting_each_other():
    from ablation_sweep import ablation_run_dir

    root = Path("/tmp/results")
    common = {
        "results_root": str(root), "track": "target_comparison",
        "method": "dgnt", "target_angle": 0, "seed": 2, "data_budget": 0.1,
    }
    identity = {**common, "ablation": {"name": "interpolator_identity"}}
    residual = {**common, "ablation": {"name": "interpolator_residual"}}

    identity_path = ablation_run_dir(identity)
    residual_path = ablation_run_dir(residual)

    assert identity_path != residual_path
    assert identity_path.parent.name == "dgnt"
    assert identity_path.name == "target_0_seed_2_budget_0.1"
    assert identity_path.parents[2].name == "interpolator_identity"
    assert identity_path.parents[1].name != residual_path.parents[1].name


def test_config_fingerprint_is_order_independent_and_scientifically_sensitive():
    from ablation_sweep import config_fingerprint

    first = _base_config()
    first["method"] = "dnt"
    first["ablation"] = {"name": "baseline", "changed_knobs": []}
    reordered = {
        key: deepcopy(first[key]) for key in reversed(tuple(first))
    }
    reordered["loss"] = {
        key: reordered["loss"][key] for key in reversed(tuple(reordered["loss"]))
    }
    changed = deepcopy(first)
    changed["loss"]["interpolation_lambda"] = 0.0

    assert config_fingerprint(first) == config_fingerprint(reordered)
    assert config_fingerprint(first) != config_fingerprint(changed)
    assert len(config_fingerprint(first)) >= 12


def test_run_directory_contains_the_exact_config_fingerprint():
    from ablation_sweep import ablation_run_dir, config_fingerprint

    configuration = _base_config()
    configuration.update({
        "method": "dnt", "target_angle": 30, "seed": 2,
        "ablation": {"name": "baseline", "changed_knobs": []},
    })

    path = ablation_run_dir(configuration)

    assert config_fingerprint(configuration) in path.parts


def test_run_boundary_injects_identity_and_rejects_stale_generated_configs(
    tmp_path,
):
    from run_experiment import _resolve_ablation_identity

    configuration = _base_config()
    configuration["ablation"] = {
        "name": "baseline", "scientific_question": "test",
        "changed_knobs": [],
    }

    resolved = _resolve_ablation_identity(configuration, tmp_path)

    assert "code_fingerprint" not in configuration["ablation"]
    assert resolved["ablation"]["code_fingerprint"]
    assert resolved["ablation"]["config_fingerprint"]
    stale = deepcopy(resolved)
    stale["ablation"]["code_fingerprint"] = "stale"
    try:
        _resolve_ablation_identity(stale, tmp_path)
    except ValueError as error:
        assert "stale" in str(error).lower()
    else:
        raise AssertionError("stale generated configs must be rejected")


def test_dry_run_prints_commands_without_creating_files_or_directories(
    tmp_path, monkeypatch, capsys,
):
    import ablation_sweep

    base = _base_config()
    base["results_root"] = str(tmp_path / "results")
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(base))
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    monkeypatch.setattr(ablation_sweep, "parse_args", lambda: Namespace(
        config=config_path,
        matrix=None,
        methods=["dnt"],
        target_angles=[30],
        seeds=[0],
        data_budget=1.0,
        variants=["baseline"],
        skip_existing=False,
        dry_run=True,
    ))
    monkeypatch.setattr(
        ablation_sweep.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not launch training")
        ),
    )

    ablation_sweep.main()

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after == before
    assert "run_experiment.py" in capsys.readouterr().out


def test_endpoint_history_selector_reaches_dry_run_commands(
    tmp_path, monkeypatch, capsys,
):
    import ablation_sweep

    base = _base_config()
    base["results_root"] = str(tmp_path / "results")
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(base))
    monkeypatch.setattr(ablation_sweep, "parse_args", lambda: Namespace(
        config=config_path,
        matrix="endpoint_history",
        methods=["dnt", "dger"],
        target_angles=[75],
        seeds=[0],
        data_budget=0.1,
        variants=None,
        skip_existing=False,
        dry_run=True,
    ))

    ablation_sweep.main()

    commands = capsys.readouterr().out.splitlines()
    assert len(commands) == 5
    assert all("run_experiment.py" in command for command in commands)
    assert any("hist_mlp_mse" in command for command in commands)
    assert any("hist_mlp_l2" in command for command in commands)
    assert any("hist_conv_mse" in command for command in commands)
    assert any("hist_conv_l2" in command for command in commands)
    assert any("dger_shared_control" in command for command in commands)


def test_summary_reports_seed_statistics_separately_for_each_target():
    from ablation_sweep import _summary

    report = _summary([
        ("dnt", "baseline", 30, 0, .10, .20),
        ("dnt", "baseline", 30, 1, .30, .40),
        ("dnt", "baseline", 75, 0, .80, .70),
        ("dnt", "baseline", 75, 1, .80, .90),
    ])

    assert "method | variant | target" in report
    assert "dnt | baseline | 30 | 20.00 ± 10.00 | 30.00 ± 10.00" in report
    assert "dnt | baseline | 75 | 80.00 ± 0.00 | 80.00 ± 10.00" in report
