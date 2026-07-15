"""Run isolated DNT/DGER/DGNT diagnostic ablations on RotatedMNIST."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


METHODS = ("deepall", "dnt", "dger", "dgnt")
ANGLES = (0, 15, 30, 45, 60, 75)
BUDGETS = (1.0, .2, .1, .05)


def config_fingerprint(configuration: dict[str, Any]) -> str:
    """Hash the complete resolved run contract, excluding only its own hash."""
    canonical = deepcopy(configuration)
    canonical.pop("results_root", None)
    ablation = canonical.get("ablation")
    if isinstance(ablation, dict):
        ablation.pop("config_fingerprint", None)
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _code_fingerprint(repo_root: Path) -> str:
    """Hash source, dependency declarations, and the active runtime."""
    digest = hashlib.sha256()
    for path in sorted((repo_root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(repo_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in ("pyproject.toml", "requirements.txt"):
        path = repo_root / name
        if path.exists():
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    digest.update(sys.version.encode())
    for distribution in ("numpy", "PyYAML", "torch", "torchvision"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        digest.update(f"{distribution}={version}\0".encode())
    return digest.hexdigest()[:16]


def _variant(
    base: dict[str, Any], method: str, name: str, question: str,
    matrix: str | None = None,
) -> dict[str, Any]:
    configuration = deepcopy(base)
    configuration["method"] = method
    configuration["paper_comparable"] = False
    configuration["ablation_schema_version"] = 1
    configuration["ablation"] = {
        "name": name,
        "scientific_question": question,
        "changed_knobs": [],
    }
    if matrix is not None:
        configuration["ablation"]["matrix"] = matrix
    return configuration


def _endpoint_history_configs(
    base: dict[str, Any], method: str,
) -> list[dict[str, Any]]:
    matrix = "endpoint_history"
    if method in {"deepall", "dger"}:
        name = "deepall_shared_control" if method == "deepall" else "dger_shared_control"
        control = _variant(
            base, method, name,
            f"Shared {method.upper()} reference without an endpoint objective.",
            matrix,
        )
        control["ablation"].update({
            "factorial_member": False,
            "control_for": "dnt" if method == "deepall" else "dgnt",
        })
        return [control]

    cells = (
        ("hist_mlp_mse", "mlp_3x64", "mse_mean_all", "5bb27a5"),
        ("hist_mlp_l2", "mlp_3x64", "mean_sample_l2", None),
        ("hist_conv_mse", "conv1d_3layer", "mse_mean_all", None),
        ("hist_conv_l2", "conv1d_3layer", "mean_sample_l2", "7264861"),
    )
    configurations = []
    for name, architecture, endpoint_mode, historical_match in cells:
        config = _variant(
            base, method, name,
            "Which historical interpolator architecture and endpoint reduction "
            "caused the DNT/DGNT performance change?",
            matrix,
        )
        config["loss"].update({
            "interpolation_mode": architecture,
            "endpoint_loss": endpoint_mode,
            "endpoint_normalization": "none",
        })
        changed = []
        if architecture == "mlp_3x64":
            changed.append("loss.interpolation_mode")
        if endpoint_mode == "mse_mean_all":
            changed.append("loss.endpoint_loss")
        config["ablation"].update({
            "changed_knobs": changed,
            "factorial_member": True,
            "historical_match": historical_match,
        })
        configurations.append(config)
    return configurations


def build_ablation_configs(
    base: dict[str, Any], method: str, matrix: str | None = None,
) -> list[dict[str, Any]]:
    """Return immutable, mostly one-factor diagnostic variants for one method."""
    if method not in METHODS:
        raise ValueError(f"Ablations are available for {METHODS}, got {method!r}.")
    base = deepcopy(base)
    base.setdefault("loss", {})
    base["loss"].setdefault("interpolation_mode", "learned")
    base["loss"].setdefault("endpoint_normalization", "none")
    if matrix not in {None, "endpoint_history"}:
        raise ValueError("matrix must be endpoint_history when provided.")
    if matrix == "endpoint_history":
        return _endpoint_history_configs(base, method)
    if method == "deepall":
        return [_variant(
            base, method, "control",
            "What classification-only accuracy is obtained on the identical split?",
        )]
    if method == "dger":
        alternating = _variant(
            base, method, "alternating",
            "Does the literal alternating DGER schedule remain stable?",
        )
        alternating["update_schedule"] = "method_faithful"
        two_step = _variant(
            base, method, "two_step",
            "Does reducing DGER to two optimizer steps remove representation drift?",
        )
        two_step["update_schedule"] = "two_step"
        two_step["ablation"]["changed_knobs"] = ["update_schedule"]
        return [alternating, two_step]

    questions = {
        "baseline": "How stable is the current interpolation reconstruction?",
        "lambda_0": "Does the paired training path recover classification-only performance?",
        "interpolator_identity": "Does fixed linear interpolation remove learned-interpolator instability?",
        "interpolator_residual": "Does identity initialization stabilize a trainable interpolator?",
        "endpoint_sqrt": "Is the unnormalized 64-dimensional endpoint norm too strong?",
    }
    configurations = [
        _variant(base, method, name, question) for name, question in questions.items()
    ]
    by_name = {config["ablation"]["name"]: config for config in configurations}
    by_name["lambda_0"]["loss"]["interpolation_lambda"] = 0.0
    by_name["lambda_0"]["ablation"]["changed_knobs"] = ["loss.interpolation_lambda"]
    by_name["interpolator_identity"]["loss"]["interpolation_mode"] = "identity"
    by_name["interpolator_identity"]["ablation"]["changed_knobs"] = ["loss.interpolation_mode"]
    by_name["interpolator_residual"]["loss"]["interpolation_mode"] = "residual"
    by_name["interpolator_residual"]["ablation"]["changed_knobs"] = ["loss.interpolation_mode"]
    by_name["endpoint_sqrt"]["loss"]["endpoint_normalization"] = "sqrt_latent"
    by_name["endpoint_sqrt"]["ablation"]["changed_knobs"] = ["loss.endpoint_normalization"]
    if method == "dgnt":
        two_step = _variant(
            base, method, "two_step",
            "Does reducing DGNT's DGER component to two steps improve stability?",
        )
        two_step["update_schedule"] = "two_step"
        two_step["ablation"]["changed_knobs"] = ["update_schedule"]
        configurations.append(two_step)
    return configurations


def ablation_run_dir(configuration: dict[str, Any]) -> Path:
    """Return a variant-isolated run directory."""
    ablation = configuration.get("ablation")
    if not isinstance(ablation, dict) or not ablation.get("name"):
        raise ValueError("Ablation runs require ablation.name.")
    return (
        Path(configuration["results_root"])
        / configuration["track"] / "ablations" / str(ablation["name"])
        / config_fingerprint(configuration)
        / configuration["method"]
        / f"target_{configuration['target_angle']}_seed_{configuration['seed']}_budget_{configuration['data_budget']}"
    )


def _run_command(repo_root: Path, config_path: Path) -> list[str]:
    return [
        sys.executable, str(repo_root / "src" / "run_experiment.py"),
        "--config", str(config_path),
    ]


def _metric(run_dir: Path) -> tuple[float, float]:
    path = run_dir / "final_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed ablation: {path}")
    payload = json.loads(path.read_text())
    return (
        float(payload["best_source_validation_accuracy"]),
        float(payload["target"]["accuracy"]),
    )


def _completed_run_matches(run_dir: Path, configuration: dict[str, Any]) -> bool:
    """Reuse only a completed run carrying the identical resolved contract."""
    metrics_path = run_dir / "final_metrics.json"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return False
    try:
        recorded = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(recorded, dict):
        return False
    return config_fingerprint(recorded) == config_fingerprint(configuration)


def _summary(rows: Iterable[tuple[str, str, int, int, float, float]]) -> str:
    grouped: dict[tuple[str, str, int], tuple[list[float], list[float]]] = {}
    for method, variant, target_angle, _seed, source, target in rows:
        source_values, target_values = grouped.setdefault(
            (method, variant, target_angle), ([], []),
        )
        source_values.append(source)
        target_values.append(target)
    lines = [
        "method | variant | target | source validation mean ± std | target mean ± std",
    ]
    lines.append("-" * len(lines[0]))
    for (method, variant, target_angle), (source, target) in sorted(grouped.items()):
        lines.append(
            f"{method} | {variant} | {target_angle} | "
            f"{100 * statistics.fmean(source):.2f} ± {100 * statistics.pstdev(source):.2f} | "
            f"{100 * statistics.fmean(target):.2f} ± {100 * statistics.pstdev(target):.2f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--matrix", choices=("endpoint_history",),
        help="Run a named focused matrix instead of the default diagnostics.",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS,
        default=["deepall", "dnt", "dger"],
        help="Default screens controls and components; add dgnt after they are stable.",
    )
    parser.add_argument("--target-angles", nargs="+", type=int, choices=ANGLES, default=[30, 75])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-budget", type=float, choices=BUDGETS, default=1.0)
    parser.add_argument("--variants", nargs="+", help="Run only matching ablation names.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("--seeds must be unique.")
    if any(seed < 0 for seed in arguments.seeds):
        raise ValueError("--seeds must be nonnegative.")
    if len(set(arguments.target_angles)) != len(arguments.target_angles):
        raise ValueError("--target-angles must be unique.")
    if len(set(arguments.methods)) != len(arguments.methods):
        raise ValueError("--methods must be unique.")
    repo_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load(arguments.config.expanduser().resolve().read_text())
    if not isinstance(base, dict):
        raise ValueError("--config must contain a YAML mapping.")
    results_root = Path(base["results_root"]).expanduser()
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    base["results_root"] = str(results_root.resolve())
    rows: list[tuple[str, str, int, int, float, float]] = []
    generated_root = (
        results_root / base["track"] / "ablations" / "_configs"
    ).resolve()
    code_fingerprint = _code_fingerprint(repo_root)
    planned_runs = 0
    environment = os.environ.copy()
    source = str(repo_root / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )

    variants_by_method = {
        method: build_ablation_configs(base, method, matrix=arguments.matrix)
        for method in arguments.methods
    }
    if arguments.variants:
        available = {
            config["ablation"]["name"]
            for variants in variants_by_method.values() for config in variants
        }
        unavailable = sorted(set(arguments.variants) - available)
        if unavailable:
            raise ValueError(
                "Selected methods do not provide variants: "
                + ", ".join(unavailable)
            )

    for method, method_variants in variants_by_method.items():
        variants = method_variants
        if arguments.variants:
            variants = [
                config for config in variants
                if config["ablation"]["name"] in arguments.variants
            ]
        for template in variants:
            for target in arguments.target_angles:
                for seed in arguments.seeds:
                    planned_runs += 1
                    config = deepcopy(template)
                    config.update({
                        "target_angle": target,
                        "seed": seed,
                        "data_budget": arguments.data_budget,
                    })
                    name = str(config["ablation"]["name"])
                    config["ablation"]["code_fingerprint"] = code_fingerprint
                    config["ablation"]["config_fingerprint"] = config_fingerprint(config)
                    config_path = generated_root / (
                        f"{method}_{name}_{config['ablation']['config_fingerprint']}.yaml"
                    )
                    command = _run_command(repo_root, config_path)
                    run_dir = ablation_run_dir(config)
                    if arguments.dry_run:
                        print(shlex.join(command))
                        continue
                    generated_root.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
                    if not (
                        arguments.skip_existing
                        and _completed_run_matches(run_dir, config)
                    ):
                        subprocess.run(command, cwd=repo_root, env=environment, check=True)
                    source_accuracy, target_accuracy = _metric(run_dir)
                    rows.append((
                        method, name, target, seed,
                        source_accuracy, target_accuracy,
                    ))

    if planned_runs == 0:
        raise ValueError("No ablations matched the selected methods and variants.")
    if arguments.dry_run:
        return
    report = _summary(rows)
    print(report)
    target_key = "-".join(str(value) for value in arguments.target_angles)
    seed_key = "-".join(str(value) for value in arguments.seeds)
    method_key = "-".join(arguments.methods)
    variant_key = "-".join(arguments.variants or ["all"])
    matrix_key = arguments.matrix or "diagnostic"
    summary_path = (
        results_root / base["track"] / "ablations"
        / (
            f"summary_matrix_{matrix_key}_methods_{method_key}_variants_{variant_key}_targets_{target_key}_"
            f"seeds_{seed_key}_budget_{arguments.data_budget}.txt"
        )
    )
    summary_path.write_text(report + "\n")
    print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
