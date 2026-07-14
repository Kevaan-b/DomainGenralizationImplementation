"""Run and compare all RotatedMNIST methods for one scarcity scenario."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

METHODS = ("deepall", "dnt", "dger", "dgnt")
ANGLES = (0, 15, 30, 45, 60, 75)
BUDGETS = (1.0, 0.2, 0.1, 0.05)


@dataclass(frozen=True)
class RunRecord:
    method: str
    seed: int
    source_validation_accuracy: float
    target_accuracy: float
    run_dir: Path


def run_dir(config: dict, repo_root: Path, method: str, target_angle: int,
            seed: int, budget: float) -> Path:
    root = Path(config["results_root"])
    if not root.is_absolute():
        root = repo_root / root
    return root / config["track"] / method / (
        f"target_{target_angle}_seed_{seed}_budget_{budget}"
    )


def load_record(path: Path, method: str, seed: int) -> RunRecord:
    metrics_path = path / "final_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing completed result: {metrics_path}")
    payload = json.loads(metrics_path.read_text())
    return RunRecord(
        method=method,
        seed=seed,
        source_validation_accuracy=float(payload["best_source_validation_accuracy"]),
        target_accuracy=float(payload["target"]["accuracy"]),
        run_dir=path,
    )


def run_experiment(repo_root: Path, config_path: Path, method: str,
                   target_angle: int, seed: int, budget: float) -> None:
    command = [
        sys.executable, str(repo_root / "src" / "run_experiment.py"),
        "--config", str(config_path), "--method", method,
        "--target-angle", str(target_angle), "--seed", str(seed),
        "--data-budget", str(budget),
    ]
    environment = os.environ.copy()
    source = str(repo_root / "src")
    environment["PYTHONPATH"] = (
        source if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(command, cwd=repo_root, env=environment, check=True)


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)
    output = [
        " | ".join(str(value).ljust(widths[i])
                   for i, value in enumerate(headers)),
        separator,
    ]
    output.extend(
        " | ".join(str(value).ljust(widths[i])
                   for i, value in enumerate(row))
        for row in rows
    )
    return "\n".join(output)


def metric_table(records: Sequence[RunRecord], methods: Sequence[str],
                 seeds: Sequence[int], metric: str) -> str:
    values = {
        (record.method, record.seed): getattr(record, metric)
        for record in records
    }
    headers = ["method"] + [f"seed {seed}" for seed in seeds] + ["mean ± std"]
    rows = []
    for method in methods:
        method_values = [values[(method, seed)] for seed in seeds]
        mean = statistics.fmean(method_values)
        deviation = statistics.pstdev(method_values)
        rows.append(
            [method]
            + [format_percent(value) for value in method_values]
            + [f"{format_percent(mean)} ± {format_percent(deviation)}"]
        )
    return format_table(headers, rows)


def default_log_path(config: dict, repo_root: Path, target_angle: int,
                     budget: float, seeds: Sequence[int]) -> Path:
    root = Path(config["results_root"])
    if not root.is_absolute():
        root = repo_root / root
    seed_key = "-".join(str(seed) for seed in seeds)
    return root / config["track"] / "sweeps" / (
        f"target_{target_angle}_budget_{budget}_seeds_{seed_key}.log"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-angle", type=int, required=True, choices=ANGLES)
    parser.add_argument("--data-budget", type=float, required=True, choices=BUDGETS)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS),
                        help="Methods to run; default is all four.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse runs containing final_metrics.json.")
    parser.add_argument("--log-file", type=Path,
                        help="Output log path; defaults under results/sweeps.")
    return parser.parse_args()

def main() -> None:
    arguments = parse_args()
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("--seeds must not contain duplicates.")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = arguments.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    records: list[RunRecord] = []

    for method in arguments.methods:
        for seed in arguments.seeds:
            path = run_dir(
                config, repo_root, method, arguments.target_angle, seed,
                arguments.data_budget,
            )
            if arguments.skip_existing and (path / "final_metrics.json").exists():
                print(f"Reusing {method}, seed {seed}: {path}")
            else:
                print(f"Running {method}, seed {seed}...")
                run_experiment(
                    repo_root, config_path, method, arguments.target_angle, seed,
                    arguments.data_budget,
                )
            records.append(load_record(path, method, seed))

    print(
        f"\nComparison: target angle {arguments.target_angle}°, "
        f"data budget {arguments.data_budget}, seeds {list(arguments.seeds)}"
    )
    print("\nTarget accuracy (%)")
    print(metric_table(
        records, arguments.methods, arguments.seeds, "target_accuracy",
    ))
    print("\nBest source-validation accuracy (%)")
    print(metric_table(
        records, arguments.methods, arguments.seeds,
        "source_validation_accuracy",
    ))



    log_path = arguments.log_file or default_log_path(
        config, repo_root, arguments.target_angle, arguments.data_budget,
        arguments.seeds,
    )
    if not log_path.is_absolute():
        log_path = repo_root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report = (
        f"Comparison: target angle {arguments.target_angle}°, " 
        f"data budget {arguments.data_budget}, seeds {list(arguments.seeds)}\n\n" 
        f"Target accuracy (%)\n{metric_table(records, arguments.methods, arguments.seeds, 'target_accuracy')}\n\n" 
        f"Best source-validation accuracy (%)\n{metric_table(records, arguments.methods, arguments.seeds, 'source_validation_accuracy')}\n"
    )
    log_path.write_text(report)
    print(f"\nSaved comparison table to {log_path}")
if __name__ == "__main__":
    main()
