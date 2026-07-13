"""Aggregate completed runs without touching checkpoint-selection decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dg.training.metrics import aggregate

PAPER_REFERENCE_ACCURACY = {
    "deepall": {1.0: 92.69, 0.2: 80.80, 0.1: 73.95, 0.05: 67.99},
    "dnt": {1.0: 97.36, 0.2: 84.48, 0.1: 78.89, 0.05: 73.51},
    "dger": {1.0: 95.61, 0.2: 79.89, 0.1: 73.69, 0.05: 68.78},
    "dgnt": {1.0: 95.92, 0.2: 83.85, 0.1: 77.27, 0.05: 72.38},
}


def aggregate_results(results_root: Path) -> dict:
    """Aggregate every target metric per angle and across angles/seeds."""
    per_angle_values: dict[str, dict[str, list[float]]] = {}
    per_seed_values: dict[tuple[str, float, int], dict[str, list[float]]] = {}
    for path in results_root.rglob("final_metrics.json"):
        payload = json.loads(path.read_text())
        configuration = yaml.safe_load((path.parent / "resolved_config.yaml").read_text())
        method, budget, target, seed = configuration["method"], float(configuration["data_budget"]), configuration["target_angle"], int(configuration["seed"])
        angle_key = f"{method}/budget_{budget}/target_{target}"
        metrics = payload["target"]
        for metric, value in metrics.items():
            per_angle_values.setdefault(angle_key, {}).setdefault(metric, []).append(value)
            per_seed_values.setdefault((method, budget, seed), {}).setdefault(metric, []).append(value)
    per_angle = {key: {metric: aggregate(values) for metric, values in metrics.items()} for key, metrics in sorted(per_angle_values.items())}
    overall_values: dict[str, dict[str, list[float]]] = {}
    for (method, budget, _seed), metrics in per_seed_values.items():
        overall_key = f"{method}/budget_{budget}"
        for metric, values in metrics.items():
            overall_values.setdefault(overall_key, {}).setdefault(metric, []).append(sum(values) / len(values))
    overall = {key: {metric: aggregate(values) for metric, values in metrics.items()} for key, metrics in sorted(overall_values.items())}
    references = {key: PAPER_REFERENCE_ACCURACY[method][budget] for key in overall for method, budget in [(key.split("/")[0], float(key.split("budget_")[1]))]}
    return {"per_angle": per_angle, "overall_across_angles": overall, "paper_reference_accuracy_percent": references}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    arguments = parser.parse_args()
    report = aggregate_results(arguments.results_root)
    output = arguments.results_root / "aggregate.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
