import json

import pytest
import yaml

from evaluate import aggregate_results


def test_aggregation_reports_per_angle_and_seed_level_overall_metrics(tmp_path):
    for seed, values in ((0, (0.8, 0.7)), (1, (0.6, 0.5))):
        for angle, accuracy in zip((0, 15), values):
            run = tmp_path / f"run_{seed}_{angle}"
            run.mkdir()
            (run / "resolved_config.yaml").write_text(yaml.safe_dump({"method": "deepall", "data_budget": 1.0, "target_angle": angle, "seed": seed}))
            (run / "final_metrics.json").write_text(json.dumps({"target": {"accuracy": accuracy, "cross_entropy": 1 - accuracy, "mean_per_class_accuracy": accuracy}}))
    report = aggregate_results(tmp_path)
    assert report["per_angle"]["deepall/budget_1.0/target_0"]["accuracy"]["mean"] == 0.7
    assert report["overall_across_angles"]["deepall/budget_1.0"]["accuracy"]["mean"] == 0.65
    assert report["paper_reference_accuracy_percent"]["deepall/budget_1.0"] == 92.69
    assert report["paper_matrix_complete"]["deepall/budget_1.0"] is False

    with pytest.raises(ValueError, match="six target angles and exactly five seeds"):
        aggregate_results(tmp_path, require_paper_matrix=True)


def test_paper_aggregation_excludes_diagnostic_ablations(tmp_path):
    run = tmp_path / "ablation"
    run.mkdir()
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({
        "method": "dnt", "data_budget": 1.0, "target_angle": 75, "seed": 0,
        "paper_comparable": False, "ablation": {"name": "lambda_0"},
    }))
    (run / "final_metrics.json").write_text(json.dumps({
        "target": {"accuracy": .99},
    }))

    report = aggregate_results(tmp_path)

    assert report["overall_across_angles"] == {}
    assert report["paper_reference_accuracy_percent"] == {}
