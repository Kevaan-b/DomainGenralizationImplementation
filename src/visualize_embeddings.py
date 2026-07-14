"""Visualize trained RotatedMNIST encoder embeddings with PCA and UMAP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from dg.data.rotated_mnist import build_or_load_cache
from dg.training.engine import make_fold
from dg.training.reproducibility import resolve_device
from run_experiment import _create_method


METHODS = ("deepall", "dnt", "dger", "dgnt")
ANGLES = (0, 15, 30, 45, 60, 75)
BUDGETS = (1.0, 0.2, 0.1, 0.05)
SCOPES = ("all", "source_train", "source_validation", "target")


def _run_dir(config: dict, repo_root: Path, method: str, target_angle: int,
             seed: int, budget: float) -> Path:
    root = Path(config["results_root"])
    if not root.is_absolute():
        root = repo_root / root
    return root / config["track"] / method / (
        f"target_{target_angle}_seed_{seed}_budget_{budget}"
    )


def _pairs_to_tensors(cache, pairs: Sequence[tuple[int, int]]) -> tuple[torch.Tensor, ...]:
    domains = torch.tensor([domain for domain, _ in pairs], dtype=torch.long)
    indices = torch.tensor([index for _, index in pairs], dtype=torch.long)
    return (
        cache.images[domains, indices],
        cache.labels[domains, indices],
        cache.angles[domains],
        cache.mnist_indices[domains, indices],
    )


def _dataset_tensors(cache, config: dict, scope: str) -> tuple[torch.Tensor, ...]:
    if scope == "all":
        count = cache.images.shape[1]
        domains = torch.arange(len(cache.angles)).repeat_interleave(count)
        return (
            cache.images.reshape(-1, *cache.images.shape[2:]),
            cache.labels.reshape(-1),
            cache.angles[domains],
            cache.mnist_indices.reshape(-1),
        )

    fold = make_fold(
        cache,
        int(config["target_angle"]),
        int(config["seed"]),
        float(config["data_budget"]),
    )
    if scope == "source_train":
        pairs = fold.train.pairs
    elif scope == "source_validation":
        pairs = fold.validation.pairs
    elif scope == "target":
        pairs = fold.target.pairs
    else:
        raise ValueError(f"Unsupported embedding scope: {scope}")
    return _pairs_to_tensors(cache, pairs)


def _sample_points(images: torch.Tensor, labels: torch.Tensor,
                   angles: torch.Tensor, mnist_indices: torch.Tensor,
                   max_points: int, random_state: int) -> tuple[torch.Tensor, ...]:
    if max_points <= 0 or len(images) <= max_points:
        return images, labels, angles, mnist_indices
    generator = np.random.default_rng(random_state)
    selected = np.sort(generator.choice(len(images), max_points, replace=False))
    selected_tensor = torch.from_numpy(selected)
    return tuple(value[selected_tensor] for value in
                 (images, labels, angles, mnist_indices))


def _extract_features(method, images: torch.Tensor,
                      device: torch.device) -> np.ndarray:
    method.eval()
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), 256):
            batch = ((images[start:start + 256] - 0.1307) / 0.3081).to(device)
            features.append(method.network(batch).features.cpu())
    return torch.cat(features).numpy()


def _load_features(run_dir: Path, config: dict, cache, scope: str,
                   device: torch.device, max_points: int,
                   random_state: int) -> tuple[np.ndarray, ...]:
    images, labels, angles, mnist_indices = _dataset_tensors(cache, config, scope)
    images, labels, angles, mnist_indices = _sample_points(
        images, labels, angles, mnist_indices, max_points, random_state,
    )
    method = _create_method(config, len(config.get("source_angles", [])))
    checkpoint = torch.load(
        run_dir / "best_source_val.pt",
        map_location=device,
        weights_only=False,
    )
    method.load_state_dict(checkpoint["model"])
    method.to(device)
    features = _extract_features(method, images, device)
    return features, labels.numpy(), angles.numpy(), mnist_indices.numpy()


def _save_embedding_plot(embedding: np.ndarray, labels: np.ndarray,
                        angles: np.ndarray, title: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for angle in ANGLES:
        mask = angles == angle
        axes[0].scatter(
            embedding[mask, 0], embedding[mask, 1], s=7, alpha=0.55,
            label=f"{angle}°",
        )
    axes[0].set_title("Colored by rotation domain")
    axes[0].legend(title="Angle", markerscale=2, fontsize=8)
    axes[0].set_xlabel("component 1")
    axes[0].set_ylabel("component 2")

    for label in range(10):
        mask = labels == label
        axes[1].scatter(
            embedding[mask, 0], embedding[mask, 1], s=7, alpha=0.55,
            label=str(label),
        )
    axes[1].set_title("Colored by digit label")
    axes[1].legend(title="Digit", markerscale=2, fontsize=8, ncol=2)
    axes[1].set_xlabel("component 1")
    axes[1].set_ylabel("component 2")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_visualizations(features: np.ndarray, labels: np.ndarray,
                         angles: np.ndarray, output_dir: Path,
                         title_prefix: str, random_state: int,
                         umap_neighbors: int, umap_min_dist: float) -> None:
    from sklearn.decomposition import PCA

    output_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(features)
    _save_embedding_plot(
        pca_embedding, labels, angles, f"{title_prefix} — PCA",
        output_dir / "pca.png",
    )

    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "UMAP requires umap-learn. Install it with "
            "python -m pip install umap-learn>=0.5."
        ) from error

    neighbors = min(umap_neighbors, max(2, len(features) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    umap_embedding = reducer.fit_transform(features)
    _save_embedding_plot(
        umap_embedding, labels, angles, f"{title_prefix} — UMAP",
        output_dir / "umap.png",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-angle", type=int, required=True, choices=ANGLES)
    parser.add_argument("--data-budget", type=float, required=True, choices=BUDGETS)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS),
        help="Methods to visualize; default is all four.",
    )
    parser.add_argument(
        "--scope", choices=SCOPES, default="all",
        help="Points to embed (default: all 6,000 cached domain images).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Root output directory; defaults to each run's embeddings folder.",
    )
    parser.add_argument(
        "--max-points", type=int, default=6000,
        help="Maximum points per run; 0 means all points (default: 6000).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if arguments.max_points < 0:
        raise ValueError("--max-points must be non-negative.")
    if arguments.umap_neighbors < 2:
        raise ValueError("--umap-neighbors must be at least 2.")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = arguments.config.expanduser().resolve()
    base_config = yaml.safe_load(config_path.read_text())
    device = resolve_device(arguments.device)

    for method_name in arguments.methods:
        for seed in arguments.seeds:
            run_dir = _run_dir(
                base_config, repo_root, method_name, arguments.target_angle,
                seed, arguments.data_budget,
            )
            resolved_path = run_dir / "resolved_config.yaml"
            checkpoint_path = run_dir / "best_source_val.pt"
            if not resolved_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Completed run artifacts not found in {run_dir}. "
                    "Run the experiment first."
                )
            config = yaml.safe_load(resolved_path.read_text())
            data_root = Path(config["data_root"])
            if not data_root.is_absolute():
                data_root = repo_root / data_root
            cache = build_or_load_cache(
                data_root, int(config["dataset_seed"]), tuple(config["angles"]),
            )
            features, labels, angles, mnist_indices = _load_features(
                run_dir, config, cache, arguments.scope, device,
                arguments.max_points, arguments.random_state,
            )
            if arguments.output_dir:
                output_dir = arguments.output_dir.expanduser()
                if not output_dir.is_absolute():
                    output_dir = repo_root / output_dir
                output_dir = output_dir / f"{method_name}_seed_{seed}"
            else:
                output_dir = run_dir / "embeddings" / arguments.scope
            title = (
                f"{method_name}, seed {seed}, target {arguments.target_angle}°, "
                f"budget {arguments.data_budget}, {arguments.scope}"
            )
            _save_visualizations(
                features, labels, angles, output_dir, title,
                arguments.random_state, arguments.umap_neighbors,
                arguments.umap_min_dist,
            )
            (output_dir / "metadata.json").write_text(json.dumps({
                "method": method_name,
                "seed": seed,
                "target_angle": arguments.target_angle,
                "data_budget": arguments.data_budget,
                "scope": arguments.scope,
                "num_points": int(len(features)),
                "latent_dimension": int(features.shape[1]),
                "angles": sorted(set(int(value) for value in angles)),
                "mnist_indices": mnist_indices.tolist(),
            }, indent=2))
            print(f"Saved PCA and UMAP plots to {output_dir}")


if __name__ == "__main__":
    main()


