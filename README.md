# RotatedMNIST domain generalization

This project implements the controlled RotatedMNIST comparison specified in
[implementation.md](implementation.md): DeepAll, DGER, DNT, and DGNT use the
same 64-dimensional MNIST CNN, data split, optimizer, and source-validation
selection rule. The accompanying [theory.md](theory.md) documents the
objectives and their assumptions. DIRT is intentionally not included because
it needs a separately trained image translation model and is outside this
experiment.

## What is reproducible

- Six cached domains at clockwise `0, 15, 30, 45, 60, 75` degrees. Every
  domain is generated from the same deterministic 1,000-image, 100-per-class
  MNIST subset; the selected indices and exact rotation policy are stored next
  to the cache.
- The comparison track uses leave-one-angle-out folds with source-only
  stratified 90/10 validation, proportional data budgets, and exact 64-item
  near-balanced batches (13/13/13/13/12, rotating the short source). DNT and
  DGNT use 64 same-class, cross-domain pairs. The original-DGER track trains on all
  1,000 examples per source domain and has no validation holdout.
- A documented five-point uniform DNT path (`0, .25, .5, .75, 1`), mean L2
  endpoint consistency, and the reported three-layer convolutional interpolator.
  The paper does not report the path-point count or convolution dimensions.
  DGNT adds interpolation to the first DGER phase and preserves the subsequent
  alternating auxiliary phases; its standalone DGER baseline uses those same
  phases and differs only by the interpolation term/network.
  The original-DGER reconstruction follows Algorithm 1's alternating per-domain updates with
  fresh samples and explicit frozen-parameter boundaries.
- CUDA is selected automatically when available. Pass `device: cuda` to fail
  rather than fall back to CPU; deterministic-kernel settings and GPU/version
  metadata are recorded in every run.

## Setup

Use a CUDA-compatible PyTorch build appropriate for the NVIDIA driver, then
install the project:

```bash
python3 -m pip install -e '.[dev]'
```

The generic `requirements.txt` intentionally does not pin a CUDA wheel URL;
selecting that wheel is platform and driver dependent. See the official
PyTorch installer for the matching CUDA index before running an experiment.

## Run a smoke experiment

This downloads MNIST once into `data/`, creates the deterministic cache, and
stores artifacts under `results/`. Start with a single seed, target, and
method rather than launching the full 480-run matrix.

```bash
PYTHONPATH=src python3 src/run_experiment.py \
  --config configs/rotated_mnist_target.yaml \
  --method deepall --target-angle 0 --seed 0
```

For the Algorithm-1-aligned DGER reconstruction (3,000 complete Algorithm 1
iterations, separate auxiliary learning rate, and declared final-iteration
selection policy):

```bash
PYTHONPATH=src python3 src/run_experiment.py \
  --config configs/rotated_mnist_dger_original.yaml
```

Run tests with `PYTHONPATH=src python3 -m pytest --cov=dg --cov-fail-under=80`. Tests cover the backbone,
gradient reversal, class-stratified budgeting, pair constraints, and
interpolation gradients. After runs finish, aggregate target analysis metrics:

```bash
PYTHONPATH=src python3 src/evaluate.py results --require-paper-matrix
```

The target-comparison CLI accepts `--method`, `--target-angle`, `--seed`, and
`--data-budget` overrides. This makes the complete matrix explicit and easy to
schedule on an NVIDIA worker; run all 100% jobs first, then the 20%, 10%, and
5% jobs. Do not use target accuracy to decide which jobs or checkpoints to
keep.

## Run diagnostic ablations

Use the dedicated sweep to isolate the observed DNT/DGER/DGNT optimization
instability. Its defaults screen targets 30° and 75° with seeds 0, 1, and 2:

```bash
PYTHONPATH=src python3 src/ablation_sweep.py \
  --config configs/rotated_mnist_target.yaml \
  --data-budget 1.0
```

The first-stage default includes a DeepAll control; DNT `lambda=0`, identity,
zero-initialized residual, and endpoint-`sqrt(64)` variants; and alternating
versus two-step DGER schedules. Once DNT and DGER are stable, add
`--methods dgnt` to screen the corresponding DGNT controls. Use `--methods`,
`--target-angles`, `--seeds`, or `--variants` to narrow it. Preview commands without training:

```bash
PYTHONPATH=src python3 src/ablation_sweep.py \
  --config configs/rotated_mnist_target.yaml \
  --methods dnt dger --target-angles 75 --seeds 0 \
  --data-budget 0.1 --dry-run
```

Ablations are saved below `results/target_comparison/ablations/`, marked
`paper_comparable: false`, and excluded from the paper-result aggregator. The
two-step option preserves the alternating route's sampled domain episodes,
loss definitions, reductions, and coefficients, while grouping them into one
stabilizer update and one main-model update. It is diagnostic rather than the
official RotatedMNIST schedule. Result paths include a full configuration and
source-code fingerprint so `--skip-existing` cannot silently reuse stale runs.

To isolate the historical DNT/DGNT change, run the endpoint-history matrix:

```bash
PYTHONPATH=src .venv/bin/python src/ablation_sweep.py \
  --config configs/rotated_mnist_target.yaml \
  --matrix endpoint_history --methods dnt dger dgnt \
  --target-angles 30 75 --seeds 0 --data-budget 1.0
```

For DNT and DGNT this crosses the historical three-layer MLP versus Conv1d
interpolator with historical all-element MSE versus mean per-example L2
endpoint loss. DGER is run once per target and seed as a shared reference; it
has no interpolator or endpoint loss. All cells retain the current data,
pairing, optimizer, path loss, and update protocols.

## Run artifacts and evaluation policy

Comparison runs record per-epoch losses, source validation per angle, `last.pt`,
and `best_source_val.pt` selected exclusively by macro source-validation
accuracy. The original-DGER track records per-iteration losses, periodically
updates `last.pt`, writes `paper_final.pt` at iteration 3,000, and evaluates the
target once from that final state.
Checkpoint RNG states and all active optimizer states are retained. At
inference, `predict` uses only the encoder and main classifier; DGER/DGNT
discard their discriminator and auxiliary heads, and DNT/DGNT discard the
interpolator.
