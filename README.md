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
