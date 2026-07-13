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
- Leave-one-angle-out folds with source-only stratified 90/10 validation,
  proportional data budgets, and domain-balanced batches (12 samples from
  each of five source domains, hence an effective batch of 60).
- Fixed five-point DNT path (`0, .25, .5, .75, 1`) and endpoint consistency.
  DGER retains its separate stabilizer fit and GRL-based adversarial update.
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

For the original-DGER-fidelity update schedule (3,000 iterations, separate
auxiliary learning rate):

```bash
PYTHONPATH=src python3 src/run_experiment.py \
  --config configs/rotated_mnist_dger_original.yaml
```

Run tests with `PYTHONPATH=src python3 -m pytest --cov=dg --cov-fail-under=80`. Tests cover the backbone,
gradient reversal, class-stratified budgeting, pair constraints, and
interpolation gradients. After runs finish, aggregate target analysis metrics:

```bash
PYTHONPATH=src python3 src/evaluate.py results
```

The target-comparison CLI accepts `--method`, `--target-angle`, `--seed`, and
`--data-budget` overrides. This makes the complete matrix explicit and easy to
schedule on an NVIDIA worker; run all 100% jobs first, then the 20%, 10%, and
5% jobs. Do not use target accuracy to decide which jobs or checkpoints to
keep.

## Run artifacts and evaluation policy

Each run records its input and resolved YAML configurations, environment and
git metadata, per-epoch JSONL losses, source validation per angle, target
metrics explicitly marked analysis-only, `last.pt`, and a
`best_source_val.pt` selected exclusively by macro source-validation accuracy.
Checkpoint RNG states and all active optimizer states are retained. At
inference, `predict` uses only the encoder and main classifier; DGER/DGNT
discard their discriminator and auxiliary heads, and DNT/DGNT discard the
interpolator.
