# RotatedMNIST Implementation Specification

This document specifies the implementation required to compare **DeepAll,
DGER, DNT, and DGNT** on RotatedMNIST using one shared MNIST CNN.

It makes explicit the reconstruction choices that are not fully specified by
the papers, so the experiment can be implemented without silently changing the
benchmark.

## 1. Scope and source material

Implement these four methods:

1. **DeepAll**: supervised pooled/domain-balanced ERM.
2. **DGER**: Domain Generalization via Entropy Regularization.
3. **DNT**: DeepAll plus nonlinear latent interpolation robustness.
4. **DGNT**: DGER plus interpolation robustness.

Primary sources:

- [Target paper](https://proceedings.mlr.press/v222/palakkadavath24a/palakkadavath24a.pdf)
- [Target supplement](https://proceedings.mlr.press/v222/palakkadavath24a/palakkadavath24a-supp.pdf)
- [DGER paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf)
- [DGER supplement](https://proceedings.neurips.cc/paper_files/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Supplemental.pdf)
- [Official DGER code](https://github.com/sshan-zhao/DG_via_ER)

DIRT is not part of this experiment. It requires a separate StarGAN/domain-
translation stage and is not needed for the requested comparison.

## 2. Two reproduction tracks

The target paper and original DGER paper use different training schedules. Run
both tracks explicitly instead of blending their settings.

### 2.1 Track A: target-paper-comparable comparison

Use this as the primary comparison of all four methods:

```text
optimizer:       SGD
learning rate:   0.001
momentum:        0.9
weight decay:    0.001
effective batch: 60 source examples
epochs:          100
latent size:     64
seeds:           5
```

The effective batch is 60 because five domains do not divide evenly into the
paper's reported batch size of 64. Use 12 examples per source domain per step.
This is a fairness reconstruction; record it in every run manifest.

Loss weights:

```text
DGER alpha_1 = 0.5
DGER alpha_2 = 0.005
DGER alpha_3 = 0.01
DNT/DGNT lambda = 1.0
```

The target supplement reports $\lambda=1$ for DNT and DGNT at every
RotatedMNIST data budget.

### 2.2 Track B: original-DGER fidelity check

Run DGER separately using the original RotatedMNIST settings reported by Zhao
et al.:

```text
main F/T/D learning rate: 1e-4
auxiliary T_i/T'_i rate:  1e-5
training budget:          3000 iterations
alpha_1:                  0.5
alpha_2:                  0.005
alpha_3:                  0.01
```

This is a DGER sanity check, not a controlled comparison with Track A. The
official DGER repository contains PACS code but no RotatedMNIST implementation,
so this track adapts its verified update logic to the shared MNIST CNN.

## 3. Suggested project layout

```text
configs/
    rotated_mnist_target.yaml
    rotated_mnist_dger_original.yaml

src/
    data/
        mnist_base.py
        rotated_mnist.py
        splits.py
        samplers.py
    models/
        mnist_cnn.py
        dger_modules.py
        interpolator.py
    methods/
        deepall.py
        dger.py
        dnt.py
        dgnt.py
    training/
        engine.py
        losses.py
        metrics.py
        checkpointing.py
        reproducibility.py
    run_experiment.py
    evaluate.py

tests/
    test_dataset.py
    test_splits.py
    test_mnist_cnn.py
    test_dger.py
    test_interpolation.py
    test_training_smoke.py

results/
    ... generated run artifacts ...
```

Expose every method through a common interface. The shared engine owns data
loading, validation, checkpoints, metrics, and seeding; method classes own
forward passes, losses, and parameter-update logic.

## 4. Canonical RotatedMNIST dataset

### 4.1 Dataset construction

Create six domains:

```text
domain_id  angle
0          0 degrees
1          15 degrees
2          30 degrees
3          45 degrees
4          60 degrees
5          75 degrees
```

Each domain contains 1,000 images, with exactly 100 examples per digit class.

Recommended construction:

1. Download the standard MNIST training split.
2. For each class, sample 100 images without replacement using a fixed seed.
3. Save the selected original MNIST indices.
4. Reuse those same 1,000 base images for every rotation domain.
5. Rotate each base image by the domain angle.

This follows the DGER paper's description of sampling 100 examples per class
for $M_0$ and creating the remaining domains with 15-degree rotations. The
papers do not publish the exact selected indices, so saving them is required
for reproducibility.

### 4.2 Rotation and preprocessing policy

Use one deterministic policy for all methods:

```text
rotation direction: clockwise
angles:             [0, 15, 30, 45, 60, 75]
interpolation:      bilinear
fill value:         0
output:             1 × 28 × 28
raw value range:    [0, 1]
normalization:      mean=0.1307, std=0.3081
```

This is a reconstruction: the papers do not specify the rotation library,
interpolation kernel, or fill behavior. Rotate once, cache the six domains, and
do not apply random rotations online in the primary experiment.

Cache:

```text
images:          float32 [6, 1000, 1, 28, 28]
labels:          int64   [6, 1000]
angles:          int64   [6]
mnist_index:     int64   [6, 1000]
dataset_seed:    integer
rotation_config: serialized policy
```

### 4.3 Leave-one-angle-out evaluation

For each target angle $a_t$:

```text
source_domains = all angles except target angle
target_domain  = the held-out target angle
```

For every source domain independently:

1. Stratify by class.
2. Split 90% into training and 10% into validation.
3. Preserve the target domain's entire 1,000 images for testing.

At full data, each fold contains approximately:

```text
source train: 5 × 900 = 4,500 images
source val:   5 × 100 = 500 images
target test:  1 × 1,000 = 1,000 images
```

For 20%, 10%, and 5% settings, subsample source train and validation data
proportionally while preserving class counts. Never subsample or alter target
test data.

### 4.4 Dataset item contract

Every dataset item must return:

```python
{
    "image": FloatTensor[1, 28, 28],
    "label": LongTensor scalar,
    "domain": LongTensor scalar,
    "angle": integer,
    "mnist_index": integer,
}
```

Retain domain and original-index metadata in validation/test loaders for leakage
checks and per-angle metrics.

## 5. Shared MNIST CNN

The target paper specifies two convolutional layers, one dense feature layer,
and a 64-dimensional latent representation. It does not specify widths,
kernels, pooling, or dropout. Use this explicit reconstruction for every
method:

```text
Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
ReLU
MaxPool2d(kernel_size=2, stride=2)

Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
ReLU
MaxPool2d(kernel_size=2, stride=2)

Flatten
Linear(64 * 4 * 4, 64)
ReLU

Classifier: Linear(64, 10)
```

The spatial shape is:

```text
28 × 28 -> Conv5 -> 24 × 24 -> Pool -> 12 × 12
       -> Conv5 ->  8 ×  8 -> Pool ->  4 ×  4
```

The model API must return:

```text
features: [batch, 64]
logits:   [batch, 10]
```

Do not use dropout, batch normalization, random crops, flips, or color
augmentation in the primary experiment.

### 5.1 DGER modules

For five source domains in each fold:

```text
domain discriminator D:  Linear(64, 5)
stabilizers T_i:         five independent Linear(64, 10) heads
entropy heads T'_i:      five independent Linear(64, 10) heads
```

Use gradient reversal between the encoder and $D$, and between the encoder and
$T'_i$. Stabilizers $T_i$ do not use gradient reversal.

### 5.2 DNT interpolator

The target paper describes $T_\psi$ as a three-layer convolutional network,
although its input is a latent vector. Use this vector-compatible reconstruction
for the primary experiment:

```text
Linear(64, 64)
ReLU
Linear(64, 64)
ReLU
Linear(64, 64)
```

Keep the input/output shape `[batch, 64]`. A 1D-convolution implementation may
be added as an architectural ablation, not substituted silently.

## 6. Shared training, validation, and checkpoints

### 6.1 Primary optimizer

Track A uses:

```text
optimizer:       SGD
learning rate:   0.001
momentum:        0.9
weight decay:    0.001
effective batch: 60
epochs:          100
```

Use a constant learning rate in the primary experiment. The target materials
do not specify a RotatedMNIST scheduler; scheduler use is a later ablation.

### 6.2 Balanced source sampler

Each training step draws 12 examples from each of the five source domains.
Define one epoch as the length of the longest source loader; restart shorter
domain iterators when they exhaust. This matches the official DGER loader
behavior while giving all methods the same domain-balanced data exposure.

The DNT pair sampler is independent of this ordinary classification batch.

### 6.3 Model selection

At the end of each epoch:

1. Evaluate each source validation domain.
2. Compute macro-average source-validation accuracy.
3. Save the checkpoint if the macro-average improves.
4. Never use target accuracy for checkpoint selection or hyperparameter choice.

Save:

```text
last.pt
best_source_val.pt
best_source_val.json
```

Checkpoint contents:

```text
all active model state dicts
optimizer state
scheduler state, if any
epoch and global step
best validation metric
resolved configuration
random-number-generator states
```

At inference, retain only:

```text
DeepAll: encoder + classifier
DNT:     encoder + classifier
DGER:    encoder + main classifier
DGNT:    encoder + main classifier
```

Discard DGER auxiliaries/discriminator and DNT interpolator.

## 7. DeepAll

DeepAll uses only:

```python
logits, features = model(images)
loss = cross_entropy(logits, labels)
```

Domain IDs are used only by the balanced sampler and evaluation code.

```text
initialize E and C
for epoch in 1..100:
    for balanced source batch (x, y, domain):
        logits, z = E_C(x)
        loss = CE(logits, y)
        update E and C
    evaluate source validation
    save best source-validation checkpoint
```

## 8. DNT

### 8.1 Pair sampler

Precompute:

```text
bucket[(domain_id, class_id)] -> sample indices
```

For each pair:

1. Sample a class uniformly from 0 through 9.
2. Sample a source domain $d$.
3. Sample a different source domain $d'$.
4. Sample $x$ from bucket $(d,y)$ and $x'$ from bucket $(d',y)$.

The pair must satisfy:

```text
y == y_prime
d != d_prime
```

### 8.2 Interpolation path and loss

Encode the pair:

$$
z=E_\phi(x),\qquad z'=E_\phi(x').
$$

Use:

$$
\Delta=T_\psi(z'-z),\qquad
\hat z(w)=z+w\Delta.
$$

The primary reconstruction uses:

```text
w = [0.0, 0.25, 0.5, 0.75, 1.0]
```

The paper describes a uniformly sampled path but does not specify the number
of points. The fixed five-point grid is therefore a documented assumption.

Use:

$$
\mathcal{L}_{\mathrm{cls}}
=\mathrm{CE}(C_\theta(E_\phi(x)),y),
$$

$$
\mathcal{L}_{\mathrm{path}}
=\left[
\mathrm{CE}(C_\theta(\hat z(0)),y)
 + \mathrm{CE}(C_\theta(\hat z(0.25)),y)
 + \mathrm{CE}(C_\theta(\hat z(0.5)),y)
 + \mathrm{CE}(C_\theta(\hat z(0.75)),y)
 + \mathrm{CE}(C_\theta(\hat z(1)),y)
\right]/5,
$$

$$
\mathcal{L}_{\mathrm{end}}
=\left\|T_\psi(z'-z)-(z'-z)\right\|_2^2,
$$

$$
\mathcal{L}_{\mathrm{int}}
=\mathcal{L}_{\mathrm{path}}+\mathcal{L}_{\mathrm{end}},
\qquad
\mathcal{L}_{\mathrm{DNT}}
=\mathcal{L}_{\mathrm{cls}}+\lambda\mathcal{L}_{\mathrm{int}},
\quad \lambda=1.
$$

### 8.3 DNT pseudocode

```text
initialize E, C, and T_psi

for epoch in 1..100:
    for balanced source batch:
        sample same-class cross-domain pairs
        z       = E(x)
        z_prime = E(x_prime)
        delta   = T_psi(z_prime - z)

        L_cls = CE(C(z), y)
        L_path = 0
        for w in [0, .25, .5, .75, 1]:
            L_path += CE(C(z + w * delta), y) / 5

        L_end = MSE(delta, z_prime - z)
        L = L_cls + L_path + L_end
        update E, C, and T_psi

    evaluate source validation
    save best checkpoint
```

Add random $w\sim U(0,1)$, identity $T_\psi$, and endpoint-loss-disabled
versions as ablations.

## 9. DGER

### 9.1 Loss components

The practical DGER configuration is:

$$
\mathcal{L}_{\mathrm{DGER}}
=\mathcal{L}_{\mathrm{cls}}
+\alpha_1\mathcal{L}_{\mathrm{adv}}
+\alpha_2\mathcal{L}_{\mathrm{er}}
+\alpha_3\mathcal{L}_{\mathrm{cel}},
$$

with:

```text
alpha_1 = 0.5
alpha_2 = 0.005
alpha_3 = 0.01
```

Use:

```text
L_cls: main 10-class classification loss
L_adv: 5-class domain loss through gradient reversal
L_er:  entropy/GRL classifier losses T'_i
L_cel: stabilizing own-domain/cross-domain classifier losses T_i
```

Use class-balanced auxiliary losses and inverse domain-size weighting for the
domain discriminator. With equal RotatedMNIST domain sizes, domain weighting is
uniform, but implement the weighting generically.

### 9.2 DGER update order

For a composite batch containing 12 examples from every source domain:

#### Phase A: train stabilizers

```text
freeze F, T, D, and T'_i
enable T_i
compute z_detached = stop_gradient(F(x))
for every source domain i:
    select examples with domain == i
    update T_i using class-balanced CE on its own-domain features
```

#### Phase B: train main DGER system

```text
enable F, T, D, and T'_i
freeze T_i

z, main_logits = F_T(x)
L_cls = CE(main_logits, y)
L_adv = CE(D(GRL(z)), domain)
L_er  = sum_i CE(T'_i(GRL(z_i)), y_i)
L_cel = sum_i CE(T_i(z_not_i), y_not_i)

L = L_cls + .5*L_adv + .005*L_er + .01*L_cel
update F, T, D, and T'_i
```

The stabilizer $T_i$ is frozen during the cross-domain feature update. The
domain discriminator and $T'_i$ use GRL; $T_i$ does not.

### 9.3 DGER pseudocode

```text
initialize F, T, D, T_1..T_5, and T'_1..T'_5

for epoch in 1..100:
    for balanced source batch:
        freeze F
        z_detached = stop_gradient(F(x))
        for i in 1..5:
            update T_i on own-domain examples

        freeze T_i
        z, logits = F_T(x)
        compute L_cls, L_adv, L_er, and L_cel
        L = L_cls + .5*L_adv + .005*L_er + .01*L_cel
        update F, T, D, and T'_i

    evaluate source validation
    save best checkpoint
```

### 9.4 Original-DGER track

Track B uses separate learning-rate parameter groups:

```text
F/T/D:      1e-4
T_i/T'_i:   1e-5
iterations: 3000
```

Log Track B separately from the target-paper-comparable Track A.

## 10. DGNT

DGNT is DGER plus DNT:

$$
\mathcal{L}_{\mathrm{DGNT}}
=\mathcal{L}_{\mathrm{DGER}}
+\lambda\mathcal{L}_{\mathrm{int}},
\qquad \lambda=1.
$$

The primary implementation applies DGER losses to original source examples and
adds interpolation loss to the encoder/classifier/interpolator update. Applying
DGER auxiliary losses to interpolated features is a separate ablation.

```text
for each balanced source batch:
    update DGER stabilizers T_i
    compute DGER main/adversarial/entropy/stabilizer losses

    sample same-class cross-domain pairs
    compute interpolation path and endpoint loss

    add L_int to the F/T update
    update T_psi

    discard D, T_i, T'_i, and T_psi at inference
```

Control parameter sets explicitly so $T_i$, $T'_i$, and $D$ do not receive
interpolation gradients unless the ablation enables them.

## 11. Experiment matrix and outputs

### 11.1 Primary matrix

```text
methods:      DeepAll, DGER, DNT, DGNT
targets:      0, 15, 30, 45, 60, 75 degrees
data budgets: 100%, 20%, 10%, 5%
seeds:        5
```

This is 480 runs. Start with one target, one seed, and the 100% budget. Run the
full 100% matrix before adding limited-data settings.

### 11.2 Target-paper reference values

Use these nonlinear RotatedMNIST results as reproduction references:

| Model | 100% | 20% | 10% | 5% | Average |
|---|---:|---:|---:|---:|---:|
| DeepAll | 92.69 | 80.80 | 73.95 | 67.99 | 78.86 |
| DNT | 97.36 | 84.48 | 78.89 | 73.51 | 83.66 |
| DGER | 95.61 | 79.89 | 73.69 | 68.78 | 79.49 |
| DGNT | 95.92 | 83.85 | 77.27 | 72.38 | 82.36 |

These are reported means; the paper reports standard errors over five runs.
Exact equality is not expected because several implementation details are
under-specified.

### 11.3 Per-run artifact schema

Save for every method/budget/target/seed:

```text
config.yaml
resolved_config.yaml
git commit hash and dirty flag
seed
target angle
source angles
data budget
selected MNIST indices
train/validation indices
rotation configuration
model configuration
optimizer configuration
loss weights
metrics.jsonl
last.pt
best_source_val.pt
best_source_val.json
```

Each epoch's metrics should include:

```text
train loss components
train accuracy
source validation accuracy by angle
source validation macro accuracy
target accuracy for analysis only
best epoch so far
```

Final aggregation should include per-angle accuracy, mean per-class accuracy,
cross-entropy, standard deviation, and standard error.

## 12. Reproducibility controls

Seed all of:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA/MPS
DataLoader workers
dataset sampling
train/validation splits
DNT pair sampling
interpolation coefficients
```

Record Python, PyTorch, torchvision, device, OS, and deterministic-kernel
settings. Save the selected MNIST indices and all split indices; configuration
alone is insufficient to reproduce the dataset.

## 13. Tests

### 13.1 Dataset tests

```text
six domains exist
each domain contains 1,000 examples
each class has 100 examples per domain
images have shape [1, 28, 28]
labels are in [0, 9]
the same MNIST index appears at every angle
0-degree images equal the cached base images
target indices never appear in source train/validation
limited-data subsets preserve class counts
```

### 13.2 Shape and model tests

```text
encoder output: [batch, 64]
main logits: [batch, 10]
domain logits: [batch, 5]
auxiliary logits: [batch, 10]
interpolator output: [batch, 64]
```

### 13.3 DNT tests

```text
pair labels are equal
pair domains are different
w=0 returns z
identity T_psi with w=1 returns z_prime
endpoint loss is zero for matching displacement
L_int produces gradients for E, C, and T_psi
```

### 13.4 DGER tests

```text
domain discriminator has one output per source domain
own-domain T_i update does not update F
cross-domain stabilizer update does not update T_i
GRL reverses the encoder gradient direction
T'_i receives entropy/GRL gradients
inference uses only F and T
```

### 13.5 Smoke and overfit tests

Run one target angle, one batch per domain, and two update steps for every
method. Confirm that all losses are finite and checkpoints load.

On a tiny fixed source subset:

- DeepAll must overfit the labels;
- DNT must reduce classification and endpoint losses;
- DGER must reduce main classification loss without NaNs;
- DGNT must produce finite DGER and interpolation losses.

## 14. Ablations

Run after the primary matrix:

1. Domain-balanced versus pooled random batches.
2. Learned DNT interpolator versus identity/linear interpolation.
3. Five-point path versus one random $w\sim U(0,1)$.
4. Endpoint consistency enabled versus disabled.
5. DGER auxiliary terms enabled versus classification plus domain adversarial
   loss only.
6. DGNT interpolation applied only to $F/T$ versus all DGER components.
7. Lambda sweep:

   ```text
   [1, 0.1, 0.01, 0.001, 0.0001]
   ```

8. CNN width sensitivity.
9. Bilinear versus nearest-neighbor rotation.
10. Track A versus Track B DGER schedule.

Every ablation must identify which primary setting it changes.

## 15. Implementation order

1. Build and cache RotatedMNIST plus the index manifest.
2. Implement stratified leave-one-angle-out splits.
3. Implement the shared MNIST CNN and classifier.
4. Implement shared engine, metrics, checkpoints, and seeding.
5. Implement and validate DeepAll.
6. Implement DNT pair sampler, interpolator, and tests.
7. Implement DGER GRL modules, auxiliary losses, and update order.
8. Implement DGNT composition.
9. Run one-target/one-seed smoke experiments.
10. Run the full 100% data matrix.
11. Run 20%, 10%, and 5% data budgets.
12. Run ablations and aggregate results.

Do not launch the 480-run matrix until the smoke, shape, gradient, leakage, and
checkpoint tests pass.

## 16. Acceptance criteria

The implementation is complete when:

- all four methods run across all six target angles;
- all methods use the same cached dataset and MNIST CNN;
- target data is excluded from training, validation, and model selection;
- DGER has correct auxiliary modules, GRL behavior, and update order;
- DNT uses same-class, cross-domain pairs and endpoint-constrained paths;
- DGNT combines DGER and DNT without dropping either objective;
- five-seed means, standard deviations, and standard errors are reported;
- per-angle and macro-average results are saved;
- reference results are shown beside reproduced results;
- all unit, gradient, leakage, smoke, and checkpoint tests pass;
- every under-specified reconstruction choice appears in the run manifest.

## 17. Known limitations

Exact numerical equality with the target paper is not expected because it does
not publish:

- the exact MNIST sample indices;
- convolution widths and kernel sizes;
- rotation library/interpolation/fill details;
- exact DNT path length;
- how batch size 64 is divided among five source domains;
- the target-paper DGER learning-rate schedule;
- whether a scheduler was used for RotatedMNIST.

The correct outcome is therefore a deterministic, inspectable reproduction with
all reconstruction choices recorded, not an unsupported claim of exact
replication.
