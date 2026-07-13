# Domain Generalization Methods: Theory and Implementation

This document explains the methods used as baselines and extensions in
[Domain Generalization with Interpolation Robustness](https://proceedings.mlr.press/v222/palakkadavath24a/palakkadavath24a.pdf)
(Palakkadavath et al., ACML 2023/2024 proceedings). It combines the target
paper's equations and Algorithm 1 with the original DIRT and DGER papers and
the official DIRT implementation.

The document uses the following convention:

- **Source-derived** means the formulation is stated directly or nearly
  directly in a cited paper.
- **Implementation reconstruction** means the procedure is inferred from
  equations, prose, pseudocode, or official code. It is intended to be
  executable, but should not be confused with a verbatim algorithm listing.

## 1. Common domain-generalization setup

### 1.1 Problem definition

There are $K$ labeled source domains

$$
\mathcal{S}=\{S_1,\ldots,S_K\},\qquad
S_d=\{(x_i^d,y_i^d)\}_{i=1}^{n_d},
$$

where $x\in\mathcal{X}$ is an input, $y\in\mathcal{Y}$ is its class, and
$d\in\mathcal{D}_s$ is a domain identifier. The source domains are sampled
from distributions $P_d(X,Y)$. At training time, the learner has no samples
from the target domain $d_t\notin\mathcal{D}_s$. The goal is to learn a
predictor that performs well on $P_{d_t}(X,Y)$.

The target paper focuses on **covariate shift**: domains differ primarily in
their input distribution, while the task and label semantics remain shared.
Its examples include different visual styles in PACS and different rotation
angles in RotatedMNIST.

### 1.2 Representation-based classifier

Most methods decompose the predictor into an encoder and classifier:

$$
z=E_\phi(x),\qquad \hat p(y\mid x)=C_\theta(z)=C_\theta(E_\phi(x)).
$$

The encoder maps high-dimensional inputs into a latent representation $z$.
The classifier maps $z$ to class probabilities. For a labeled example, the
standard classification loss is cross-entropy:

$$
\mathcal{L}_{\mathrm{cls}}(\theta,\phi)
 =\mathbb{E}_{d,x,y}\left[-\log C_\theta(E_\phi(x))_y\right].
$$

All methods below retain this task objective. They differ in what additional
structure they impose on $E_\phi$, on the data, or on the latent space.

### 1.3 Evaluation protocol used by the target paper

The target paper uses leave-one-domain-out evaluation:

1. Select one domain as the unseen target.
2. Train on all remaining domains.
3. Split source data into 90% training and 10% validation data.
4. Select the checkpoint with the best source-validation performance.
5. Evaluate on the held-out target domain.
6. Repeat for every domain and average the results over five random seeds.

For limited-data experiments, the source training and validation sets are
sampled proportionally at 20%, 10%, and 5% of their original size while
preserving the class distribution $P_d(Y)$. The target test set is unchanged.

## 2. DeepAll / pooled ERM

### 2.1 Intuition

DeepAll is the ordinary deep-learning baseline: aggregate all labeled source
examples and train one classifier with the task loss. It does not explicitly
remove domain information, generate translated images, or simulate held-out
domains during training.

In the target paper, DeepAll is exactly the (\mathcal{L}_{\mathrm{cls}})-only
baseline. It is also commonly called ERM in modern domain-generalization
benchmarks.

### 2.2 Objective

For pooled empirical risk minimization:

$$
\min_{\theta,\phi}\;
\widehat R_{\mathrm{pool}}(\theta,\phi)
=\frac{1}{N}\sum_{d=1}^{K}\sum_{i=1}^{n_d}
\ell\left(C_\theta(E_\phi(x_i^d)),y_i^d\right),
\qquad N=\sum_{d=1}^{K}n_d.
$$

For classification, (\ell) is normally cross-entropy.

If each minibatch contains the same number of examples from every domain, the
implemented objective is closer to a domain-balanced risk:

$$
\min_{\theta,\phi}\;
\frac{1}{K}\sum_{d=1}^{K}
\frac{1}{n_d}\sum_{i=1}^{n_d}
\ell\left(C_\theta(E_\phi(x_i^d)),y_i^d\right).
$$

These objectives are equivalent only when domain sizes or sampling weights are
matched. The sampling policy therefore matters when source domains are
imbalanced.

### 2.3 Pseudocode

```text
Input: labeled source datasets S1, ..., SK
Initialize encoder E_phi and classifier C_theta

repeat until training ends:
    sample a minibatch of labeled examples from the source domains
    optionally balance the number of examples per domain

    z      = E_phi(x)
    logits = C_theta(z)
    loss   = cross_entropy(logits, y)

    update theta and phi using gradient descent on loss

Return E_phi and C_theta
At test time: predict C_theta(E_phi(x)) on the unseen domain
```

### 2.4 Assumptions and limitations

DeepAll assumes that sufficiently diverse source data and ordinary supervised
regularization will cause the network to learn transferable features. It does
not state or enforce a domain-invariance condition. As a result, the model can
use correlations that are predictive in the source mixture but are absent or
reversed in the target domain.

DeepAll remains important because every extension should be compared against
the same task model and data protocol. A claimed DG improvement is not
meaningful if it changes the backbone, sampling, validation rule, or data
budget at the same time.

## 3. DIRT: Domain Invariant Representation Learning with Domain Density Transformations

DIRT is introduced in Nguyen et al., *Domain Invariant Representation Learning
with Domain Density Transformations* (NeurIPS 2021). The original paper is
available [here](https://proceedings.neurips.cc/paper/2021/file/2a2717956118b4d223ceca17ce3865e2-Paper.pdf),
and the authors' implementation is available in the
[official DIRT repository](https://github.com/atuannguyen/DIRT).

### 3.1 Motivation

DIRT argues that aligning only the marginal feature distribution is not enough.
The representation should satisfy both:

**Marginal alignment**

$$
p_d(z)=p_{d'}(z),
$$

and **conditional alignment**

$$
p_d(y\mid z)=p_{d'}(y\mid z).
$$

The method learns transformations between source domains and forces the
representation of an image to match the representation of its translated
version.

### 3.2 Probabilistic assumptions

DIRT uses the factorization

$$
p(d,x,y,z)=p(d)p(y)p(x\mid y,d)p(z\mid x).
$$

The important label assumption is that $p(y\mid d)=p(y)$: the class prior is
stable across domains. The paper shows that this condition is necessary and
sufficient for the existence of a representation that aligns both marginal and
conditional distributions, although a trivial constant representation also
satisfies the theorem and is not useful for prediction.

For every pair of domains $(d,d')$, DIRT assumes an invertible,
differentiable transformation $f_{d,d'}$ such that

$$
x'=f_{d,d'}(x)
$$

maps the class-conditional density from domain $d$ to domain $d'$:

$$
p(x\mid y,d)\longrightarrow p(x'\mid y,d').
$$

The inverse is $f_{d',d}=f_{d,d'}^{-1}$. Under these assumptions, the
change-of-variables relation is

$$
p(x\mid y,d)
=p(x'\mid y,d')
\left|\det J_{f_{d',d}}(x')\right|^{-1}.
$$

### 3.3 Invariance theorem and objective

If the representation is invariant under the domain transformation,

$$
p(z\mid x)=p(z\mid f_{d,d'}(x)),
$$

then DIRT's theorem shows that the representation aligns both the marginal and
conditional distributions between $d$ and $d'$.

For a deterministic encoder $z=g_\phi(x)$, the practical objective is:

$$
\mathcal{L}_{\mathrm{DIRT}}
=\mathcal{L}_{\mathrm{cls}}
+\alpha\,\mathcal{L}_{\mathrm{dirt}},
$$

where

$$
\mathcal{L}_{\mathrm{dirt}}
=\mathbb{E}_{d,d',x,y}
\left[
\left\|g_\phi(x)-g_\phi(f_{d,d'}(x))\right\|_2^2
\right].
$$

The squared feature distance is the implementation of the theoretical
invariance constraint. The classifier is trained on the original examples;
the translated examples primarily provide the representation-consistency
signal.

### 3.4 Stage 1: learning domain transformations with StarGAN

The theory requires $f_{d,d'}$, but these transformations are not known in a
real dataset. DIRT learns them with a multi-domain image-to-image generator.
The implementation uses a StarGAN-style generator:

$$
x'=G(x,d,d'),
$$

where $d$ is the source domain and $d'$ is the requested destination
domain. The explicit source-domain conditioning is a DIRT implementation
choice; the theoretical method only requires a transformation with the stated
properties.

The generator is trained with:

- an adversarial loss so generated images look real;
- a domain-classification loss so the generated image appears to belong to
  (d');
- a reconstruction/cycle loss so content and class information are preserved.

For $x'=G(x,d,d')$, the reconstruction term is:

$$
\mathcal{L}_{\mathrm{rec}}
=\mathbb{E}_{x,d,d'}
\left[\left\|x-G(x',d',d)\right\|_1\right].
$$

The official StarGAN implementation uses WGAN-style adversarial training with
gradient penalty. A representative discriminator update is:

```text
d_loss_real = -mean(D_src(x_real))
d_loss_fake =  mean(D_src(x_fake))
d_loss_cls  = cross_entropy(D_cls(x_real), domain_label)
d_loss_gp   = gradient_penalty(D_src, x_real, x_fake)
d_loss      = d_loss_real + d_loss_fake
              + lambda_cls * d_loss_cls
              + lambda_gp  * d_loss_gp
```

The corresponding generator update is:

```text
g_loss_fake = -mean(D_src(x_fake))
g_loss_cls  = cross_entropy(D_cls(x_fake), target_domain)
g_loss_rec  = mean(abs(x_real - x_reconstructed))
g_loss      = g_loss_fake + lambda_cls * g_loss_cls
              + lambda_rec * g_loss_rec
```

The official implementation uses Adam for StarGAN and typically uses
(\lambda_{\mathrm{cls}}=1), (\lambda_{\mathrm{rec}}=10),
(\lambda_{\mathrm{gp}}=10), and five discriminator updates per generator
update. These are implementation defaults, not universal theoretical
requirements.

### 3.5 Stage 2: learning the invariant representation

After training the generator, freeze it and use

$$
f_{d,d'}(x):=G(x,d,d').
$$

For each source example, sample a destination domain $d'$, generate $x'$,
and minimize the classification plus feature-consistency objective.

```text
Input: source samples (x, y, d), trained StarGAN generator G
Initialize encoder g_phi and classifier h

repeat until representation training ends:
    sample a labeled source minibatch (x, y, d)
    sample destination domains d' != d

    x_new = G(x, d, d')       # G is frozen
    z     = g_phi(x)
    z_new = g_phi(x_new)

    loss_cls  = cross_entropy(h(z), y)
    loss_dirt = alpha * squared_l2(z_new - z)
    loss      = loss_cls + loss_dirt

    update phi and h using loss

Return g_phi and h; discard G and the StarGAN discriminator at inference
```

### 3.6 What DIRT does and does not guarantee

The exact theorem assumes that the learned transformations preserve class
conditional densities and are invertible. StarGAN only approximates those
properties. In particular:

- a translation can change class-relevant content;
- cycle consistency does not prove semantic correctness;
- the generator may overfit when the source dataset is small;
- the representation penalty only inherits the quality of the generated
  transformations;
- StarGAN adds substantial training complexity, although it is not needed at
  test time.

The target paper specifically observes that StarGAN translations become
incoherent on PACS when only about 5% of the source data is available. This is
the main motivation for adding interpolation robustness as a complementary
regularizer.

## 4. DGER: Domain Generalization via Entropy Regularization

DGER is introduced in Zhao et al., *Domain Generalization via Entropy
Regularization* (NeurIPS 2020). See the [paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf)
and [supplement](https://proceedings.neurips.cc/paper_files/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Supplemental.pdf).

### 4.1 Motivation

Adversarial domain alignment commonly makes the marginal feature distributions
similar:

$$
P_1(F(X))=\cdots=P_K(F(X)).
$$

This does not imply that the label posterior is shared:

$$
P_1(Y\mid F(X))=\cdots=P_K(Y\mid F(X)).
$$

DGER combines marginal adversarial alignment with an entropy-based objective
designed to reduce domain-specific differences in the conditional label
distribution.

### 4.2 Model components

DGER uses several networks:

- $F_\theta$: shared feature extractor;
- $T_\phi$: main classifier used for prediction;
- $D_\psi$: domain discriminator used for marginal alignment;
- $T'_{i,\phi'_i}$: one entropy-regularization classifier per source domain;
- $T_{i,\phi_i}$: one stabilizing classifier per source domain.

Only $F_\theta$ and $T_\phi$ are needed at inference time. The discriminator
and auxiliary classifiers exist to shape the learned feature representation.

### 4.3 Classification and adversarial losses

The supervised loss is:

$$
\mathcal{L}_{\mathrm{cls}}
=-\sum_{i=1}^{K}
\mathbb{E}_{(X,Y)\sim P_i}
\left[\log Q_T(Y\mid F(X))\right].
$$

The domain discriminator receives $F(X)$ and predicts the source-domain
identity. In minimax notation, the adversarial objective is:

$$
\min_F\max_D\;\mathcal{L}_{\mathrm{adv}}
=\sum_{i=1}^{K}
\mathbb{E}_{X\sim P_i(X)}
\left[\log D_i(F(X))\right].
$$

The discriminator maximizes its ability to identify the domain, while $F$
minimizes the same value and therefore tries to make domains indistinguishable.
In code this min-max behavior is commonly implemented using a gradient-reversal
layer.

### 4.4 Conditional-invariance objective

DGER starts from the desired KL objective:

$$
\min_{F,T}\sum_{i=1}^{K}
\mathrm{KL}\left(P_i(Y\mid F(X))\;\middle\|\;
Q_T(Y\mid F(X))\right).
$$

Expanding the KL divergence gives:

$$
\begin{aligned}
\sum_i \mathrm{KL}(P_i\|Q_T)
&=\sum_i \mathbb{E}_{P_i}
\left[\log P_i(Y\mid F(X))\right]\\
&\quad-\sum_i\mathbb{E}_{P_i}
\left[\log Q_T(Y\mid F(X))\right].
\end{aligned}
$$

The second term is the classification loss. The first is the sum of negative
conditional entropies:

$$
\sum_i \mathbb{E}_{P_i}[\log P_i(Y\mid F(X))]
=-\sum_i H_{P_i}(Y\mid F(X)).
$$

The true posterior $P_i(Y\mid F(X))$ is unknown, so DGER estimates the
regularizer adversarially.

### 4.5 Entropy regularization and Jensen–Shannon divergence

The paper assumes equal class priors for the entropy derivation. Under this
assumption, minimizing negative conditional entropy is equivalent, up to a
constant, to minimizing the Jensen–Shannon divergence among the class-
conditional feature distributions:

$$
P_i(F(X)\mid Y=1),\ldots,P_i(F(X)\mid Y=C).
$$

The minimum is reached when, within each domain,

$$
P_i(F(X)\mid Y=1)=\cdots=P_i(F(X)\mid Y=C).
$$

This result should be interpreted together with the classification loss. The
classification loss keeps classes discriminable by the main classifier, while
the entropy regularizer discourages domain-specific conditional structure in
the feature distributions. The theoretical statement is a global-optimum
statement for the minimax game, not a guarantee that finite neural-network
optimization reaches the ideal solution.

### 4.6 Entropy-classifier minimax game

For each source domain, DGER introduces $T'_i$. Its objective is:

$$
\min_F\max_{\{T'_i\}_{i=1}^{K}}
\mathcal{L}_{\mathrm{er}}
=\sum_{i=1}^{K}
\mathbb{E}_{(X,Y)\sim P_i}
\left[\log Q_{T'_i}^{i}(Y\mid F(X))\right].
$$

The auxiliary classifier maximizes its domain-specific label prediction
ability. The feature extractor minimizes the value, making the domain-specific
class-conditional feature distributions harder to distinguish.

The first combined objective is therefore:

$$
\min_{F,T}\max_{D,\{T'_i\}}
\mathcal{L}_{\mathrm{cls}}
+\alpha_1\mathcal{L}_{\mathrm{adv}}
+\alpha_2\mathcal{L}_{\mathrm{er}}.
$$

### 4.7 Stabilizing cross-entropy loss

The authors report that directly optimizing the minimax objective can be
unstable. They add a second set of classifiers $T_i$ and a stabilizing loss.
For domain $i$, $T_i$ is trained on examples from its own domain with the
feature extractor fixed, and the feature extractor is also trained so that
$T_i$ remains useful on examples from the other domains.

Using bars to indicate frozen parameters, the paper writes:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{cel}}
={}&-\sum_{i=1}^{K}
\mathbb{E}_{(X,Y)\sim P_i}
\left[\log Q_{T_i}^{i}(Y\mid \bar F(X))\right]\\
&-\sum_{i=1}^{K}\sum_{j\ne i}
\mathbb{E}_{(X,Y)\sim P_j}
\left[\log Q_{\bar T_i}^{i}(Y\mid F(X))\right].
\end{aligned}
$$

The final DGER objective is:

$$
\min_{F,T,\{T_i\}}\max_{D,\{T'_i\}}
\mathcal{L}_{\mathrm{DGER}}
=\mathcal{L}_{\mathrm{cls}}
+\alpha_1\mathcal{L}_{\mathrm{adv}}
+\alpha_2\mathcal{L}_{\mathrm{er}}
+\alpha_3\mathcal{L}_{\mathrm{cel}}.
$$

### 4.8 DGER pseudocode

The following is an implementation reconstruction of Algorithm 1 in the DGER
paper. The update order and frozen-parameter semantics are important.

```text
Input: source datasets S1, ..., SK
       weights alpha_1, alpha_2, alpha_3
Initialize F, main classifier T, discriminator D
Initialize entropy classifiers T'_1, ..., T'_K
Initialize stabilizing classifiers T_1, ..., T_K

repeat until training ends:
    sample minibatches from every source domain

    # Main task and marginal alignment
    update F, T using L_cls
    update D to classify source domains from F(x)
    update F adversarially through D using L_adv

    for i = 1, ..., K:
        sample a minibatch from source domain i

        # Fit the stabilizing classifier for domain i
        freeze F
        update T_i using its own-domain term in L_cel

        # Entropy regularization game
        update T'_i to maximize its domain-i label log-likelihood
        update F to minimize L_er

        # Make T_i useful outside its own domain
        freeze T_i
        sample minibatches from domains j != i
        update F using the cross-domain term in L_cel

Return F and T
At test time: predict T(F(x)); discard D, all T_i, and all T'_i
```

### 4.9 DGER assumptions and pitfalls

- The task is supervised classification with discrete classes.
- Source domain labels are available.
- The entropy derivation assumes equal class priors. Balanced datasets or
  biased sampling are used to approximate this condition.
- The idealized result assumes sufficiently expressive classifiers and
  optimization to a global minimax solution.
- A domain discriminator aligns marginal feature distributions, but this alone
  does not establish conditional invariance.
- The many alternating updates and auxiliary networks can make training
  sensitive to learning rates and loss weights.
- The entropy objective must be implemented with the correct minimax direction;
  treating every term as an ordinary minimization loss changes the method.

## 5. Interpolation Robustness

Interpolation robustness is the main contribution of Palakkadavath et al. The
target paper views each source domain as a point on a domain manifold and
constructs intermediate domains in latent space. Unlike ordinary Mixup, it
interpolates only between same-class examples from different domains and trains
on a path of interpolation coefficients.

### 5.1 Pair construction

Sample two examples with the same class and different domains:

$$
x\sim P_d(X\mid Y=y),\qquad
x'\sim P_{d'}(X\mid Y=y),\qquad d\ne d'.
$$

Encode both:

$$
z=E_\phi(x),\qquad z'=E_\phi(x').
$$

The same label is assigned to every latent point along the interpolation path.
This is the key class-consistency condition.

### 5.2 Linear interpolation

The simplest interpolator is:

$$
\hat z(w)=z+w(z'-z),\qquad w\in[0,1].
$$

The endpoints are $z$ and $z'$. Sampling multiple values of $w$ creates a
path rather than one fixed Mixup coefficient.

### 5.3 Nonlinear interpolation

The target paper introduces a learnable transformation $T_\psi$ over the
displacement vector:

$$
\hat z(w)=I(z,z',w,T_\psi)
=z+wT_\psi(z'-z).
$$

Without an endpoint constraint, the learned curve could fail to reach $z'$ at
$w=1$. The endpoint-consistency penalty is:

$$
\mathcal{L}_{\mathrm{end}}
=\left\|T_\psi(z'-z)-(z'-z)\right\|_2^2.
$$

When $T_\psi$ is the identity, the method reduces to linear interpolation.

### 5.4 Interpolation loss

The interpolation loss combines classification of all intermediate points with
the endpoint constraint:

$$
\mathcal{L}_{\mathrm{int}}(\theta,\phi,\psi)
=\mathbb{E}_{d,d',x,x',y,w}
\left[
\ell\left(C_\theta(\hat z(w)),y\right)
+\left\|T_\psi(z'-z)-(z'-z)\right\|_2^2
\right].
$$

The complete standalone objective is:

$$
\mathcal{L}_{\mathrm{DNT}}
=\mathcal{L}_{\mathrm{cls}}
+\lambda\mathcal{L}_{\mathrm{int}}.
$$

### 5.5 DNT pseudocode

This is Algorithm 1 from the target paper expressed with explicit tensor
operations.

```text
Input: source data S, batch size B, learning rate eta, lambda
Initialize encoder E_phi, classifier C_theta, interpolator T_psi

for each training epoch:
    sample paired batches (x_i, y_i, d_i) and (x'_i, y'_i, d'_i)
    such that y_i = y'_i and d_i != d'_i

    for each pair i:
        z_i     = E_phi(x_i)
        z'_i    = E_phi(x'_i)
        delta_i = T_psi(z'_i - z_i)

        for w in the chosen interpolation grid or sampled path:
            z_hat_i(w) = z_i + w * delta_i
            assign label y_i to z_hat_i(w)

    L_cls = sum_i cross_entropy(C_theta(z_i), y_i)
    L_int = sum_i,w cross_entropy(C_theta(z_hat_i(w)), y_i)
            + squared_l2(delta_i - (z'_i - z_i))
    L_dnt = L_cls + lambda * L_int

    update theta, phi, and psi by gradient descent on L_dnt

Return E_phi and C_theta; discard T_psi at inference
```

The paper writes $w\sim\mathrm{Unif}([0,1])$, while its description also
refers to a uniformly spaced interpolation path. An implementation should
choose one policy and keep it fixed when comparing experiments. The reported
implementation uses a three-layer convolutional network for $T_\psi$.

### 5.6 Reported implementation details

The target paper and supplement report:

| Setting | Reported choice |
|---|---|
| Optimizer | SGD |
| Learning rate | (0.001) |
| Momentum | (0.9) |
| Weight decay | (0.001) |
| Minibatch size | (64) |
| Training duration | (100) epochs |
| Interpolator | Three-layer convolutional network |
| Classifier | Dense classifier preceded by ReLU |
| PACS encoder | ResNet-18 for DeepAll, DIRT, DGER; ResNet-50 for Mixup comparisons |
| VLCS encoder | AlexNet for DeepAll/DIRT; ResNet-18 for DGER; ResNet-50 for Mixup comparisons |
| RotatedMNIST encoder | Standard MNIST CNN |
| Latent dimension | 64 for RotatedMNIST; 256/512/2048 depending on dataset and backbone |

The regularization weight (\lambda) was tuned over values from (1) to
(10^{-4}). The supplement reports that larger nonzero values were generally
more useful as the data budget decreased, but the best value varied by method,
dataset, and split.

## 6. Combining interpolation robustness with invariant-representation methods

The target paper treats interpolation robustness as a meta-regularizer that can
be added to existing methods. The loss combinations are:

$$
\begin{array}{lll}
\textbf{DNT:}   & \mathcal{L}_{\mathrm{cls}}
                  +\lambda\mathcal{L}_{\mathrm{int}},\\[3pt]
\textbf{DRINT:} & \mathcal{L}_{\mathrm{cls}}
                  +\mathcal{L}_{\mathrm{dirt}}
                  +\lambda\mathcal{L}_{\mathrm{int}},\\[3pt]
\textbf{DGNT:}  & \mathcal{L}_{\mathrm{cls}}
                  +\mathcal{L}_{\mathrm{dger}}
                  +\lambda\mathcal{L}_{\mathrm{int}}.
\end{array}
$$

Here (\mathcal{L}_{\mathrm{dirt}}) and (\mathcal{L}_{\mathrm{dger}}) include
their own internal weights and auxiliary-network optimization rules. The
interpolation term is added to the representation/classifier training rather
than replacing the original method.

### 6.1 Combined training logic

```text
For every source minibatch:
    compute the base method's classification loss
    compute the base method's invariant-representation losses

    independently sample same-class, cross-domain pairs
    encode the pair and construct latent interpolation points
    compute L_int and endpoint consistency

    combine losses according to:
        DNT   : L_cls + lambda * L_int
        DRINT : L_cls + L_dirt + lambda * L_int
        DGNT  : L_cls + L_dger + lambda * L_int

    update the encoder and classifier
    update any base-method auxiliary networks using their own rules
    update T_psi using the endpoint and interpolation losses
```

For DRINT, the StarGAN generator is trained before representation training and
is frozen during the representation/interpolation updates. For DGNT, DGER's
alternating discriminator and auxiliary-classifier updates remain active.

### 6.2 Comparison table

| Method | Extra data/latent operation | Main invariance target | Extra training networks | Test-time model |
|---|---|---|---|---|
| DeepAll | None | None explicitly | None | Encoder + classifier |
| DIRT | StarGAN translation $x\to f_{d,d'}(x)$ | $g(x)\approx g(f_{d,d'}(x))$ | StarGAN generator/discriminator | Encoder + classifier |
| DGER | Adversarial feature/domain training | Marginal and conditional feature invariance | Domain discriminator, entropy classifiers, stabilizing classifiers | Feature extractor + main classifier |
| DNT | Same-class cross-domain latent paths | Robust class prediction along interpolation | Interpolator $T_\psi$ | Encoder + classifier |
| DRINT | DIRT translations plus latent paths | DIRT invariance plus interpolation robustness | DIRT networks + $T_\psi$ | Encoder + classifier |
| DGNT | DGER adversarial training plus latent paths | DGER invariance plus interpolation robustness | DGER auxiliary networks + $T_\psi$ | Feature extractor + main classifier |

## 7. Key distinctions from Mixup and Manifold Mixup

Interpolation robustness should not be described as ordinary Mixup with a
different name. The target paper emphasizes four distinctions:

1. **Pair selection:** examples have the same class and come from different
   domains.
2. **Location:** interpolation occurs in the learned latent representation,
   not necessarily in pixel space.
3. **Path:** a range of coefficients is used to expose the classifier to an
   interpolation path.
4. **Learned geometry:** the nonlinear displacement transformation $T_\psi$
   can bend the path instead of restricting it to a straight line.

Pixel-space Mixup instead combines two inputs and their labels, often with a
fixed coefficient sampled from a Beta distribution. Manifold Mixup interpolates
latent features but is not necessarily class-consistent across domains and
does not learn the endpoint-constrained nonlinear path used by DNT.

## 8. Failure modes and implementation checklist

### 8.1 Data and pairing

- Verify that each interpolation pair has the same class label.
- Verify that the two domain IDs are different.
- Handle classes present in only one source domain; such classes cannot produce
  valid cross-domain pairs without changing the sampling policy.
- Preserve class proportions when constructing limited-data subsets.
- Do not use the held-out target domain for pair construction, validation, or
  hyperparameter selection.

### 8.2 Loss and gradient handling

- Keep the task classification loss active in every method.
- For DIRT, freeze the image translator during representation training.
- For DGER, implement the discriminator and entropy-classifier min-max signs
  correctly, whether using explicit alternating updates or gradient reversal.
- In DGER, respect the frozen-$F$ and frozen-$T_i$ portions of the
  stabilizing loss.
- In DNT-family methods, include both interpolation classification and
  endpoint-consistency terms.
- Ensure the endpoint penalty uses the same displacement convention as the
  interpolator: $T_\psi(z'-z)$, not $T_\psi(z')-T_\psi(z)$.

### 8.3 Architecture and evaluation fairness

- Keep backbone, latent dimension, optimizer, data budget, and validation
  protocol matched when comparing baselines.
- Report which networks remain at inference time.
- Report whether source batches are pooled or domain-balanced.
- Tune regularization weights using source validation only.
- Repeat leave-one-domain-out experiments across all domains and random seeds.

### 8.4 Interpretation

The methods impose different notions of robustness:

- DeepAll relies on source-data diversity.
- DIRT makes representations invariant to learned image-level domain
  transformations.
- DGER seeks feature distributions whose marginal and conditional structure is
  less domain-specific.
- DNT makes class predictions robust along latent paths between same-class
  source examples from different domains.
- DRINT and DGNT combine these mechanisms; they are not replacements for the
  original DIRT or DGER objectives.

The target paper's results support interpolation robustness as a useful
complement, particularly under limited data, but they do not establish that
every interpolated latent point corresponds to a physically realizable image or
that every target domain lies on a straight or learned path between source
domains.

## 9. References

1. Palakkadavath, R., Nguyen-Tang, T., Le, H., Venkatesh, S., and Gupta, S.
   “Domain Generalization with Interpolation Robustness.” *Proceedings of the
   Asian Conference on Machine Learning*, vol. 222, pp. 1039–1054.
   [Paper](https://proceedings.mlr.press/v222/palakkadavath24a/palakkadavath24a.pdf)
   · [Supplement](https://proceedings.mlr.press/v222/palakkadavath24a/palakkadavath24a-supp.pdf)
2. Nguyen, A. T., Tran, T., Gal, Y., and Baydin, A. G. “Domain Invariant
   Representation Learning with Domain Density Transformations.” *NeurIPS
   2021*.
   [Paper](https://proceedings.neurips.cc/paper/2021/file/2a2717956118b4d223ceca17ce3865e2-Paper.pdf)
   · [Official code](https://github.com/atuannguyen/DIRT)
3. Zhao, S., Gong, M., Liu, T., Fu, H., and Tao, D. “Domain Generalization via
   Entropy Regularization.” *NeurIPS 2020*.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf)
   · [Supplement](https://proceedings.neurips.cc/paper_files/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Supplemental.pdf)
4. Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. M. “Deeper, Broader and
   Artier Domain Generalization.” *ICCV 2017*.
   [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf)
