# Physics-Informed GPR Report for the OPV PCE Example

## Purpose

This report documents the physics-informed Gaussian Process Regression
(PI-GPR) models used in `opv_gpr_modeling.ipynb` for predicting organic
photovoltaic (OPV) power conversion efficiency (PCE). The goal is to compare a
standard data-driven GPR model against GPR models whose mean functions encode
OPV physics from the source paper:

Sahu, Rao, Troisi, and Ma, "Toward Predicting Efficiency of Organic Solar Cells
via Machine Learning and Improved Descriptors," Advanced Energy Materials,
2018.

The dataset contains 280 small-molecule OPV systems and 13 quantum-chemical
descriptors. The notebook predicts experimental `PCE` using a fixed holdout
test set and learning curves over increasing training-set sizes.

## Why Modify the GP Mean Function?

A standard GPR model usually uses a learned constant mean and relies on the
kernel to discover all structure from data. That is flexible, but inefficient
when the training set is small.

The physics-informed approach used here keeps the same GP covariance model but
replaces the constant prior mean with a mechanistic trend:

```text
PCE = physics-informed mean + GP residual
```

The mean function captures a physically motivated first-order expectation for
PCE. The GP residual then learns deviations from that expectation. This makes
the model especially useful in the low-data regime, where the kernel has not
yet seen enough examples to learn the trend on its own.

## Physical Basis from the Paper

The paper argues that OPV efficiency depends on multiple microscopic processes:

- photon absorption and exciton formation,
- exciton diffusion to the donor/acceptor interface,
- exciton dissociation into free carriers,
- charge transport,
- recombination and loss channels.

The most distinctive physical insight is that frontier orbitals beyond only
HOMO and LUMO can participate in these processes. When the donor HOMO/HOMO-1
gap, donor LUMO/LUMO+1 gap, or acceptor LUMO/LUMO+1 gap is small, nearby
orbitals can contribute to exciton formation, exciton dissociation, and charge
transport. This near-degeneracy motivates the first physics-informed mean.

The paper also motivates using exciton binding energy, conjugation length,
polarizability, and hole reorganization energy because these descriptors are
connected to charge separation and carrier transport.

## Models Compared

The notebook compares four models. They use the same GPyTorch exact GPR
training function, Matern kernel, automatic relevance determination (ARD), and
target standardization. They differ only in the GP mean function.

### Model 0: Standard GPR

**Notebook label:** `Standard GPR`

**Mean function:** learned constant mean.

**Reason for inclusion:** This is the data-driven baseline. It represents the
common GPR workflow where all structure is learned from the covariance kernel.

**Implementation:**

```python
mean_module = None
result = fit_gpytorch_gpr(..., mean_module=mean_module)
```

When `mean_module=None`, `fit_gpytorch_gpr` uses a standard constant GPyTorch
mean internally.

### Model 1: Frontier-Orbital Near-Degeneracy Mean

**Notebook label:** `PI-GPR 1: degeneracy`

**Physics included:**

- donor HOMO/HOMO-1 gap: `delHD`
- donor LUMO/LUMO+1 gap: `delLD`
- acceptor LUMO/LUMO+1 gap: `delLA`

**Physical reason:**

Small values of these gaps indicate near-degenerate frontier orbitals. The
paper connects this near-degeneracy to additional orbital participation in
exciton formation, exciton dissociation, hole transport, and ultimately higher
PCE.

**Mean form:**

```text
mean_PCE = baseline + w_degeneracy * degeneracy_score
```

where:

```text
degeneracy_score = average(-z(delHD), -z(delLD), -z(delLA))
```

Here `z(...)` means a training-set z-score. The negative sign makes the score
larger when the orbital gaps are smaller, which is physically favorable.

**Implementation details:**

- Implemented by `degeneracy_mean_equation`.
- Wrapped with `PhysicsInformedMean`.
- `baseline` and `degeneracy_weight` are learnable.
- `degeneracy_weight` is constrained positive, so the model learns how much to
  reward near-degeneracy without allowing the direction to flip.

### Model 2: Near-Degeneracy + Exciton Binding Mean

**Notebook label:** `PI-GPR 2: degeneracy + binding`

**Physics included:**

- all terms from Model 1,
- hole-electron binding energy: `E_bind`.

**Physical reason:**

Lower exciton binding energy should make charge separation easier and reduce
losses from bound electron-hole pairs. The paper identifies `E_bind` as an
important descriptor because it estimates the strength of hole-electron
interaction.

**Mean form:**

```text
mean_PCE = baseline
         + w_degeneracy * degeneracy_score
         + w_binding * exciton_separation_score
```

where:

```text
exciton_separation_score = -z(E_bind)
```

The score is larger when `E_bind` is smaller.

**Implementation details:**

- Implemented by `degeneracy_binding_mean_equation`.
- `baseline`, `degeneracy_weight`, and `binding_weight` are learnable.
- Both physics weights are positive-constrained.

### Model 3: Near-Degeneracy + Binding + Transport Mean

**Notebook label:** `PI-GPR 3: degeneracy + binding + transport`

**Physics included:**

- all terms from Model 2,
- conjugation length proxy: `N_atom`,
- donor polarizability: `polarizability`,
- hole reorganization energy: `lamda_h`.

**Physical reason:**

Charge transport and fill factor are expected to improve when the donor has a
larger conjugated path and greater polarizability, while lower hole
reorganization energy should reduce the barrier for hole transport. These
effects are not a full device-physics model, but they are chemically
interpretable proxies for transport quality.

**Mean form:**

```text
mean_PCE = baseline
         + w_degeneracy * degeneracy_score
         + w_binding * exciton_separation_score
         + w_transport * transport_score
```

where:

```text
transport_score = average(
    z(log(N_atom)),
    z(log(polarizability)),
    -z(lamda_h)
)
```

**Implementation details:**

- Implemented by `degeneracy_binding_transport_mean_equation`.
- `baseline`, `degeneracy_weight`, `binding_weight`, and
  `transport_weight` are learnable.
- All physics weights are positive-constrained.
- The use of `log(N_atom)` and `log(polarizability)` reduces the influence of
  very large molecules or high-polarizability outliers.

## How the Physics Mean Is Implemented

The implementation uses the reusable `PhysicsInformedMean` class from
`genmatics_gpr`.

Each physics equation receives two dictionaries:

```python
def physics_equation(features, parameters):
    ...
    return mean_pce
```

- `features` maps selected descriptor names to torch tensors.
- `parameters` maps fixed statistics and learnable parameters to torch tensors.

Important implementation choices:

- The notebook standardizes all GPR input features with `StandardScaler`.
- `PhysicsInformedMean` receives the scaler means and standard deviations so
  the physics equation can work in original descriptor units.
- Physics scores use training-set means and standard deviations only, avoiding
  test-set leakage.
- The GP target is standardized during training, but predictions are returned
  in original PCE units.
- The physics mean is trained jointly with GP kernel hyperparameters and noise.

This means the physics model is not hard-coded as a separate preprocessing
step. It is part of the probabilistic GP model.

## Learning-Curve Performance

The notebook was run with:

- fixed 20 percent holdout test set,
- stratified training subsets from the remaining 224 samples,
- training fractions: 0.10, 0.20, 0.40, 0.60, 1.00,
- 2 repeats per training fraction,
- 80 optimization iterations per GPyTorch fit,
- Matern ARD kernel for every model.

### Holdout RMSE

Lower RMSE is better.

| Model | 20 samples | 45 samples | 90 samples | 134 samples | 224 samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard GPR | 1.706 | 1.450 | 1.352 | 1.252 | 1.095 |
| PI-GPR 1: degeneracy | 1.593 | 1.395 | 1.291 | 1.295 | 1.169 |
| PI-GPR 2: degeneracy + binding | 1.526 | 1.365 | 1.251 | 1.227 | 1.169 |
| PI-GPR 3: degeneracy + binding + transport | 1.518 | 1.353 | 1.280 | 1.172 | 1.157 |

### Holdout Pearson Correlation

Higher Pearson `r` is better.

| Model | 20 samples | 45 samples | 90 samples | 134 samples | 224 samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard GPR | 0.245 | 0.553 | 0.590 | 0.651 | 0.755 |
| PI-GPR 1: degeneracy | 0.368 | 0.550 | 0.628 | 0.624 | 0.710 |
| PI-GPR 2: degeneracy + binding | 0.434 | 0.571 | 0.658 | 0.672 | 0.706 |
| PI-GPR 3: degeneracy + binding + transport | 0.440 | 0.581 | 0.651 | 0.716 | 0.713 |

## Performance Interpretation

The learning curves show the main value of physics-informed GPR: data
efficiency.

At the smallest training size, all physics-informed models outperform the
standard GPR baseline:

- Standard GPR RMSE: 1.706
- Best PI-GPR RMSE: 1.518
- Standard GPR Pearson `r`: 0.245
- Best PI-GPR Pearson `r`: 0.440

This is a large improvement in the regime that matters most for materials
informatics: when only a small number of measured systems are available.

At intermediate training sizes, the physics-informed models remain competitive
and often better. For example, at 134 samples, the most complex PI-GPR model
has the best RMSE and Pearson correlation:

- Standard GPR RMSE: 1.252
- PI-GPR 3 RMSE: 1.172
- Standard GPR Pearson `r`: 0.651
- PI-GPR 3 Pearson `r`: 0.716

At the full training size, the standard GPR slightly outperforms the
physics-informed variants:

- Standard GPR RMSE: 1.095
- PI-GPR 3 RMSE: 1.157
- Standard GPR Pearson `r`: 0.755
- PI-GPR 3 Pearson `r`: 0.713

This does not invalidate the physics-informed approach. Instead, it shows a
healthy behavior: when the dataset is larger, a flexible standard GP can learn
much of the trend directly from data. The physics prior is most valuable when
data are scarce or when interpretability is important.

## Learned Physics Parameters

For the full training-pool fit, the most complex model learned positive weights
for every physics term:

| Parameter | Learned value |
| --- | ---: |
| `baseline` | 5.410 |
| `degeneracy_weight` | 1.155 |
| `binding_weight` | 0.126 |
| `transport_weight` | 0.364 |

The largest learned physics contribution is the near-degeneracy term, which is
consistent with the paper's central argument. The transport term is also used,
while the binding contribution is smaller after near-degeneracy and transport
are included.

These weights should not be interpreted as causal coefficients. They are
diagnostics showing how the GP prior mean used each physically motivated score.

## Conclusion: Why Use Physics-Informed GPR?

Physics-informed GPR should be used here because it gives the model a
scientifically meaningful starting point. For OPV PCE prediction, the prior
mean can encode mechanisms that are known before fitting:

- near-degenerate frontier orbitals can improve exciton formation,
  dissociation, and charge transport,
- lower exciton binding energy can help charge separation,
- conjugation, polarizability, and hole reorganization energy can influence
  transport.

Compared with standard GPR, the physics-informed models are more data-efficient
and more interpretable. They give better learning-curve performance when the
training set is small, which is the typical situation in experimental materials
informatics. They also make the model easier to explain to domain scientists,
because part of the prediction comes from explicit OPV physics rather than only
from a black-box covariance fit.

The practical recommendation is:

- use standard GPR as a strong baseline,
- use physics-informed GPR when training data are limited,
- compare several physics-informed means of increasing complexity,
- keep the physics prior only if it improves learning curves or provides useful
  scientific interpretability.

For this OPV example, the strongest story is not that physics-informed GPR
always beats standard GPR. The stronger and more defensible conclusion is that
physics-informed GPR substantially improves the low-data regime and provides a
mechanistically interpretable model, which is exactly where materials
informatics needs help most.
