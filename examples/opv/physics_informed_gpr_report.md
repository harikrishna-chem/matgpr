# Physics-Informed GPR Report for the OPV PCE Example

## Purpose

This report documents the OPV physics-informed Gaussian Process Regression
(PI-GPR) example in `opv_gpr_modeling.ipynb`. The notebook predicts organic
photovoltaic (OPV) power conversion efficiency (PCE) from 13 molecular
descriptors and compares:

- one standard GPR baseline,
- three selected physics-informed GPR models,
- repeated low-data 20 percent training / 80 percent test splits,
- uncertainty-aware external-test parity plots from the fixed 20/80 split,
- a production model trained on 100 percent of the data,
- SHAP interpretation for the production model.

## Reference

H. Sahu, W. Rao, A. Troisi, and H. Ma, "Toward Predicting Efficiency of
Organic Solar Cells via Machine Learning and Improved Descriptors," Advanced
Energy Materials, 8, 1801032, 2018. DOI:
[10.1002/aenm.201801032](https://doi.org/10.1002/aenm.201801032).

## Why Modify the GP Mean Function?

A standard GPR model uses a learned constant mean and relies on the covariance
kernel to discover structure from data. That is flexible, but it can be
inefficient for small materials datasets.

The PI-GPR models use the same GP covariance model but replace the constant
prior mean with a physically motivated trend:

```text
PCE = physics-informed mean + GP residual
```

The mean function gives the model a chemically meaningful starting point. The
kernel then learns residual deviations from that physics prior.

## Physical Basis

The source paper emphasizes that OPV efficiency depends on coupled microscopic
processes:

- photon absorption and exciton formation,
- exciton diffusion to the donor/acceptor interface,
- exciton dissociation into free carriers,
- charge transport,
- recombination and energetic loss channels.

The most direct physics prior is frontier-orbital near-degeneracy. Small donor
HOMO/HOMO-1, donor LUMO/LUMO+1, and acceptor LUMO/LUMO+1 gaps can allow nearby
orbitals to participate in exciton formation, exciton dissociation, and charge
transport. The notebook also tests low exciton binding energy, conjugation,
polarizability, hole reorganization energy, optical gap, donor-acceptor energy
alignment, and excited-state descriptors.

## Model Selection

An updated low-data screening run evaluated candidate PI-GPR means over 20
random stratified splits, each with 20 percent of the dataset used for training
and 80 percent held out for testing. The three PI-GPR models retained for the
final notebook were selected by low mean RMSE with small RMSE standard
deviation.

| Retained PI-GPR model | 20 percent RMSE mean | 20 percent RMSE std |
| --- | ---: | ---: |
| PI-GPR: degeneracy + binding | 1.3239 | 0.0646 |
| PI-GPR: degeneracy + binding + transport | 1.3298 | 0.0758 |
| PI-GPR: degeneracy | 1.3361 | 0.0634 |

For reference, the standard GPR baseline had mean RMSE 1.5229 with standard
deviation 0.0732 under the same repeated 20/80 protocol. This supports the
intended low-data demonstration: physically informed mean functions provide a
better starting point when only a small training set is available.

The final comparison therefore keeps four models:

| Model | Mean function |
| --- | --- |
| Standard GPR | learned constant mean |
| PI-GPR: degeneracy + binding | compact orbital and binding prior |
| PI-GPR: degeneracy + binding + transport | orbital near-degeneracy, low binding energy, and transport proxies |
| PI-GPR: degeneracy | orbital near-degeneracy prior |

## How Physics Enters the Mean Function

For a standard GPR model, the prior mean is only a learned constant:

```text
m_standard(x) = c
```

The physics-informed models replace this constant prior with a descriptor-based
mean function:

```text
y(x) = m_physics(x) + f_residual(x) + noise
```

where:

- `m_physics(x)` is the physics-informed mean in PCE units,
- `f_residual(x)` is the GP residual learned by the covariance kernel,
- `noise` is the learned Gaussian observation noise.

The GP does not use the physics equation as a separate preprocessing step. The
equation is part of the GP model through `PhysicsInformedMean`, so the physics
parameters are optimized during GP training.

## Physics Scores

The retained PI-GPR models are built from standardized physics scores. Let
`z(q)` be the z-score of descriptor `q` using only the active training subset:

```text
z(q) = (q - mean_train(q)) / std_train(q)
```

The physics scores are:

```text
s_deg   = (-z(delHD) - z(delLD) - z(delLA)) / 3
s_bind  = -z(E_bind)
s_trans = (z(log(N_atom)) + z(log(polarizability)) - z(lamda_h)) / 3
```

The signs are chosen so that larger scores represent physically favorable
conditions: smaller frontier-orbital gaps, lower exciton binding energy, larger
conjugation and polarizability, and lower hole reorganization energy.

## Retained Mean-Function Equations

The three retained PI-GPR models use the following prior means.

### PI-GPR: Degeneracy + Binding

```text
m_DB(x) = b
        + w_deg  * s_deg(x)
        + w_bind * s_bind(x)
```

Features used:

- `delHD`, `delLD`, `delLA` for frontier-orbital near-degeneracy,
- `E_bind` for exciton separation.

This is the most compact retained physics prior. It keeps the paper's central
orbital-degeneracy idea and one charge-separation descriptor while avoiding
extra terms.

### PI-GPR: Degeneracy + Binding + Transport

```text
m_DBT(x) = b
         + w_deg   * s_deg(x)
         + w_bind  * s_bind(x)
         + w_trans * s_trans(x)
```

Features used:

- `delHD`, `delLD`, `delLA` for frontier-orbital near-degeneracy,
- `E_bind` for exciton separation,
- `N_atom`, `polarizability`, `lamda_h` for charge-transport proxies.

### PI-GPR: Degeneracy

```text
m_deg(x) = b
         + w_deg * s_deg(x)
```

Features used:

- `delHD`, `delLD`, `delLA` for frontier-orbital near-degeneracy.

This model isolates the source paper's central physical argument. It asks
whether orbital near-degeneracy alone is enough to outperform a standard GPR
mean in the low-data regime.

## Learned and Fixed Parameters

The physics-informed mean has both learned parameters and fixed training-set
statistics.

Learned during each GP fit:

- `b`: the baseline PCE level in the physics mean,
- `w_deg`: strength of the near-degeneracy score,
- `w_bind`: strength of the low-binding-energy score,
- `w_trans`: strength of the transport score.

Only the weights present in a given model are learned. For example, `m_deg`
learns `b` and `w_deg`, while `m_DBT` learns `b`, `w_deg`, `w_bind`, and
`w_trans`.

The physics weights are constrained positive in the implementation. This keeps
the sign of each physics score aligned with the intended OPV mechanism. The
baseline `b` is unconstrained.

Learned at the same time as the physics parameters:

- ARD Matern kernel length scales,
- GP output scale,
- Gaussian likelihood noise.

These parameters are learned on the fly for every learning-curve split, every
20 percent training-pool fit, and the final 100 percent production fit. They
are optimized jointly by minimizing the negative exact GP marginal log
likelihood with Adam. The physics equation is therefore not fitted first and
then frozen; it is part of the probabilistic model.

Fixed within each fit:

- feature means and standard deviations used for each `z(...)` score,
- `log(N_atom)` and `log(polarizability)` means and standard deviations,
- input-scaler means and standard deviations used before GP fitting,
- target mean and standard deviation used for target standardization.

These fixed quantities are computed only from the training data available in
that fit. The external test set is never used to compute physics-score
statistics or scaling parameters.

## Learning-Curve Design

The notebook now uses:

- 20 percent of the dataset as the maximum training pool,
- 80 percent of the dataset as the external test set,
- training data percentages of 10 and 20 percent of the full dataset,
- 20 random stratified 20/80 splits per training percentage,
- RMSE and R2 learning curves with standard-deviation error bars,
- Matern ARD kernels and target standardization for every model.

At the 10 percent point, the model trains on half of each 20 percent training
split. At the 20 percent point, it trains on the full low-data training split.
This design intentionally emphasizes the regime where physics-informed priors
should be most useful.

## External-Test Parity and Uncertainty

After the learning curve, the four retained models are trained on the fixed 20
percent training split and evaluated on the fixed 80 percent test set. The
notebook creates one publication-style 2 by 2 parity figure:

- experimental PCE on the x-axis,
- predicted PCE on the y-axis,
- GP predictive standard deviation as vertical error bars,
- RMSE, R2, and Pearson `r` annotated on each subplot.

The purpose is to compare both accuracy and uncertainty quality across the
standard and physics-informed models.

## Production Model

The notebook selects the best model from the retained four-model comparison
using the 20 percent training-data RMSE summary. That selected model is then
refit on 100 percent of the OPV dataset.

If the selected model is physics-informed, the production model keeps an
interpretable prior mean. If standard GPR wins, that is useful evidence that
the current physics priors should be revised or expanded before claiming a
physics-informed advantage.

## SHAP Analysis

The notebook computes SHAP values for the production model using a
model-agnostic permutation explainer. The SHAP section provides:

- a top-feature table,
- a mean absolute SHAP bar plot,
- dependence-style plots for the strongest features,
- a SHAP summary plot showing feature value and prediction impact.

The goal is to identify which descriptors drive the production model and
whether their impacts agree with OPV intuition.

## Conclusion

This OPV example is designed to support a defensible materials-informatics
story:

- compare standard GPR against physics-informed GPR under the same kernel,
- select PI-GPR candidates from repeated low-data RMSE results,
- evaluate uncertainty on a true external test set,
- train the final production model on all available data,
- use SHAP to explain the final model.

The strongest claim should come from the learning curves and parity plots after
the full notebook is run. If a PI-GPR model wins with only 10 to 20 percent of
the data while maintaining reasonable uncertainty, it provides a clear example
of why physics-informed GPR is valuable for small experimental materials
datasets.
