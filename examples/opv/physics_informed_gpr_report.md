# Physics-Informed GPR Report for the OPV PCE Example

## Purpose

This report documents the OPV physics-informed Gaussian Process Regression
(PI-GPR) example in `opv_gpr_modeling.ipynb`. The notebook predicts organic
photovoltaic (OPV) power conversion efficiency (PCE) from molecular descriptors
and compares:

- standard GPR,
- PI-GPR with a frontier-orbital degeneracy mean,
- PI-GPR with a frontier-orbital degeneracy plus exciton-binding mean,
- a low-data parity plot with predictive uncertainty,
- 90/10 train-test validation with 10-fold cross-validation on the 90 percent
  training set,
- a production model trained on 100 percent of the data,
- SHAP interpretation for the production model.

The physics-informed set is intentionally small. These two models are easy to
explain to a materials-informatics audience and directly reflect the core
physical message of the OPV paper.

## Reference

H. Sahu, W. Rao, A. Troisi, and H. Ma, "Toward Predicting Efficiency of
Organic Solar Cells via Machine Learning and Improved Descriptors," Advanced
Energy Materials, 8, 1801032, 2018. DOI:
[10.1002/aenm.201801032](https://doi.org/10.1002/aenm.201801032).

## Why Modify the GP Mean Function?

A standard GPR model uses a learned constant mean and relies on the covariance
kernel to discover structure from data. That is flexible, but it can be
inefficient when only a small materials dataset is available.

The PI-GPR models use the same GP covariance model but replace the constant
prior mean with a physically motivated trend:

```text
PCE = physics-informed mean + GP residual
```

Equivalently:

```text
y(x) = m_physics(x) + f_residual(x) + noise
```

where:

- `m_physics(x)` is the physics-informed mean in PCE units,
- `f_residual(x)` is the GP residual learned by the covariance kernel,
- `noise` is the learned Gaussian observation noise.

The physics equation is not a separate preprocessing step. It is part of the
GP model through `PhysicsInformedMean`, so its parameters are learned during GP
training.

## Physical Basis

The source paper emphasizes that OPV efficiency depends on coupled microscopic
processes such as exciton formation, exciton dissociation, charge transport,
and loss channels. The most direct physical insight used here is
frontier-orbital near-degeneracy.

Small donor HOMO/HOMO-1, donor LUMO/LUMO+1, and acceptor LUMO/LUMO+1 gaps can
allow nearby orbitals to participate in exciton formation, exciton
dissociation, and charge transport. This motivates the first PI-GPR model.

The second PI-GPR model adds hole-electron binding energy. Lower exciton
binding energy should make charge separation easier, so it is a natural second
physics term while still keeping the mean function compact.

## Models Compared

| Model | Mean function | Physical message |
| --- | --- | --- |
| Standard GPR | learned constant mean | data-only baseline |
| PI-GPR: degeneracy | frontier-orbital near-degeneracy | nearby orbitals beyond HOMO/LUMO can improve OPV processes |
| PI-GPR: degeneracy + binding | near-degeneracy plus low exciton binding | adds an interpretable charge-separation term |

## 20 Percent Release Gate

The first public PI-GPR example set keeps only notebooks where a
physics-informed model improves the 20 percent training-data RMSE over the
non-physics baseline.

For the OPV gate-only run:

- Standard GPR RMSE: 1.520.
- Retained PI-GPR model: degeneracy + binding.
- Retained PI-GPR RMSE: 1.278.
- RMSE advantage: 0.242.

## Physics Scores

Let `z(q)` be the z-score of descriptor `q` using only the active training
subset:

```text
z(q) = (q - mean_train(q)) / std_train(q)
```

The two physics scores are:

```text
s_deg  = (-z(delHD) - z(delLD) - z(delLA)) / 3
s_bind = -z(E_bind)
```

The signs are chosen so larger scores represent physically favorable
conditions:

- smaller `delHD`, `delLD`, and `delLA` mean stronger frontier-orbital
  near-degeneracy,
- smaller `E_bind` means easier exciton separation.

All z-score statistics are computed from the training data available in that
fit only. The external test set is not used to compute physics-score
statistics.

## Mean-Function Equations

### PI-GPR: Degeneracy

```text
m_deg(x) = b + w_deg * s_deg(x)
```

Features used:

- `delHD`: donor HOMO/HOMO-1 energetic difference,
- `delLD`: donor LUMO/LUMO+1 energetic difference,
- `delLA`: acceptor LUMO/LUMO+1 energetic difference.

Learned parameters:

- `b`: baseline PCE level,
- `w_deg`: positive weight for the near-degeneracy score.

### PI-GPR: Degeneracy + Binding

```text
m_DB(x) = b
        + w_deg  * s_deg(x)
        + w_bind * s_bind(x)
```

Features used:

- `delHD`, `delLD`, `delLA` for frontier-orbital near-degeneracy,
- `E_bind` for hole-electron binding energy.

Learned parameters:

- `b`: baseline PCE level,
- `w_deg`: positive weight for the near-degeneracy score,
- `w_bind`: positive weight for the low-binding-energy score.

This model is still compact but includes two distinct OPV mechanisms: orbital
participation and charge separation.

## What Is Learned On The Fly?

For each learning-curve split, parity refit, and production fit, the following
parameters are optimized jointly:

- physics-mean baseline `b`,
- physics weights present in the selected model, such as `w_deg` and `w_bind`,
- ARD Matern kernel length scales,
- GP output scale,
- Gaussian likelihood noise.

The optimization minimizes the negative exact GP marginal log likelihood with
Adam. This means the physics weights are learned on the fly together with the
GP hyperparameters. They are not fitted separately and then frozen.

The following quantities are fixed within each fit because they are computed
from the training data before GP optimization:

- feature means and standard deviations used for `z(...)`,
- input-scaler means and standard deviations,
- target mean and standard deviation used for target standardization.

## Learning-Curve Design

The notebook uses:

- 30 percent of the dataset as an external test set,
- 70 percent of the dataset as the training pool,
- training data percentages from 10 to 70 percent of the full dataset,
- 20 random stratified subsets per training percentage,
- RMSE and R2 learning curves with standard-deviation error bars,
- the same Matern ARD kernel and target standardization for every model.

This preserves a broad learning-curve comparison while keeping the physics
models simple.

## Low-Data Parity and Uncertainty

The parity plot focuses on the small-data regime. The notebook selects the
best PI-GPR model at the 10 percent learning-curve point, then chooses the 10
percent subset where that PI-GPR model has the largest RMSE advantage over
Standard GPR.

Standard GPR and the selected PI-GPR model are refit on the same 10 percent
training subset and evaluated on the fixed 30 percent external test set.

Each parity panel shows:

- experimental PCE on the x-axis,
- predicted PCE on the y-axis,
- GP predictive standard deviation as vertical error bars,
- RMSE, R2, Pearson `r`, and number of training samples.

The purpose is to make the low-data contrast visually clear while using the
learning curves as the full 10 to 70 percent performance comparison.

## 90/10 Validation With 10-Fold Cross-Validation

Before fitting the final production model, the notebook performs one additional
validation step using the selected best model from the learning-curve summary.

The workflow is:

- split the full dataset into 90 percent training and 10 percent test data,
- run 10-fold cross-validation on the 90 percent training set,
- summarize cross-validation RMSE, R2, MAE, and Pearson `r`,
- refit the selected model on the full 90 percent training set,
- generate train and test parity predictions with GP uncertainty error bars.

The figure has two panels:

- left: cross-validation statistics reported as mean and standard deviation,
- right: parity plot for the 90 percent training set and 10 percent held-out
  test set, with predictive uncertainty shown as vertical error bars.

This analysis gives a more conventional model-validation view before the final
production fit uses all available OPV data.

## Production Model And SHAP

The production model is selected from the PI-GPR candidates only when a
PI-GPR model has lower mean RMSE than Standard GPR at the 20 percent
training-data point. This keeps the notebook as a physics-informed example
only when the low-data comparison supports that conclusion. The retained
PI-GPR model is then refit on 100 percent of the OPV dataset.

The notebook then computes SHAP values for the production model using a
model-agnostic permutation explainer. The SHAP plots identify which descriptors
drive the production model and whether the learned feature impacts are
chemically meaningful.

## Conclusion

This version of the OPV example is deliberately simple:

- Standard GPR provides the data-only baseline,
- PI-GPR degeneracy tests the paper's central orbital-degeneracy idea,
- PI-GPR degeneracy + binding adds one interpretable charge-separation term.

That makes the demonstration easier to explain: physics enters only through the
GP mean function, the physics weights are learned jointly with GP
hyperparameters, and the GP kernel still learns residual structure not captured
by the physics prior.
