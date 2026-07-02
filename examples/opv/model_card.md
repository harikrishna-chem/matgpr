# Model Card: OPV Physics-Informed GPR

## Summary

This model card describes the OPV models in `opv_gpr_modeling.ipynb`. The
notebook compares a standard Gaussian Process Regression baseline against two
physics-informed GPR models that modify the GP mean function.

## Model Family

- Model type: exact Gaussian Process Regression.
- Implementation: GPyTorch through `matgpr.fit_gpytorch_gpr`.
- Kernel: Matern kernel with automatic relevance determination.
- Target handling: internal target standardization during GP optimization.
- Uncertainty: posterior predictive standard deviation from the GP.

## Models Compared

| Model | Mean function | Physical role |
| --- | --- | --- |
| Standard GPR | learned constant mean | data-only baseline |
| PI-GPR: degeneracy | frontier-orbital near-degeneracy | rewards small donor and acceptor orbital gaps |
| PI-GPR: degeneracy + binding | near-degeneracy plus low binding energy | combines orbital participation and easier charge separation |

## Physics Scores

For a descriptor `q`, the notebook computes a training-set z-score:

```text
z(q) = (q - mean_train(q)) / std_train(q)
```

The physics scores are:

```text
s_deg  = (-z(delHD) - z(delLD) - z(delLA)) / 3
s_bind = -z(E_bind)
```

Larger scores are designed to represent more favorable physical conditions.
All z-score statistics are fit on the active training data only.

## Mean Functions

The degeneracy model uses:

```text
m_deg(x) = b + w_deg * s_deg(x)
```

The degeneracy plus binding model uses:

```text
m_DB(x) = b + w_deg * s_deg(x) + w_bind * s_bind(x)
```

The GP model is:

```text
y(x) = m_physics(x) + f_residual(x) + noise
```

`f_residual(x)` is learned by the GP covariance kernel.

## Learned Parameters

The physics-informed models learn:

- `b`: baseline PCE level,
- `w_deg`: positive degeneracy-score weight,
- `w_bind`: positive binding-score weight when the binding term is present,
- Matern kernel length scales,
- GP output scale,
- Gaussian likelihood noise.

The physics weights are learned during GP training together with the kernel and
likelihood hyperparameters. They are not fit in a separate preprocessing step.

## Features Used By Physics

- `delHD`: donor HOMO/HOMO-1 energetic difference.
- `delLD`: donor LUMO/LUMO+1 energetic difference.
- `delLA`: acceptor LUMO/LUMO+1 energetic difference.
- `E_bind`: hole-electron binding energy.

The GP kernel uses the full OPV descriptor matrix, so the physics mean sets a
prior trend while the residual GP can still learn descriptor effects not
captured by the simple physics equation.

## Validation Summary

The public notebook keeps the OPV PI-GPR example because a physics-informed
model improves the 20 percent training-data RMSE over the standard baseline:

- Standard GPR RMSE at the 20 percent gate: 1.520.
- Retained PI-GPR model: degeneracy + binding.
- Retained PI-GPR RMSE at the 20 percent gate: 1.278.
- RMSE advantage: 0.242 PCE percentage points.

The notebook also includes learning curves, low-data parity plots with
uncertainty, 90/10 validation with 10-fold cross-validation, a production fit
on all data, and SHAP interpretation for the selected production model.

## Intended Use

Use this model card as a reference for:

- building a compact PI-GPR mean function,
- reporting learned physics parameters,
- comparing low-data standard GPR and PI-GPR behavior,
- adapting the OPV workflow to another molecular materials problem.

## Limitations

- The physics mean is intentionally simple and does not represent a full OPV
  device model.
- The reported advantage is tied to the notebook's split protocol, random
  seeds, descriptors, and training settings.
- GP uncertainty is model uncertainty under the chosen kernel and likelihood,
  not a guarantee of experimental coverage.
- The model should not be used for high-stakes experimental decisions without
  additional validation in the intended chemical and device domain.

