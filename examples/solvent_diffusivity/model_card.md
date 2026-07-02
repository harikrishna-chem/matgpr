# Model Card: Solvent Diffusivity Physics-Informed GPR

## Summary

This model card describes the single-task solvent diffusivity models in
`solvent_diffusivity_gpr_modeling.ipynb`. The notebook compares standard GPR
with physics-informed GPR mean functions for experimental polymer-solvent
diffusivity.

## Model Family

- Model type: exact Gaussian Process Regression.
- Implementation: GPyTorch through `matgpr.fit_gpytorch_gpr`.
- Kernel: Matern kernel with automatic relevance determination.
- Target: `log10_diffusivity`.
- Features: RDKit polymer and solvent features plus temperature,
  concentration, and solvent molecular weight physics features.
- Uncertainty: posterior predictive standard deviation from the GP.

## Models Compared

| Model | Mean function | Physical role |
| --- | --- | --- |
| Standard GPR | learned constant mean | data-only baseline |
| PI-GPR: Arrhenius | temperature activation trend | higher temperature can increase diffusivity |
| PI-GPR: concentration | concentration/plasticization trend | higher solvent concentration can increase diffusivity |
| PI-GPR: solvent size | molecular-size penalty | larger solvents can diffuse more slowly |
| PI-GPR: combined | Arrhenius plus concentration plus size | compact multi-term transport prior |

## Mean Functions

Arrhenius mean:

```text
log10(D) = log10(D0) - Ea / (ln(10) R T)
```

Concentration mean:

```text
log10(D) = baseline + concentration_slope * log10(weight_fraction)
```

Solvent-size mean:

```text
log10(D) = baseline - size_penalty * log10(MW_solvent)
```

Combined mean:

```text
log10(D) = log10(D0)
           - Ea / (ln(10) R T)
           + concentration_slope * log10(weight_fraction)
           - size_penalty * log10(MW_solvent)
```

The GP model is:

```text
y(x) = m_physics(x) + f_residual(x) + noise
```

`f_residual(x)` is learned by the GP covariance kernel.

## Learned Parameters

Depending on the selected physics mean, the notebook learns:

- `log_d0`: diffusivity prefactor in log10 units,
- `activation_energy_kj_mol`: positive activation energy,
- `baseline`: baseline log10 diffusivity,
- `concentration_slope`: positive concentration-effect weight,
- `size_penalty`: positive solvent-size penalty,
- Matern kernel length scales,
- GP output scale,
- Gaussian likelihood noise.

The physics parameters are optimized during GP training together with the
kernel and likelihood hyperparameters.

## Features Used By Physics

- `temperature_k`: temperature in kelvin for Arrhenius behavior.
- `log_weight_fraction`: log10 solvent weight fraction for concentration
  effects.
- `solvent_molwt`: RDKit solvent molecular weight for size effects.

The GP kernel uses the full feature matrix, including polymer fingerprints,
polymer descriptors, solvent fingerprints, solvent descriptors, and the
physics features.

## Validation Summary

The public notebook keeps this PI-GPR example because a physics-informed model
improves the 20 percent training-data RMSE over the standard baseline:

- Standard GPR RMSE at the 20 percent gate: 1.384.
- Retained PI-GPR model: concentration.
- Retained PI-GPR RMSE at the 20 percent gate: 1.333.
- RMSE advantage: 0.051 log10 diffusivity units.

The notebook also includes repeated learning curves, 90/10 validation with
10-fold cross-validation, uncertainty-aware parity plotting, and a production
fit on all filtered experimental rows.

## Intended Use

Use this model card as a reference for:

- building physics-informed GPR on polymer-solvent transport data,
- reporting which physical variables enter the mean function,
- comparing simple transport priors in a low-data setting,
- adapting `matgpr` polymer and molecule featurization to a new dataset.

## Limitations

- This is a single-task model, while the source paper motivates richer
  simulation, experiment, sorption, and physics-integrated workflows.
- The concentration prior is the retained low-data model, but the observed
  advantage is modest and should be interpreted with repeated-split statistics.
- The cyclic trimer fingerprint is a practical structure surrogate, not a full
  polymer conformational or morphology model.
- GP uncertainty is conditional on the selected descriptors, kernel, and
  likelihood; it is not a guarantee of experimental coverage.

