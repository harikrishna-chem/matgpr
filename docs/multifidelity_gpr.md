# Multi-Fidelity GPR

Multi-fidelity GPR combines inexpensive low-fidelity data with scarce
high-fidelity measurements. In materials informatics, common fidelity pairs are
simulation plus experiment, low-accuracy plus high-accuracy calculation, or
screening measurement plus validated measurement.

The first `matgpr` multi-fidelity model is a delta model:

$$
y_H(\mathbf{x}) = \rho y_L(\mathbf{x}) + b + \delta(\mathbf{x}) + \epsilon_H
$$

where \(y_H\) is the high-fidelity target, \(y_L\) is the low-fidelity value,
\(\rho\) and \(b\) map low fidelity onto high fidelity, and
\(\delta(\mathbf{x})\) is a GPR correction learned from high-fidelity data.

## When To Use It

Use delta multi-fidelity GPR when:

- low-fidelity data are cheaper or more abundant than high-fidelity data,
- low-fidelity values are correlated with high-fidelity values,
- the low-fidelity bias is smooth enough to learn as a correction,
- the final decision should target the high-fidelity property.

Typical materials examples:

- DFT or molecular simulation plus experimental property data,
- approximate descriptors or fast simulations plus validated measurements,
- low-throughput high-accuracy measurements plus high-throughput proxy data.

Avoid this model when the low-fidelity source is uncorrelated with the
high-fidelity property or when high-fidelity data cover a completely different
materials domain.

## Public API

Use externally available low-fidelity values at high-fidelity training and
prediction points:

```python
from matgpr import MultiFidelityGPRRegressor

model = MultiFidelityGPRRegressor(
    correction_kernel="matern",
    training_iter=1000,
    random_state=7,
)

model.fit(
    X_high,
    y_high,
    low_fidelity=simulation_at_high_points,
)

prediction = model.predict_distribution(
    X_test,
    low_fidelity=simulation_at_test_points,
    confidence_level=0.95,
)
```

If low-fidelity values are not available at all high-fidelity or prediction
points, fit an internal low-fidelity surrogate:

```python
model.fit(
    X_high,
    y_high,
    X_low=X_simulation,
    y_low=y_simulation,
)

prediction = model.predict_distribution(X_test, confidence_level=0.95)
```

The prediction object exposes:

- `mean`, `std`, `lower`, `upper`: high-fidelity prediction,
- `low_fidelity_mean`, `low_fidelity_std`: low-fidelity component,
- `correction_mean`, `correction_std`: learned high-fidelity correction,
- `rho`, `intercept`: fitted linear fidelity mapping.

## Observation Data Preparation

For joint co-kriging and multi-level fidelity workflows, use
`prepare_multifidelity_observations` to validate row-wise fidelity data before
modeling. The first co-kriging implementation supports exactly two ordered
fidelity levels:

```python
from matgpr import prepare_multifidelity_observations


observations = prepare_multifidelity_observations(
    X=X_all,
    y=y_all,
    fidelity=fidelity_labels,
    fidelity_order=("simulation", "experiment"),
    target_fidelity="experiment",
    sample_id=material_ids,
    noise_variance=known_noise_variance,
)
```

The returned `MultiFidelityObservationData` stores:

- numeric `X` and `y` rows,
- integer `fidelity_index` values aligned with `fidelity_names`,
- the target fidelity and per-fidelity observation counts,
- optional sample identifiers, known noise variances, and feature names,
- helper row selections such as `rows_for_fidelity("experiment")`.

This preparation layer supports non-nested datasets where high-fidelity
materials are not necessarily a subset of low-fidelity materials. The delta
model still uses `X_high`, `y_high`, and low-fidelity arrays directly; the
observation container is the input contract for joint co-kriging models.

## Two-Level Co-Kriging

Use `CoKrigingGPRRegressor` when low- and high-fidelity observations should be
fit jointly instead of using a two-stage correction:

```python
from matgpr import CoKrigingGPRRegressor


model = CoKrigingGPRRegressor(
    fidelity_order=("simulation", "experiment"),
    target_fidelity="experiment",
    low_fidelity_kernel="matern",
    discrepancy_kernel="matern",
    training_iter=1000,
    random_state=7,
)

model.fit(X_all, y_all, fidelity=fidelity_labels)

prediction = model.predict_distribution(
    X_test,
    confidence_level=0.95,
)
```

The current co-kriging model learns:

$$
f_H(\mathbf{x}) = \rho f_L(\mathbf{x}) + \delta(\mathbf{x})
$$

where \(f_L\) is the low-fidelity latent GP, \(\delta\) is an independent
discrepancy GP, and \(\rho\) is a learned scalar autoregressive coefficient.
The covariance blocks are fit jointly across all low- and high-fidelity
observations.

Current limits:

- exactly two fidelity levels,
- one shared learned Gaussian observation-noise term,
- no known-noise or per-fidelity noise mode yet,
- prediction reports one fidelity at a time; component summaries are planned
  next.

## High-Fidelity Learning Curves

Use `multifidelity_learning_curve` to test whether low-fidelity information
helps in the low-data high-fidelity regime. The train-size axis always varies
the number of high-fidelity samples. Low-fidelity inputs are either supplied at
the same high-fidelity rows or held fixed as a separate low-fidelity dataset
used to fit a surrogate inside each split.

```python
from matgpr import MultiFidelityGPRRegressor, multifidelity_learning_curve


model = MultiFidelityGPRRegressor(training_iter=500, random_state=7)

lc_result = multifidelity_learning_curve(
    {"delta multi-fidelity GPR": model},
    X_high,
    y_high,
    low_fidelity_high=simulation_at_high_points,
    train_size_start=10,
    train_size_stop=100,
    train_size_step=10,
    train_size_unit="percent",
    n_splits=20,
    test_size=0.30,
    random_state=42,
    metrics=("RMSE", "R2", "MAE", "r"),
    metric_splits="test",
    store_predictions=True,
)

run_metrics = lc_result.runs
summary = lc_result.summary(metrics=("RMSE", "R2"), splits="test")
component_predictions = lc_result.predictions
```

For an internal low-fidelity surrogate, replace `low_fidelity_high` with
`X_low=X_simulation` and `y_low=y_simulation`. The returned run table includes
the fitted `rho` and `intercept` values for each split. When predictions are
stored, the table includes component columns such as `low_fidelity_pred`,
`correction_pred`, and their uncertainties when the estimator exposes them.

## Component Reporting

Use component reports to explain whether the multi-fidelity model is mostly
using the scaled low-fidelity source or making a large high-fidelity correction:

```python
from matgpr import decompose_multifidelity_prediction, summarize_multifidelity_components


prediction = model.predict_distribution(
    X_test,
    low_fidelity=simulation_at_test_points,
    confidence_level=0.95,
)

component_rows = decompose_multifidelity_prediction(
    prediction,
    y_true=y_test,
    sample_labels=test_sample_ids,
    model_name="delta multi-fidelity GPR",
    split="test",
)
component_summary = summarize_multifidelity_components(component_rows)
```

The per-sample report includes:

- `scaled_low_fidelity_pred`: \(\rho y_L(\mathbf{x})\),
- `intercept`: fitted fidelity-map offset,
- `correction_pred`: \(\delta(\mathbf{x})\),
- `reconstructed_y_pred`: sum of the reported components,
- `component_residual`: difference between `y_pred` and the reconstructed sum,
- variance fractions from the scaled low-fidelity and correction uncertainties
  when standard deviations are available.

For learning-curve outputs, pass `lc_result.predictions` directly:

```python
component_summary = summarize_multifidelity_components(
    lc_result.predictions,
    group_by=("model", "split", "train_size_percent"),
)
```

## Low-Fidelity Uncertainty

When an internal low-fidelity surrogate is used, total uncertainty can include
both the correction uncertainty and propagated low-fidelity surrogate
uncertainty:

$$
\sigma_H^2(\mathbf{x}) =
\sigma_\delta^2(\mathbf{x}) + \rho^2 \sigma_L^2(\mathbf{x})
$$

This behavior is controlled by `include_low_fidelity_uncertainty`. If
low-fidelity values are supplied directly at prediction time, `matgpr` treats
them as known inputs and only reports correction uncertainty unless future
extensions add user-supplied low-fidelity uncertainty.

## Validation Protocol

Validate on held-out high-fidelity data. Useful comparisons are:

- high-fidelity-only standard GPR,
- low-fidelity-only baseline,
- delta multi-fidelity GPR with supplied low-fidelity values,
- delta multi-fidelity GPR with an internal low-fidelity surrogate.
- two-level co-kriging GPR when low- and high-fidelity rows should be fit
  jointly.

Report RMSE, MAE, R2, uncertainty coverage, and whether uncertainty includes
low-fidelity surrogate uncertainty. Learning curves should vary the number of
high-fidelity training points while keeping the low-fidelity source fixed.
Use `multifidelity_learning_curve` for this repeated-split protocol when using
`matgpr` estimators.

## Current Scope

Implemented:

- two-stage delta multi-fidelity GPR,
- learned \(\rho\) and intercept from high-fidelity training pairs,
- GPR correction model for \(\delta(\mathbf{x})\),
- optional internal low-fidelity GPR surrogate,
- multi-fidelity observation data preparation for ordered fidelity datasets,
- two-level autoregressive co-kriging GPR with learned constant \(\rho\),
  shared learned noise, and target-fidelity prediction,
- estimator API and lower-level function,
- component-wise prediction output,
- high-fidelity learning-curve validation helper with component predictions,
- reporting helpers for low-fidelity and correction contributions.

Planned later:

- full co-kriging with joint covariance across fidelities,
- more than two fidelity levels,
- fidelity/source-specific known noise,
- Bayesian optimization acquisition functions targeting high-fidelity outcomes.

See [Co-Kriging And Multi-Level Fidelity Design](multifidelity_design.md) for
the planned joint co-kriging and multi-level fidelity API.
