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

Report RMSE, MAE, R2, uncertainty coverage, and whether uncertainty includes
low-fidelity surrogate uncertainty. Learning curves should vary the number of
high-fidelity training points while keeping the low-fidelity source fixed.

## Current Scope

Implemented:

- two-stage delta multi-fidelity GPR,
- learned \(\rho\) and intercept from high-fidelity training pairs,
- GPR correction model for \(\delta(\mathbf{x})\),
- optional internal low-fidelity GPR surrogate,
- estimator API and lower-level function,
- component-wise prediction output.

Planned later:

- full co-kriging with joint covariance across fidelities,
- more than two fidelity levels,
- fidelity/source-specific known noise,
- validation helpers specialized for low-data high-fidelity learning curves,
- Bayesian optimization acquisition functions targeting high-fidelity outcomes.
