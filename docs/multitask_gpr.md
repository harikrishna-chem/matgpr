# Multitask GPR

Multitask Gaussian Process Regression models multiple related materials
properties together. Instead of fitting one independent GP for each target, the
model learns both an input-space covariance and a task covariance:

$$
\operatorname{cov}[f_i(\mathbf{x}), f_j(\mathbf{x}')] =
k_x(\mathbf{x}, \mathbf{x}')\, k_{task}(i, j)
$$

where \(i\) and \(j\) index target properties such as efficiency, stability,
band gap, conductivity, hardness, or diffusivity. This is useful when tasks
share trends because data from one property can help stabilize another property.

## Current Scope

The first `matgpr` multitask implementation supports complete multitask
training data:

- one shared feature matrix \(X\),
- one target matrix \(Y\) with shape `(n_samples, n_tasks)`,
- every task observed for every training sample,
- one learned inter-task covariance,
- per-task target standardization,
- per-task predictive means and uncertainties.

This is the right starting point for published materials datasets where the
same material candidates have multiple measured or computed properties.

## What It Does Not Cover Yet

The initial API does not yet support sparse task observations, task-specific
feature matrices, explicit physics-informed task means, or simulation-plus-
experiment multi-fidelity data. Those are planned extensions and should be
implemented separately so the assumptions stay clear.

## Minimal Example

```python
import numpy as np
from matgpr import MultitaskGPRRegressor

x = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
y = np.column_stack([
    np.sin(2.0 * np.pi * x).ravel(),
    0.5 * np.sin(2.0 * np.pi * x).ravel() + x.ravel(),
])

model = MultitaskGPRRegressor(
    task_names=["property_a", "property_b"],
    training_iter=200,
    verbose=False,
)

model.fit(x, y)
prediction = model.predict_distribution(x[:5], confidence_level=0.95)
prediction.mean.shape
```

The prediction arrays have shape `(n_samples, n_tasks)`, and
`prediction.task_names` records the task-column order.

For lower-level GPyTorch access, use `fit_multitask_gpytorch_gpr`, which
returns a `MultitaskGPyTorchResult` containing the fitted model, likelihood,
loss history, and task metadata.

## Choosing The Task Covariance Rank

Use `task_covar_rank=1` as a conservative default for small datasets. Increase
the rank only when there are enough observations to justify a more flexible
task-correlation structure and validation shows improvement.

## Reporting Guidance

For a materials-informatics study, report:

- the task names and units,
- whether every task is observed for every material,
- target scaling or transforms,
- the input descriptors and kernel,
- the task covariance rank,
- per-task validation metrics,
- uncertainty diagnostics for each task.

Use `evaluate_multitask_train_test_split` to generate these task-level tables
from any compatible multitask estimator:

```python
from matgpr import MultitaskGPRRegressor, evaluate_multitask_train_test_split

model = MultitaskGPRRegressor(
    task_names=["property_a", "property_b"],
    training_iter=200,
    verbose=False,
)

validation = evaluate_multitask_train_test_split(
    model,
    X,
    y,
    test_size=0.2,
    random_state=7,
    model_name="multitask_gpr",
)

validation.task_metrics
validation.predictions
```

`task_metrics` contains one row per split and task with RMSE, MAE, R2,
Pearson \(r\), sample count, mean predictive standard deviation, Gaussian
negative log predictive density, and interval coverage when uncertainties are
available. `predictions` is a long-form parity-plot table with one row per
sample and task.

Multitask GPR can improve low-data predictions when tasks are genuinely
related. It can also hurt performance when tasks are weakly related, measured
under incompatible protocols, or dominated by different noise sources.
