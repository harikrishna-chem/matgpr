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

`matgpr` now provides two exact multitask workflows.

Use `MultitaskGPRRegressor` when the multitask target matrix is complete:

- one shared feature matrix \(X\),
- one target matrix \(Y\) with shape `(n_samples, n_tasks)`,
- every task observed for every training sample,
- one learned inter-task covariance,
- per-task target standardization,
- per-task predictive means and uncertainties.

Use `SparseMultitaskGPRRegressor` when some target entries are missing:

- one shared feature matrix \(X\),
- one target matrix \(Y\) with `NaN` for unobserved task values,
- every finite target value converted to one observed `(sample, task)` pair,
- one learned input-space kernel multiplied by a learned task-index kernel,
- per-task target standardization using only observed values,
- shared or task-specific observation noise,
- dense per-task predictions for any new feature row.

The sparse form is useful for materials datasets where different properties
were measured for overlapping but incomplete sets of materials. See the
[Sparse Multitask Walkthrough](sparse_multitask_walkthrough.md) for a compact
end-to-end workflow.

## What It Does Not Cover Yet

The current sparse API supports shared or task-specific Gaussian observation
noise. It does not yet support known per-observation sparse noise,
task-specific feature matrices, explicit physics-informed task means, or
simulation-plus-experiment multi-fidelity data. Those are planned extensions
and should be implemented separately so the assumptions stay clear. See
[Sparse Multitask Noise](sparse_multitask_noise_design.md) for the current
noise API and later noise extensions.

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

## Sparse Target Matrix Example

```python
import numpy as np
import pandas as pd
from matgpr import SparseMultitaskGPRRegressor

x = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
y = pd.DataFrame({
    "property_a": np.sin(2.0 * np.pi * x).ravel(),
    "property_b": 0.5 * np.sin(2.0 * np.pi * x).ravel() + x.ravel(),
})
y.loc[[2, 7, 13], "property_a"] = np.nan
y.loc[[4, 10], "property_b"] = np.nan

model = SparseMultitaskGPRRegressor(
    training_iter=200,
    min_observations_per_task=2,
    noise_mode="task",
    initial_task_noises={"property_a": 0.05, "property_b": 0.10},
    verbose=False,
)

model.fit(x, y)
prediction = model.predict_distribution(x[:5], confidence_level=0.95)
model.task_observation_counts_
```

The sparse estimator preserves partially observed rows. Rows with all targets
missing provide no training signal. Missing feature values are still controlled
by the estimator-level `missing` policy.

For lower-level access, use `fit_sparse_multitask_gpytorch_gpr` or
`prepare_sparse_multitask_observations`.

## Choosing The Task Covariance Rank

Use `task_covar_rank=1` as a conservative default for small datasets. Increase
the rank only when there are enough observations to justify a more flexible
task-correlation structure and validation shows improvement. Sparse multitask
models need enough observed values for each task; start with at least two
observations per task and prefer substantially more for publication-quality
comparisons.

## Reporting Guidance

For a materials-informatics study, report:

- the task names and units,
- whether every task is observed for every material,
- for sparse models, the per-task observation counts,
- target scaling or transforms,
- the input descriptors and kernel,
- the task covariance rank,
- per-task validation metrics,
- uncertainty diagnostics for each task.

Use `evaluate_multitask_train_test_split` to generate these task-level tables
from any compatible complete multitask estimator:

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

For sparse targets with `NaN` entries, use
`evaluate_sparse_multitask_train_test_split`:

```python
from matgpr import (
    SparseMultitaskGPRRegressor,
    evaluate_sparse_multitask_train_test_split,
)

model = SparseMultitaskGPRRegressor(
    task_names=["property_a", "property_b"],
    training_iter=200,
    verbose=False,
)

validation = evaluate_sparse_multitask_train_test_split(
    model,
    X,
    y_sparse,
    test_size=0.2,
    random_state=7,
    model_name="sparse_multitask_gpr",
)

validation.task_metrics
validation.observed_predictions
```

Sparse `task_metrics` includes `n_observed`, `n_missing`,
`observed_fraction`, and `missing_fraction`. The full `predictions` table keeps
every sample-task prediction with an `observed` flag, while
`observed_predictions` is ready for parity plots and observed-entry metrics.

Both validation functions return one `task_metrics` row per split and task with
RMSE, MAE, R2, Pearson \(r\), sample counts, mean predictive standard
deviation, Gaussian negative log predictive density, and interval coverage when
uncertainties are available. `predictions` is a long-form table with one row
per sample and task.

Multitask GPR can improve low-data predictions when tasks are genuinely
related. It can also hurt performance when tasks are weakly related, measured
under incompatible protocols, or dominated by different noise sources.
