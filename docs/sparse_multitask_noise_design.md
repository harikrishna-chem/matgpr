# Sparse Multitask Noise

This page explains shared, task-specific, and known per-observation noise in
sparse multitask GPR. These options are useful for incomplete multi-property
materials datasets where different targets, sources, or literature values have
different measurement uncertainty.

## Sparse Observation Model

The sparse multitask model currently converts every finite target value into
one scalar observation:

$$
y_{n,t} = f_t(\mathbf{x}_n) + \epsilon_{n,t}
$$

with latent covariance

$$
\operatorname{cov}[f_i(\mathbf{x}), f_j(\mathbf{x}')] =
k_x(\mathbf{x}, \mathbf{x}')\, k_{task}(i, j)
$$

The simplest setting uses one shared Gaussian observation-noise variance:

$$
\epsilon_{n,t} \sim \mathcal{N}(0, \sigma^2)
$$

This is simple and robust, but it assumes all observed target entries have the
same noise level after optional per-task standardization.

## Task-Specific Noise Model

`matgpr` can learn one observation-noise variance per task:

$$
\epsilon_{n,t} \sim \mathcal{N}(0, \sigma_t^2)
$$

For standardized targets, \(\sigma_t^2\) is learned in standardized units. In
original target units, the task-noise standard deviation is:

$$
\sigma_{t,\mathrm{original}} =
\sigma_{t,\mathrm{standardized}}\, s_t
$$

where \(s_t\) is the target standard deviation used for task \(t\).

This model should be used when target properties have different experimental
precision, different measurement protocols, or different data sources. It
should not be used as a substitute for source-specific, replicate-specific, or
input-dependent noise when that richer information is available.

## Known Per-Observation Noise

Use `noise_mode="known"` when each observed target entry has a reported or
estimated variance:

$$
\epsilon_{n,t} \sim \mathcal{N}(0, \sigma_{n,t}^2)
$$

Examples include literature tables with reported standard deviations, standard
errors, replicate-derived variances, or source-aware noise profiles. The known
variances are supplied in original target units through `known_noise_variance`.
If `standardize_y=True`, `matgpr` internally divides each observed variance by
the square of the corresponding task standard deviation before fitting.

Known per-observation noise is fixed during training; it is not a learned
heteroscedastic model. It tells the GP how much to trust each observed entry.

## Public API

Keep the current behavior as the default:

```python
model = SparseMultitaskGPRRegressor(
    noise_mode="shared",
    initial_noise=0.1,
)
```

Use task-wise noise when target properties have different measurement
precision:

```python
model = SparseMultitaskGPRRegressor(
    noise_mode="task",
    initial_task_noises={
        "yield_strength_mpa": 0.10,
        "elongation_percent": 0.15,
        "conductivity_ms_m": 0.08,
    },
)
```

Use known per-observation noise when the training data include measurement
uncertainty. A target-shaped matrix is usually the clearest format. Entries for
unobserved targets can be `NaN`; finite target entries must have positive
finite variances.

```python
noise_variance = reported_std[target_columns] ** 2
noise_variance = noise_variance.mask(y_train[target_columns].isna())

model = SparseMultitaskGPRRegressor(
    noise_mode="known",
    known_noise_variance=noise_variance,
)
model.fit(X_train, y_train[target_columns])
```

At prediction time, pass `prediction_noise_variance` when the future
measurement noise is known. If omitted, `matgpr` adds the task-wise mean known
training variance when `include_observation_noise=True`.

```python
prediction = model.predict_distribution(
    X_test,
    prediction_noise_variance=np.full((len(X_test), len(target_columns)), 0.02),
    confidence_level=0.95,
)
```

Low-level functions mirror the estimator:

```python
result = fit_sparse_multitask_gpytorch_gpr(
    X_train,
    y_train,
    task_names=target_columns,
    noise_mode="task",
    initial_task_noises=[0.10, 0.15, 0.08],
)
```

Key parameters:

- `noise_mode`: `"shared"`, `"task"`, or `"known"`.
- `initial_noise`: scalar initialization for shared noise.
- `initial_task_noises`: sequence or mapping of task-wise noise variances in
  the model training scale. With `standardize_y=True`, these are standardized
  target-unit variances.
- `known_noise_variance`: scalar, observed-entry vector, or target-shaped
  matrix of fixed known variances in original target units for
  `noise_mode="known"`.
- `prediction_noise_variance`: optional scalar, per-task vector, dense
  prediction vector, or `(n_samples, n_tasks)` matrix of future measurement
  variances in original target units.
- `noise_lower_bound`: positive lower constraint for learned noise variances.

Result objects expose:

- `noise_mode`,
- `task_noise_variance` in original target units,
- `task_noise_std` in original target units,
- `standardized_task_noise_variance` for debugging and reproducibility.
- `observation_noise_variance` and `standardized_observation_noise_variance`
  for `noise_mode="known"`.

`SparseMultitaskGPRRegressor` exposes the same learned values as fitted
attributes with trailing underscores: `noise_mode_`, `task_noise_variance_`,
`task_noise_std_`, `standardized_task_noise_variance_`,
`observation_noise_variance_`, and `standardized_observation_noise_variance_`.

## Implementation Notes

Use `gpytorch.likelihoods.HadamardGaussianLikelihood` for `noise_mode="task"`.
It learns constant task-wise noise and expects task indices to be supplied to
the likelihood call.

For task-wise noise, training uses:

```python
likelihood = gpytorch.likelihoods.HadamardGaussianLikelihood(
    num_tasks=num_tasks,
)
output = model(train_x, train_task_indices)
```

The likelihood call inside the marginal log likelihood must receive task
indices with shape `(n_observations, 1)`:

```python
task_inputs = train_task_indices.reshape(-1, 1)
loss = -mll(output, train_y_model, task_inputs)
```

Prediction uses the same task-index convention:

```python
latent = model(pred_x, pred_task_indices)
task_inputs = pred_task_indices.reshape(-1, 1)
predictive = likelihood(latent, task_inputs)
```

Use `gpytorch.likelihoods.FixedNoiseGaussianLikelihood` for
`noise_mode="known"`. Training stores the observed-entry noise vector inside
the likelihood:

```python
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise=standardized_observed_noise_variance,
)
```

Dense prediction uses either user-supplied `prediction_noise_variance` or the
task-wise mean known training variance:

```python
predictive = likelihood(latent, noise=standardized_prediction_noise_variance)
```

The existing `include_observation_noise` option should keep its meaning:

- `False`: return latent GP uncertainty.
- `True`: add learned shared/task noise or known prediction noise.

## Test Coverage

The test suite covers:

- `noise_mode="shared"` keeps current behavior and public defaults.
- `noise_mode="task"` fits and predicts dense `(n_samples, n_tasks)` outputs.
- `noise_mode="known"` fits with fixed observed-entry variances.
- Learned task-noise arrays have one value per task and positive values.
- `initial_task_noises` accepts both sequences and task-name mappings.
- `known_noise_variance` accepts target-shaped matrices and rejects missing or
  incompatible observed-entry variances.
- A wrong number of task-noise initializers raises a clear `ValueError`.
- Predictive standard deviations differ across tasks when task noises differ.
- Estimator attributes expose task-noise summaries after fitting.

## Reporting Guidance

For papers or examples using sparse noise models, report:

- whether noise is shared, task-specific, or known per observation,
- initial learned-noise values when applicable,
- how known variances were obtained when using `noise_mode="known"`,
- per-task noise standard-deviation summaries in target units,
- whether predictive intervals include observation noise,
- per-task interval coverage.

Task-wise noise can improve uncertainty calibration when target measurements
have different precision. It is not guaranteed to improve RMSE, and it should
be validated separately from the task covariance rank.

## Later Extensions

Possible later extensions are:

- task-wise plus source-wise noise,
- replicate-informed sparse multitask noise,
- input-dependent sparse multitask noise.

Those should be added as separate APIs so the statistical assumptions remain
clear.
