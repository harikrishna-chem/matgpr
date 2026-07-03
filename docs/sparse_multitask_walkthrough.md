# Sparse Multitask Walkthrough

This walkthrough shows how to model an incomplete multi-property materials
dataset with sparse multitask GPR. Use this workflow when different properties
were measured for overlapping but not identical sets of materials.

The key convention is simple:

- rows are materials, processing conditions, or experiments,
- columns in `X` are shared descriptors,
- columns in `Y` are target properties,
- `NaN` in `Y` means "this property was not observed for this row".

Do not use `NaN` for failed experiments, censored measurements, or values that
are known to be outside a detection limit unless you have decided how those
observations should enter the statistical model.

## 1. Build A Sparse Multi-Property Dataset

This synthetic example uses alloy-like descriptors and three related
properties. In a real study, replace this section with your experimental,
literature, or simulation dataframe.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(18)
n_samples = 72

ni_at_percent = rng.uniform(5.0, 35.0, n_samples)
cr_at_percent = rng.uniform(8.0, 28.0, n_samples)
anneal_temperature_c = rng.uniform(650.0, 1050.0, n_samples)
grain_size_um = rng.lognormal(mean=2.0, sigma=0.35, size=n_samples)

strength_mpa = (
    260.0
    + 7.0 * ni_at_percent
    + 4.5 * cr_at_percent
    - 0.18 * anneal_temperature_c
    + 95.0 / np.sqrt(grain_size_um)
    + rng.normal(0.0, 18.0, n_samples)
)
elongation_percent = (
    42.0
    - 0.045 * strength_mpa
    + 0.015 * anneal_temperature_c
    - 0.35 * cr_at_percent
    + rng.normal(0.0, 1.8, n_samples)
)
conductivity_ms_m = (
    19.0
    - 0.12 * ni_at_percent
    - 0.10 * cr_at_percent
    + 0.003 * anneal_temperature_c
    + rng.normal(0.0, 0.7, n_samples)
)

data = pd.DataFrame(
    {
        "ni_at_percent": ni_at_percent,
        "cr_at_percent": cr_at_percent,
        "anneal_temperature_c": anneal_temperature_c,
        "grain_size_um": grain_size_um,
        "yield_strength_mpa": strength_mpa,
        "elongation_percent": elongation_percent,
        "conductivity_ms_m": conductivity_ms_m,
    }
)
```

Now mask a fraction of target entries to mimic a real sparse property matrix.
The feature columns stay complete; only target values are missing.

```python
target_columns = [
    "yield_strength_mpa",
    "elongation_percent",
    "conductivity_ms_m",
]
feature_columns = [
    "ni_at_percent",
    "cr_at_percent",
    "anneal_temperature_c",
    "grain_size_um",
]

sparse_targets = data[target_columns].copy()
sparse_targets.loc[rng.random(n_samples) < 0.25, "yield_strength_mpa"] = np.nan
sparse_targets.loc[rng.random(n_samples) < 0.35, "elongation_percent"] = np.nan
sparse_targets.loc[rng.random(n_samples) < 0.40, "conductivity_ms_m"] = np.nan

observed_counts = sparse_targets.notna().sum()
observed_counts
```

For real datasets, report these counts next to every sparse multitask model.
They define how much evidence each task contributes to the shared covariance.
Rows with no observed targets provide no training signal; the estimator below
uses `missing="drop"` so those rows are ignored during fitting.

## 2. Fit Sparse Multitask GPR

`SparseMultitaskGPRRegressor` converts each finite target entry into one
observed `(sample, task)` pair. A row with one measured property still helps
the model learn the shared input-space trends and task covariance.

```python
from matgpr import SparseMultitaskGPRRegressor

model = SparseMultitaskGPRRegressor(
    task_names=target_columns,
    task_covar_rank=1,
    kernel="matern",
    ard=True,
    training_iter=250,
    initial_noise=0.1,
    standardize_y=True,
    min_observations_per_task=8,
    missing="drop",
    random_state=42,
    verbose=False,
)

model.fit(data[feature_columns], sparse_targets)

model.task_names_
model.task_observation_counts_
```

Use `task_covar_rank=1` as a conservative default for low-data materials
datasets. Increase it only when validation shows that a more flexible task
correlation structure helps.

## 3. Validate Observed Entries

Sparse validation must ignore unobserved target entries. The helper below keeps
the target matrix sparse during fitting, evaluates only finite target entries,
and returns a long-form table for parity plots.

```python
from matgpr import evaluate_sparse_multitask_train_test_split

validation = evaluate_sparse_multitask_train_test_split(
    model,
    data[feature_columns],
    sparse_targets,
    test_size=0.25,
    random_state=7,
    model_name="sparse_multitask_gpr",
    confidence_level=0.95,
)

validation.task_metrics
```

Important columns in `validation.task_metrics`:

- `split`: `train` or `test`,
- `task`: target property name,
- `n_observed`: number of finite target values used for metrics,
- `n_missing`: number of unobserved target entries in that split,
- `RMSE`, `MAE`, `R2`, `r`: regression metrics on observed entries only,
- `mean_std`, `observed_coverage`, `NLPD`: uncertainty diagnostics when
  predictive standard deviations are available.

Use the observed prediction rows for parity plots:

```python
observed_predictions = validation.observed_predictions
observed_predictions.head()
```

The full prediction table keeps every sample-task prediction, including tasks
that were not measured:

```python
validation.predictions.head()
```

Rows with `observed == False` are predictions for unmeasured properties. They
are useful for filling a candidate property table, but they should not be used
as validation ground truth.

## 4. Fit A Final Model And Predict Missing Properties

After choosing the model settings with validation, refit on all available
observations before predicting missing entries or new candidates.

```python
final_model = SparseMultitaskGPRRegressor(
    task_names=target_columns,
    task_covar_rank=1,
    kernel="matern",
    ard=True,
    training_iter=400,
    initial_noise=0.1,
    standardize_y=True,
    min_observations_per_task=8,
    missing="drop",
    random_state=42,
    verbose=False,
)

final_model.fit(data[feature_columns], sparse_targets)
prediction = final_model.predict_distribution(
    data[feature_columns],
    confidence_level=0.95,
)
```

Create a table of model-estimated values for target entries that were not
observed:

```python
missing_rows = []
for task_index, task_name in enumerate(prediction.task_names):
    missing_mask = sparse_targets[task_name].isna().to_numpy()
    task_frame = pd.DataFrame(
        {
            "sample_index": data.index[missing_mask],
            "task": task_name,
            "predicted_mean": prediction.mean[missing_mask, task_index],
            "predicted_std": prediction.std[missing_mask, task_index],
            "lower_95": prediction.lower[missing_mask, task_index],
            "upper_95": prediction.upper[missing_mask, task_index],
        }
    )
    missing_rows.append(task_frame)

imputed_property_table = pd.concat(missing_rows, ignore_index=True)
imputed_property_table.head()
```

Call these values predictions, not measurements. Keep the uncertainty columns
attached so downstream decisions can account for risk.

## 5. Reporting Checklist

For a sparse multitask materials model, report:

- task names and units,
- per-task observation counts,
- why task sharing is plausible,
- descriptors and preprocessing,
- kernel, ARD setting, task covariance rank, and training iterations,
- train/test split protocol,
- per-task observed-entry metrics,
- uncertainty diagnostics and interval coverage,
- how predictions for unobserved properties are used.

Sparse multitask GPR is most useful when related tasks share trends. If tasks
are weakly related, measured under incompatible protocols, or dominated by
different noise sources, independent single-task models can be more reliable.
