# matgpr

Gaussian Process Regression tools for materials informatics, with first-class
support for physics-informed mean functions.

`matgpr` is an early-stage research package for building reproducible GPR
workflows on materials datasets. The first release focuses on clean data
preparation utilities, scikit-learn and GPyTorch GPR models, uncertainty-aware
prediction, and a flexible physics-informed mean-function API.

The current importable package is `genmatics_gpr`.

## Repository Description

Gaussian Process Regression toolkit for materials informatics, including
physics-informed mean functions and uncertainty-aware prediction.

## Highlights

- Data cleaning, feature preprocessing, train/test splitting, and quick reports.
- Scikit-learn GPR helpers for baseline models and kernel search.
- GPyTorch exact GPR with ARD kernels, target standardization, and predictive
  uncertainty.
- `PhysicsInformedMean` for combining mechanistic equations with learned GP
  residuals.
- Plotting and metrics utilities for parity plots, learning curves, PCA, and
  regression quality checks.

## Suggested Workflow

1. Clean the dataframe with `normalize_column_names`,
   `replace_missing_placeholders`, `drop_duplicate_rows`, and optional
   `filter_iqr_outliers`.
2. Separate the target from features with `separate_features_target`, then make
   train/test sets with `split_train_test`.
3. Identify column types with `identify_feature_types` and build a transformer
   with `build_preprocessor`.
4. Train a model with either `build_sklearn_gpr_model`,
   `fit_gpytorch_gpr`, or `train_gpytorch_gpr`.
5. Evaluate predictions with `regression_metrics` or
   `train_test_regression_metrics`.
6. Visualize results with `plot_parity`, `plot_learning_curve`,
   `plot_distribution`, `plot_correlation_matrix`, or the PCA plotting helpers.

## Module Map

| Module | Main functions/classes | Purpose |
| --- | --- | --- |
| `genmatics_gpr.data_cleaning` | `normalize_column_names`, `replace_missing_placeholders`, `drop_duplicate_rows`, `drop_columns_by_missing_fraction`, `impute_missing_values`, `filter_iqr_outliers` | Data cleaning before modeling |
| `genmatics_gpr.data_splitting` | `separate_features_target`, `split_train_test` | Target and train/test splitting |
| `genmatics_gpr.preprocessing` | `identify_feature_types`, `build_scaler`, `build_preprocessor` | Reusable feature preprocessing |
| `genmatics_gpr.sklearn_gpr` | `build_sklearn_gpr_kernel`, `build_sklearn_gpr_model`, `build_sklearn_gpr_grid_search` | Scikit-learn GPR models |
| `genmatics_gpr.gpytorch_gpr` | `PhysicsInformedMean`, `fit_gpytorch_gpr`, `train_gpytorch_gpr`, `predict_gpytorch_gpr` | GPyTorch GPR and physics-informed mean functions |
| `genmatics_gpr.metrics` | `regression_metrics`, `train_test_regression_metrics` | Model quality metrics |
| `genmatics_gpr.pca` | `fit_pca`, `summarize_pca`, `transform_pca` | PCA fitting and transformation |
| `genmatics_gpr.visualization` | `plot_parity`, `plot_distribution`, `plot_correlation_matrix`, `plot_learning_curve`, `plot_pca_scree`, `plot_pca_scores` | Common model and data plots |
| `genmatics_gpr.reporting` | `summarize_missingness`, `summarize_numeric_columns` | Quick dataframe reports |
| `genmatics_gpr.io_utils` | `save_artifact`, `load_artifact`, `log_experiment_result` | Model/artifact persistence and result logging |

## Physics-Informed GPR

Physics-informed models use `PhysicsInformedMean`. The user supplies:

- an equation callable,
- the feature columns used by that equation,
- optional learnable parameter initial values,
- optional fixed physical constants,
- which learnable parameters should stay positive,
- optional feature scaling information to recover original units.

Example:

```python
import torch

from genmatics_gpr import PhysicsInformedMean, fit_gpytorch_gpr


def oxidation_equation(features, parameters):
    temperature_k = features["temperature_c"] + 273.15
    time_min = torch.clamp(features["time_min"], min=1e-8)
    A = parameters["A"]
    Q = parameters["Q"]
    kp = A * torch.exp(-Q / (8.314 * temperature_k))
    return torch.sqrt(torch.clamp(kp * time_min, min=1e-12))


mean_function = PhysicsInformedMean(
    equation=oxidation_equation,
    feature_indices={"temperature_c": 0, "time_min": 1},
    learnable_parameters={"A": 1.0, "Q": 100_000.0},
    positive_parameters=("A", "Q"),
)

result = fit_gpytorch_gpr(
    X_train,
    y_train,
    mean_module=mean_function,
)
prediction = result.predict(X_test)
```

This replaces hard-coded equation-specific mean classes. New physics equations
can be added in notebooks or scripts without changing the library. The older
`EquationMeanFunction` name remains available as a compatibility alias.

## Installation

From a local checkout:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Roadmap

- Add five published-paper example workflows for materials-informatics users.
- Expand documentation with tutorials, API references, and example equations.
- Add multitask Gaussian-process models.
- Add additional physics-informed model families.
- Add Bayesian optimization workflows for selecting the next experiments.

## Project Status

This repository is under active development. The public API is being shaped
around transparent examples, readable code, and research workflows that other
materials scientists can adapt.
