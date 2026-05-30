# genmatics-gpr

Small Python utilities for cleaning materials datasets, preprocessing features,
training Gaussian Process Regression models, analyzing model quality, and
making reusable plots.

## Suggested Workflow

1. Clean the dataframe with `normalize_column_names`,
   `replace_missing_placeholders`, `drop_duplicate_rows`, and optional
   `filter_iqr_outliers`.
2. Split the target from features with `split_features_target`, then create
   train/test sets with `split_train_test`.
3. Infer column types with `infer_feature_columns` and build a transformer with
   `build_preprocessor`.
4. Train a model with either `build_sklearn_gpr` or `train_gpytorch_gpr`.
5. Evaluate predictions with `calculate_regression_metrics` or
   `calculate_train_test_metrics`.
6. Visualize results with `plot_parity`, `plot_learning_curve`,
   `plot_distribution`, `plot_correlation_matrix`, or the PCA plotting helpers.

## Function Map

| Module | Main functions/classes | Purpose |
| --- | --- | --- |
| `cleaning.py` | `normalize_column_names`, `replace_missing_placeholders`, `drop_duplicate_rows`, `drop_columns_by_missing_fraction`, `impute_missing_values`, `filter_iqr_outliers` | Data cleaning before modeling |
| `splitting.py` | `split_features_target`, `split_train_test` | Target and train/test splitting |
| `scaling.py` | `make_scaler` | Standard, min-max, robust, or passthrough scaling |
| `pipeline.py` | `infer_feature_columns`, `build_preprocessor`, `save_artifact`, `load_artifact`, `append_experiment_result` | Reusable preprocessing and experiment persistence |
| `gpr_sklearn.py` | `make_sklearn_gpr_kernel`, `build_sklearn_gpr`, `build_sklearn_gpr_grid_search` | Scikit-learn GPR models |
| `gpr_gpytorch.py` | `EquationMeanFunction`, `train_gpytorch_gpr`, `predict_gpytorch_gpr` | GPyTorch GPR and physics-informed mean functions |
| `analysis.py` | `calculate_regression_metrics`, `calculate_train_test_metrics` | Model quality metrics |
| `decomposition.py` | `fit_pca`, `summarize_pca`, `transform_pca`, `plot_pca_scree`, `plot_pca_scores` | PCA analysis |
| `plot.py` | `plot_parity`, `plot_distribution`, `plot_correlation_matrix`, `plot_learning_curve` | Common model and data plots |

## Physics-Informed GPR

Physics-informed models use `EquationMeanFunction`. The user supplies:

- an equation callable,
- the feature columns used by that equation,
- optional learnable parameter initial values,
- which parameters should stay positive,
- optional feature scaling information to recover original units.

Example:

```python
import torch

from gpr_gpytorch import EquationMeanFunction, train_gpytorch_gpr


def oxidation_equation(features, parameters):
    temperature_k = features["temperature_c"] + 273.15
    time_min = torch.clamp(features["time_min"], min=1e-8)
    A = parameters["A"]
    Q = parameters["Q"]
    kp = A * torch.exp(-Q / (8.314 * temperature_k))
    return torch.sqrt(torch.clamp(kp * time_min, min=1e-12))


mean_function = EquationMeanFunction(
    equation=oxidation_equation,
    column_indices={"temperature_c": 0, "time_min": 1},
    parameter_initial_values={"A": 1.0, "Q": 100_000.0},
    positive_parameters=("A", "Q"),
)

model, likelihood = train_gpytorch_gpr(
    X_train,
    y_train,
    mean_module=mean_function,
)
```

This replaces hard-coded equation-specific mean classes. New physics equations
can be added in notebooks or scripts without changing the library.

## Dependencies

Install the dependencies with:

```bash
python3 -m pip install -r requirements.txt
```
