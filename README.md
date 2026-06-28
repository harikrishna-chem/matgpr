# matgpr

Gaussian Process Regression tools for materials informatics, with first-class
support for physics-informed mean functions.

`matgpr` is an early-stage research package for building reproducible GPR
workflows on materials datasets. The first release focuses on clean data
preparation utilities, scikit-learn and GPyTorch GPR models, uncertainty-aware
prediction, and a flexible physics-informed mean-function API.

The importable package is `matgpr`.

## Repository Description

Gaussian Process Regression toolkit for materials informatics, including
physics-informed mean functions and uncertainty-aware prediction.

## Highlights

- Data cleaning, feature preprocessing, train/test splitting, and quick reports.
- Deterministic fingerprint caching and failed-row reports for repeated
  featurization workflows.
- Scikit-learn GPR helpers for baseline models and kernel search.
- GPyTorch exact GPR with ARD kernels, target standardization, and predictive
  uncertainty.
- `PhysicsInformedMean` for combining mechanistic equations with learned GP
  residuals.
- Physics-aware kernels, including Tanimoto similarity for molecular and
  polymer fingerprints and element-fraction kernels for inorganic
  compositions plus structure-feature kernels for crystal descriptors.
- Target transforms for positive, bounded, standardized, and physics-residual
  modeling workflows.
- Soft physics-constraint anchors for known limits and monotonic trends through
  virtual observations.
- Exact RBF derivative-constrained GPR for slope-informed physics trends.
- Physics-aware observation-noise profiles for mixed-source, replicate, and
  heteroscedastic datasets.
- Uncertainty diagnostics for coverage, calibration, NLPD, standardized
  residuals, and uncertainty-error trends.
- Plotting and metrics utilities for parity plots, learning curves, PCA, and
  regression quality checks.

## Suggested Workflow

1. Clean the dataframe with `normalize_column_names`,
   `replace_missing_placeholders`, `drop_duplicate_rows`, and optional
   `filter_iqr_outliers`.
2. Separate the target from features with `separate_features_target`, then make
   train/test sets with `split_train_test`.
3. Add materials descriptors with `CompositionFeaturizer`,
   `StructureFeaturizer`, `SmilesFeaturizer`, `PolymerSmilesFeaturizer`, or the
   lower-level fingerprint helpers.
4. Identify column types with `identify_feature_types` and build a transformer
   with `build_preprocessor`.
5. Choose a kernel, including `TanimotoKernel` for molecular or polymer
   fingerprints, `ElementFractionKernel` for elemental composition vectors, or
   `StructureFeatureKernel` for crystal-structure descriptors.
6. Optionally transform targets with `LogTargetTransform`,
   `BoundedTargetTransform`, `StandardizedTargetTransform`, or
   `PhysicsResidualTransform`.
7. Optionally add physics-derived virtual observations with
   `KnownLimitConstraint`, `MonotonicTrendConstraint`, and
   `append_virtual_observations`.
8. For known slope information, fit derivative-constrained models with
   `DerivativeObservationSet`, `MonotonicDerivativeConstraint`, and
   `fit_derivative_constrained_gpr`.
9. Optionally build per-row observation noise with `SourceNoiseModel`,
   `ReplicateNoiseModel`, `FeatureNoiseModel`, and `combine_noise_profiles`.
10. Train other models with `MatGPRRegressor`, `PhysicsInformedGPRRegressor`,
   `build_sklearn_gpr_model`, or the lower-level `fit_gpytorch_gpr`.
11. Evaluate point predictions with `regression_metrics` or
   `train_test_regression_metrics`.
12. Visualize results with `plot_parity`, `plot_learning_curve`,
   `plot_uncertainty_calibration`, `plot_uncertainty_vs_error`,
   `plot_distribution`, `plot_correlation_matrix`, or the PCA plotting helpers.

## Documentation

- A MkDocs site can be built locally with `python -m mkdocs serve` after
  installing `matgpr` with the `docs` extra.
- `docs/matgpr_user_guide.md` provides a practical user guide for cleaning
  data, generating materials fingerprints, training standard and
  physics-informed GPR models, introducing custom physics equations, analyzing
  model performance, and saving artifacts.
- `docs/physics_informed_gpr.md` explains how physics equations enter the GP
  mean function and what users should report for PI-GPR models.
- `docs/fingerprinting_options.md` explains available fingerprinting backends,
  when to use each option, which dependencies are core versus optional, and how
  to implement them in `matgpr` workflows.

## Module Map

| Module | Main functions/classes | Purpose |
| --- | --- | --- |
| `matgpr.data_cleaning` | `normalize_column_names`, `replace_missing_placeholders`, `drop_duplicate_rows`, `drop_columns_by_missing_fraction`, `impute_missing_values`, `filter_iqr_outliers` | Data cleaning before modeling |
| `matgpr.data_splitting` | `separate_features_target`, `split_train_test` | Target and train/test splitting |
| `matgpr.preprocessing` | `identify_feature_types`, `build_scaler`, `build_preprocessor` | Reusable feature preprocessing |
| `matgpr.featurizers` | `CompositionFeaturizer`, `StructureFeaturizer`, `SmilesFeaturizer`, `PolymerSmilesFeaturizer` | Scikit-learn-style materials featurizers |
| `matgpr.kernels` | `TanimotoKernel`, `ElementFractionKernel`, `StructureFeatureKernel`, `FeatureSubsetKernel`, `build_additive_kernel`, `build_product_kernel` | Physics-aware scikit-learn kernels |
| `matgpr.target_transforms` | `LogTargetTransform`, `BoundedTargetTransform`, `StandardizedTargetTransform`, `PhysicsResidualTransform` | Target constraints, transforms, and physics-residual modeling |
| `matgpr.physics_constraints` | `KnownLimitConstraint`, `MonotonicTrendConstraint`, `VirtualObservationSet`, `append_virtual_observations` | Soft physics anchors and virtual observations |
| `matgpr.derivative_gpr` | `DerivativeObservationSet`, `MonotonicDerivativeConstraint`, `fit_derivative_constrained_gpr` | Exact derivative-constrained RBF GPR |
| `matgpr.noise_models` | `SourceNoiseModel`, `ReplicateNoiseModel`, `FeatureNoiseModel`, `combine_noise_profiles` | Physics-aware observation-noise profiles |
| `matgpr.estimators` | `MatGPRRegressor`, `PhysicsInformedGPRRegressor` | Scikit-learn-style GPyTorch GPR estimators |
| `matgpr.sklearn_gpr` | `build_sklearn_gpr_kernel`, `build_sklearn_gpr_model`, `build_sklearn_gpr_grid_search` | Scikit-learn GPR models |
| `matgpr.gpytorch_gpr` | `PhysicsInformedMean`, `fit_gpytorch_gpr`, `train_gpytorch_gpr`, `predict_gpytorch_gpr` | GPyTorch GPR and physics-informed mean functions |
| `matgpr.metrics` | `regression_metrics`, `train_test_regression_metrics` | Model quality metrics |
| `matgpr.uncertainty` | `interval_coverage`, `calibration_curve`, `gaussian_nlpd`, `standardized_residuals`, `uncertainty_diagnostics` | Predictive uncertainty diagnostics |
| `matgpr.pca` | `fit_pca`, `summarize_pca`, `transform_pca` | PCA fitting and transformation |
| `matgpr.visualization` | `plot_parity`, `plot_learning_curve`, `plot_uncertainty_calibration`, `plot_uncertainty_vs_error`, `plot_distribution`, `plot_correlation_matrix`, `plot_pca_scree`, `plot_pca_scores` | Common model and data plots |
| `matgpr.reporting` | `summarize_missingness`, `summarize_numeric_columns` | Quick dataframe reports |
| `matgpr.io_utils` | `save_artifact`, `load_artifact`, `log_experiment_result` | Model/artifact persistence and result logging |

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

from matgpr import PhysicsInformedMean, fit_gpytorch_gpr


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

For estimator-style workflows, use `PhysicsInformedGPRRegressor`:

```python
from matgpr import PhysicsInformedGPRRegressor

model = PhysicsInformedGPRRegressor(
    equation=oxidation_equation,
    feature_indices={"temperature_c": 0, "time_min": 1},
    learnable_parameters={"A": 1.0, "Q": 100_000.0},
    positive_parameters=("A", "Q"),
    training_iter=1000,
    verbose=False,
)

model.fit(X_train, y_train)
y_pred, y_std = model.predict(X_test, return_std=True)
learned_parameters = model.learned_physics_parameters_
```

## Installation

From a local checkout:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

For development:

```bash
python3 -m pip install -e ".[dev,examples]"
python3 -m ruff check matgpr tests scripts
python3 -m pytest
python3 -m build
```

For documentation:

```bash
python3 -m pip install -e ".[docs,examples]"
python3 -m mkdocs serve
```

## Citation

If you use `matgpr` in a publication, cite the package using `CITATION.cff`.
Individual examples should also cite the original papers and datasets listed in
their reports.

## License

`matgpr` is dual-licensed:

- Community License: GNU Affero General Public License v3.0. See `LICENSE`.
- Commercial License: available for proprietary or closed-source commercial
  applications.

For commercial licensing, contact:

```text
harikrishnasahu89@gmail.com
```

## Contributing

See `CONTRIBUTING.md` for development setup, contribution expectations, and the
current licensing status. See `docs/license_strategy.md` for licensing details.

## Examples

The first worked example is available in `examples/opv`:

- `opv_gpr_modeling.ipynb` compares standard GPR with physics-informed GPR
  models of increasing mean-function complexity.
- `physics_informed_gpr_report.md` explains the physical rationale,
  implementation details, and learning-curve results.
- `dataset.csv` contains the OPV descriptor dataset used by the notebook.

## Roadmap

- Add four more published-paper example workflows for materials-informatics
  users.
- Expand documentation with tutorials, API references, and example equations.
- Add multitask Gaussian-process models.
- Add additional physics-informed model families.
- Add Bayesian optimization workflows for selecting the next experiments.

## Project Status

This repository is under active development. The public API is being shaped
around transparent examples, readable code, and research workflows that other
materials scientists can adapt.
