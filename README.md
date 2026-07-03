# matgpr

Gaussian Process Regression tools for materials informatics, with first-class
support for physics-informed mean functions.

`matgpr` is an early-stage research package for building reproducible GPR
workflows on materials datasets. The first release focuses on clean data
preparation utilities, scikit-learn and GPyTorch GPR models, uncertainty-aware
prediction, and a flexible physics-informed mean-function API.

The importable package is `matgpr`. Python 3.10 or newer is required.

`matgpr` is currently an active-development `0.x` package. Pin release tags or
commit hashes for reproducible research workflows, and see
`docs/versioning.md` for the API-stability and versioning policy.

## Repository Description

Gaussian Process Regression toolkit for materials informatics, including
physics-informed mean functions and uncertainty-aware prediction.

## Highlights

- Data cleaning, feature preprocessing, train/test splitting, and quick reports.
- Native estimator missing-value policies with auditable reports for rejected,
  dropped, or imputed rows.
- Optional-dependency helpers with clear install messages for advanced
  fingerprinting backends.
- Deterministic fingerprint caching and failed-row reports for repeated
  featurization workflows.
- Optional matminer Magpie composition descriptors through
  `MagpieCompositionFeaturizer` and helper functions.
- Scikit-learn GPR helpers for baseline models and kernel search.
- GPyTorch exact GPR with ARD kernels, target standardization, and predictive
  uncertainty.
- Learned heteroscedastic GPR with a residual noise GP for input-dependent
  observation uncertainty.
- Exact multitask GPR for complete or sparse multi-property materials datasets
  with learned inter-task covariance, shared or task-specific sparse noise, and
  scikit-learn-style wrappers.
- Scikit-learn-compatible estimators and featurizers for pipelines, grid
  search, and reusable validation workflows.
- `PhysicsInformedMean` for combining mechanistic equations with learned GP
  residuals.
- Reusable physics equation templates for Arrhenius, square-root-time,
  power-law, Hall-Petch, free-volume, and rule-of-mixtures mean functions.
- Physics-aware kernels, including Tanimoto similarity for molecular and
  polymer fingerprints and element-fraction kernels for inorganic
  compositions plus structure-feature kernels for crystal descriptors.
- Target transforms for positive, bounded, standardized, and physics-residual
  modeling workflows, with materials-property presets for common target types.
- Soft physics-constraint anchors for known limits and monotonic trends through
  virtual observations.
- Exact RBF derivative-constrained GPR for slope-informed physics trends.
- Physics-aware observation-noise profiles for mixed-source, replicate, and
  heteroscedastic datasets.
- Validation APIs for train/test evaluation, cross-validation summaries, and
  configurable learning curves.
- Candidate-generation helpers for finite chemistry, composition,
  formulation, and processing-condition pools.
- Candidate-pool diagnostics for descriptor completeness, duplicate keys,
  numeric feature coverage, and categorical search-space coverage.
- Optional BoTorch Bayesian optimization helpers for single- and
  multi-objective finite materials candidate pools, known observation noise,
  feasibility constraints, trust regions, duplicate avoidance, and diverse or
  sequential next-experiment batches.
- BO recommendation audit summaries for acquisition scores, posterior
  uncertainty, feasibility, trust-region, duplicate, and batch-selection
  decisions.
- Multi-objective finite-pool selection with Pareto-front and weighted
  scalarization utilities.
- Closed-loop experiment logging and restart state for BO recommendations,
  selected experiments, measured observations, pending candidates, and
  campaign summaries.
- Lightweight finite-pool BO benchmarks for comparing acquisition strategies,
  physics-prior scores, model scores, and random baselines on known outcomes.
- BO visualization helpers for benchmark traces, regret curves, and
  closed-loop campaign progress.
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
   `MagpieCompositionFeaturizer`, `StructureFeaturizer`, `SmilesFeaturizer`,
   `PolymerSmilesFeaturizer`, or the lower-level fingerprint helpers.
4. Identify column types with `identify_feature_types` and build a transformer
   with `build_preprocessor`.
5. Choose a kernel, including `TanimotoKernel` for molecular or polymer
   fingerprints, `ElementFractionKernel` for elemental composition vectors, or
   `StructureFeatureKernel` for crystal-structure descriptors.
6. Optionally transform targets with `make_materials_target_transform`,
   `LogTargetTransform`, `BoundedTargetTransform`,
   `StandardizedTargetTransform`, or `PhysicsResidualTransform`.
7. Optionally add physics-derived virtual observations with
   `KnownLimitConstraint`, `MonotonicTrendConstraint`, and
   `append_virtual_observations`.
8. For known slope information, fit derivative-constrained models with
   `DerivativeObservationSet`, `MonotonicDerivativeConstraint`, and
   `fit_derivative_constrained_gpr`.
9. Optionally build per-row observation noise with `SourceNoiseModel`,
   `ReplicateNoiseModel`, `FeatureNoiseModel`, and `combine_noise_profiles`.
10. Optionally learn input-dependent noise with `fit_heteroscedastic_gpr` when
   residual variance changes across the materials space.
11. Optionally start from a reusable physics equation template with
   `summarize_physics_equation_templates`, `search_physics_equation_templates`,
   and `get_physics_equation_template`.
12. Train other models with `MatGPRRegressor`, `PhysicsInformedGPRRegressor`,
   `build_sklearn_gpr_model`, lower-level `fit_gpytorch_gpr`,
   `MultitaskGPRRegressor`, `SparseMultitaskGPRRegressor`,
   `fit_multitask_gpytorch_gpr`, or `fit_sparse_multitask_gpytorch_gpr` for
   multi-property targets.
13. Evaluate models with `evaluate_train_test_split`,
   `evaluate_multitask_train_test_split`,
   `evaluate_sparse_multitask_train_test_split`, `cross_validate_regressor`,
   `learning_curve`, `regression_metrics`, `train_test_regression_metrics`,
   `summarize_multitask_predictions`, or
   `summarize_sparse_multitask_predictions`.
14. Visualize results with `plot_parity`, `plot_learning_curve`,
   `plot_uncertainty_calibration`, `plot_uncertainty_vs_error`,
   `plot_distribution`, `plot_correlation_matrix`, BO progress plots, or the
   PCA plotting helpers.
15. Optionally build and audit finite candidate pools, rank next experiments
   with Bayesian optimization, summarize recommendation audits, and log
   closed-loop campaigns.

## Documentation

- A MkDocs site can be built locally with `python -m mkdocs serve` after
  installing `matgpr` with the `docs` extra.
- `docs/quickstart.md` shows the compact path from standard GPR to
  physics-informed GPR.
- `docs/example_cards.md` links to dataset cards and model cards for the
  public examples.
- `docs/matgpr_user_guide.md` provides a practical user guide for cleaning
  data, generating materials fingerprints, training standard and
  physics-informed GPR models, introducing custom physics equations, analyzing
  model performance, and saving artifacts.
- `docs/physics_informed_gpr.md` explains how physics equations enter the GP
  mean function and what users should report for PI-GPR models.
- `docs/pi_gpr_guarantees.md` clarifies what PI-GPR does and does not
  guarantee.
- `docs/multitask_gpr.md` explains correlated multitask GPR APIs for complete
  and sparse multi-property materials datasets.
- `docs/sparse_multitask_walkthrough.md` provides a compact sparse multitask
  workflow for incomplete multi-property target matrices.
- `docs/sparse_multitask_noise_design.md` explains shared and task-specific
  sparse multitask observation noise.
- `docs/fingerprinting_options.md` explains available fingerprinting backends,
  when to use each option, which dependencies are core versus optional, and how
  to implement them in `matgpr` workflows.
- `docs/versioning.md` explains the active-development status, API-stability
  policy, and how to pin releases or commits for reproducible workflows.
- `docs/release_checklist.md` defines the release gate for `v0.1.0` and later
  `0.x` releases.
- `docs/pypi_readiness.md` records the PyPI readiness audit and remaining
  upload blockers.

## Versioning And API Stability

Current version: `0.1.0`.

`matgpr` follows semantic-versioning conventions in spirit, but minor `0.x`
releases may include breaking API changes while the package is still being
shaped. For publications, benchmarks, and production workflows, install from a
release tag or exact commit instead of the moving `main` branch:

```bash
python3 -m pip install "matgpr[examples] @ git+https://github.com/harikrishna-chem/matgpr.git@v0.1.0"
```

Record the `matgpr` version or commit hash, Python version, and dependency lock
file alongside any published benchmark results.

## Module Map

| Module | Main functions/classes | Purpose |
| --- | --- | --- |
| `matgpr.data_cleaning` | `normalize_column_names`, `replace_missing_placeholders`, `drop_duplicate_rows`, `drop_columns_by_missing_fraction`, `impute_missing_values`, `filter_iqr_outliers` | Data cleaning before modeling |
| `matgpr.data_splitting` | `separate_features_target`, `split_train_test` | Target and train/test splitting |
| `matgpr.preprocessing` | `identify_feature_types`, `build_scaler`, `build_preprocessor` | Reusable feature preprocessing |
| `matgpr.featurizers` | `CompositionFeaturizer`, `MagpieCompositionFeaturizer`, `StructureFeaturizer`, `SmilesFeaturizer`, `PolymerSmilesFeaturizer` | Scikit-learn-style materials featurizers |
| `matgpr.kernels` | `TanimotoKernel`, `ElementFractionKernel`, `StructureFeatureKernel`, `FeatureSubsetKernel`, `build_additive_kernel`, `build_product_kernel` | Physics-aware scikit-learn kernels |
| `matgpr.target_transforms` | `make_materials_target_transform`, `summarize_target_transform_specs`, `LogTargetTransform`, `BoundedTargetTransform`, `StandardizedTargetTransform`, `PhysicsResidualTransform` | Target constraints, property presets, transforms, and physics-residual modeling |
| `matgpr.physics_constraints` | `KnownLimitConstraint`, `MonotonicTrendConstraint`, `VirtualObservationSet`, `append_virtual_observations` | Soft physics anchors and virtual observations |
| `matgpr.derivative_gpr` | `DerivativeObservationSet`, `MonotonicDerivativeConstraint`, `fit_derivative_constrained_gpr` | Exact derivative-constrained RBF GPR |
| `matgpr.noise_models` | `SourceNoiseModel`, `ReplicateNoiseModel`, `FeatureNoiseModel`, `combine_noise_profiles` | Physics-aware observation-noise profiles |
| `matgpr.heteroscedastic_gpr` | `fit_heteroscedastic_gpr`, `HeteroscedasticGPRResult` | Learned input-dependent observation-noise GPR |
| `matgpr.multitask_gpr` | `fit_multitask_gpytorch_gpr`, `train_multitask_gpytorch_gpr`, `predict_multitask_gpytorch_gpr` | Exact multitask GPR for complete multi-property target matrices |
| `matgpr.physics_equations` | `PhysicsEquationTemplate`, `get_physics_equation_template`, `search_physics_equation_templates`, `summarize_physics_equation_templates` | Reusable materials-physics mean-equation templates and metadata discovery |
| `matgpr.estimators` | `MatGPRRegressor`, `PhysicsInformedGPRRegressor`, `MultitaskGPRRegressor`, `MissingValueReport` | Scikit-learn-style GPyTorch GPR estimators with missing-value reports |
| `matgpr.sklearn_gpr` | `build_sklearn_gpr_kernel`, `build_sklearn_gpr_model`, `build_sklearn_gpr_grid_search` | Scikit-learn GPR models |
| `matgpr.gpytorch_gpr` | `PhysicsInformedMean`, `fit_gpytorch_gpr`, `train_gpytorch_gpr`, `predict_gpytorch_gpr` | GPyTorch GPR and physics-informed mean functions |
| `matgpr.validation` | `evaluate_train_test_split`, `evaluate_multitask_train_test_split`, `evaluate_sparse_multitask_train_test_split`, `cross_validate_regressor`, `learning_curve`, `summarize_multitask_predictions`, `summarize_sparse_multitask_predictions` | Reusable train/test, complete/sparse multitask task-summary, cross-validation, and learning-curve workflows |
| `matgpr.sparse_multitask_gpr` | `fit_sparse_multitask_gpytorch_gpr`, `prepare_sparse_multitask_observations`, `predict_sparse_multitask_gpytorch_gpr` | Exact sparse multitask GPR for incomplete multi-property target matrices |
| `matgpr.candidate_generation` | `build_cartesian_candidate_grid`, `build_composition_candidate_grid`, `summarize_candidate_pool`, `summarize_candidate_feature_coverage`, `summarize_candidate_category_coverage`, `exclude_existing_candidates`, `split_candidate_features` | Finite candidate-pool builders and diagnostics for BO |
| `matgpr.bayesian_optimization` | `suggest_next_experiments`, `suggest_multi_objective_next_experiments`, `summarize_bo_recommendation_audit`, `select_sequential_multi_objective_batch`, `observation_noise_variance`, `select_diverse_batch`, `CandidateConstraint`, `CandidateTrustRegion`, `CandidateDuplicatePolicy` | Optional BoTorch finite-pool Bayesian optimization |
| `matgpr.bo_benchmarking` | `BOBenchmarkStrategy`, `simulate_bo_strategy`, `compare_bo_strategies`, `summarize_bo_benchmark` | Offline finite-pool BO strategy benchmarks |
| `matgpr.multi_objective` | `ObjectiveSpec`, `rank_multi_objective_candidates`, `select_pareto_front`, `scalarize_objectives` | Pareto-front and weighted multi-objective candidate ranking |
| `matgpr.experiment_logging` | `resume_bo_campaign`, `infer_next_bo_iteration`, `log_bo_recommendations`, `log_selected_experiments`, `log_observations`, `summarize_closed_loop_log` | Closed-loop BO campaign state and audit trails |
| `matgpr.metrics` | `regression_metrics`, `train_test_regression_metrics` | Model quality metrics |
| `matgpr.uncertainty` | `interval_coverage`, `calibration_curve`, `gaussian_nlpd`, `standardized_residuals`, `uncertainty_diagnostics` | Predictive uncertainty diagnostics |
| `matgpr.pca` | `fit_pca`, `summarize_pca`, `transform_pca` | PCA fitting and transformation |
| `matgpr.visualization` | `plot_parity`, `plot_learning_curve`, `plot_bo_benchmark_trace`, `plot_bo_regret_trace`, `plot_bo_campaign_progress`, `plot_uncertainty_calibration`, `plot_uncertainty_vs_error`, `plot_distribution`, `plot_correlation_matrix`, `plot_pca_scree`, `plot_pca_scores` | Common model, BO, and data plots |
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
python3 -m pip install -e .
```

For development:

```bash
python3 -m pip install -e ".[dev,examples,bo]"
python3 -m ruff check matgpr tests scripts
python3 -m pytest
python3 -m build
```

For documentation:

```bash
python3 -m pip install -e ".[docs,examples]"
python3 -m mkdocs serve
```

For Bayesian optimization:

```bash
python3 -m pip install -e ".[bo]"
```

For optional matminer Magpie descriptors:

```bash
python3 -m pip install -e ".[materials-extra]"
```

## Citation

If you use `matgpr` in a publication, cite the package using `CITATION.cff`.
Individual examples should also cite the original papers and datasets listed in
their reports.

## License

`matgpr` is released under the Apache License 2.0. See `LICENSE`.

## Contributing

See `CONTRIBUTING.md` for development setup, contribution expectations, and
licensing details.

## Examples

Worked examples are available under `examples/`:

- `examples/opv/opv_gpr_modeling.ipynb` compares standard GPR with
  physics-informed GPR models of increasing mean-function complexity.
- `examples/solvent_diffusivity/solvent_diffusivity_gpr_modeling.ipynb`
  compares temperature, concentration, and solvent-size physics means.
- Example reports explain the physical rationale, implementation details,
  validation protocol, and modeling conclusions.
- Each public example includes `dataset_card.md` and `model_card.md` files for
  provenance, features, validation protocol, learned physics terms, intended
  use, and known limitations.

The public example notebooks include Colab badges. When opened in Colab, they
install `matgpr[examples]` from GitHub and download the matching `dataset.pkl`
from the repository.

## Roadmap

- Continue refining published-paper example workflows for
  materials-informatics users.
- Expand documentation with tutorials, API references, and example equations.
- Add multitask Gaussian-process models.
- Add additional physics-informed model families.
- Expand Bayesian optimization examples for next-experiment selection.

## Project Status

This repository is under active development. The public API is being shaped
around transparent examples, readable code, and research workflows that other
materials scientists can adapt.
