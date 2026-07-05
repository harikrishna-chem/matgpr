# matgpr User Guide

`matgpr` is a materials-informatics toolkit for building Gaussian Process
Regression workflows with clean data preparation, materials fingerprints,
uncertainty-aware prediction, model analysis, and physics-informed mean
functions.

This guide is intended for users who want to call the package functions in
their own notebooks or scripts.

## 1. Typical Workflow

Most `matgpr` projects follow this pattern:

1. Load a dataframe.
2. Clean column names, missing values, duplicates, and outliers.
3. Add materials descriptors or fingerprints.
4. Split features and target into train/test sets.
5. Scale or preprocess features using only the training data.
6. Train a standard GPR or physics-informed GPR model.
7. Predict with uncertainty.
8. Analyze RMSE, R2, Pearson r, parity plots, uncertainty calibration,
   learning curves, PCA, and feature effects.
9. Optionally build finite candidate pools and rank next experiments with
   Bayesian optimization.
10. Save models, preprocessors, metrics, and plots.

Core imports:

```python
import numpy as np
import pandas as pd

from matgpr import (
    normalize_column_names,
    replace_missing_placeholders,
    drop_duplicate_rows,
    drop_columns_by_missing_fraction,
    filter_iqr_outliers,
    impute_missing_values,
    summarize_missingness,
    summarize_numeric_columns,
    decompose_multifidelity_prediction,
    summarize_multifidelity_components,
    CompositionFeaturizer,
    MagpieCompositionFeaturizer,
    StructureFeaturizer,
    SmilesFeaturizer,
    PolymerSmilesFeaturizer,
    separate_features_target,
    split_train_test,
    identify_feature_types,
    build_preprocessor,
    MatGPRRegressor,
    PhysicsInformedGPRRegressor,
    MultitaskGPRRegressor,
    SparseMultitaskGPRRegressor,
    KnownLimitConstraint,
    MonotonicTrendConstraint,
    append_virtual_observations,
    DerivativeObservationSet,
    MonotonicDerivativeConstraint,
    fit_derivative_constrained_gpr,
    SourceNoiseModel,
    ReplicateNoiseModel,
    FeatureNoiseModel,
    combine_noise_profiles,
    BoundedTargetTransform,
    ElementFractionKernel,
    StructureFeatureKernel,
    TanimotoKernel,
    FeatureSubsetKernel,
    build_additive_kernel,
    LogTargetTransform,
    PhysicsResidualTransform,
    StandardizedTargetTransform,
    make_materials_target_transform,
    summarize_target_transform_specs,
    build_sklearn_gpr_model,
    fit_gpytorch_gpr,
    fit_heteroscedastic_gpr,
    fit_multitask_gpytorch_gpr,
    fit_sparse_multitask_gpytorch_gpr,
    regression_metrics,
    train_test_regression_metrics,
    uncertainty_diagnostics,
    calibration_curve,
    build_cartesian_candidate_grid,
    build_composition_candidate_grid,
    exclude_existing_candidates,
    split_candidate_features,
    summarize_candidate_category_coverage,
    summarize_candidate_feature_coverage,
    summarize_candidate_pool,
    ObjectiveSpec,
    rank_multi_objective_candidates,
    select_pareto_front,
    CandidateConstraint,
    select_diverse_batch,
    select_sequential_multi_objective_batch,
    suggest_multi_objective_next_experiments,
    suggest_next_experiments,
    log_bo_recommendations,
    log_selected_experiments,
    log_observations,
    summarize_closed_loop_log,
    summarize_bo_recommendation_audit,
    plot_parity,
    plot_learning_curve,
    plot_bo_benchmark_trace,
    plot_bo_regret_trace,
    plot_bo_campaign_progress,
    plot_uncertainty_calibration,
    plot_uncertainty_vs_error,
    save_artifact,
)
```

## 2. Data Cleaning

Start by making column names stable and missing values explicit.

```python
data = pd.read_pickle("dataset.pkl")

data = normalize_column_names(data)
data = replace_missing_placeholders(data)
data = drop_duplicate_rows(data)

missing_report = summarize_missingness(data)
numeric_report = summarize_numeric_columns(data)
```

The public examples use `dataset.pkl` files stored in this repository. Only
load pickle files from trusted sources; for third-party datasets, prefer CSV,
Parquet, or a documented data-preparation script.

Useful cleaning functions:

| Function | Purpose |
| --- | --- |
| `normalize_column_names(df)` | Converts columns to lowercase `snake_case` and handles duplicates. |
| `replace_missing_placeholders(df)` | Converts strings such as `""`, `"na"`, `"none"`, and `"-"` into `np.nan`. |
| `drop_duplicate_rows(df, subset=None)` | Removes duplicate rows and resets the index. |
| `drop_columns_by_missing_fraction(df, max_missing_fraction=0.5)` | Drops columns with too much missing data. |
| `impute_missing_values(df, strategy="median", columns=None)` | Imputes selected columns with a scikit-learn `SimpleImputer`. |
| `filter_iqr_outliers(df, columns, factor=1.5)` | Removes rows outside the IQR-based range for selected numeric columns. |

Example:

```python
data = drop_columns_by_missing_fraction(data, max_missing_fraction=0.4)
data = filter_iqr_outliers(data, columns=["target_property"], factor=2.5)
data = impute_missing_values(data, strategy="median")
```

Use outlier filtering carefully. In materials datasets, extreme values can be
real high-performing materials, not data errors.

## 3. Materials Fingerprints

For a detailed decision guide across composition, molecular, polymer, crystal,
and atomistic fingerprints, see `docs/fingerprinting_options.md`.

### 3.1 Inorganic Composition Fingerprints

For inorganic formulas, `matgpr` uses `pymatgen` to parse compositions and
build statistical elemental-property descriptors.

```python
from matgpr import (
    append_composition_fingerprints,
    append_element_fractions,
    featurize_compositions,
)

data = append_composition_fingerprints(
    data,
    formula_column="composition",
    errors="coerce",
)
data = data.dropna().reset_index(drop=True)
```

The default descriptor set summarizes elemental properties with statistics such
as minimum, maximum, range, fraction-weighted mean, absolute deviation, and
standard deviation.

For composition-aware kernels, create fixed element-fraction vectors instead of
statistical descriptors:

```python
composition_vectors = append_element_fractions(
    data,
    formula_column="composition",
    elements=("B", "C", "N", "O", "Al", "Si"),
    errors="coerce",
)
```

Useful functions:

| Function | Purpose |
| --- | --- |
| `clean_formula(formula)` | Normalizes formula text before parsing. |
| `composition_fingerprint(formula)` | Featurizes one inorganic formula. |
| `featurize_compositions(formulas, errors="raise")` | Featurizes many formulas and returns features plus failed rows. |
| `append_composition_fingerprints(data, formula_column="composition")` | Appends descriptors to an existing dataframe. |
| `default_element_symbols()` | Returns periodic-table symbols in atomic-number order. |
| `element_fraction_fingerprint(formula, elements=...)` | Builds one fixed element-fraction vector. |
| `featurize_element_fractions(formulas, elements=...)` | Builds element-fraction vectors for many formulas. |
| `append_element_fractions(data, formula_column="composition")` | Appends element-fraction columns to a dataframe. |

For scikit-learn-style workflows, use `CompositionFeaturizer`:

```python
from matgpr import CompositionFeaturizer

composition_featurizer = CompositionFeaturizer(
    formula_column="composition",
    errors="coerce",
    cache_dir="fingerprint_cache",
)

composition_features = composition_featurizer.fit_transform(data)
failed_formulas = composition_featurizer.failed_
feature_names = composition_featurizer.get_feature_names_out()
```

Set `cache_dir` when repeated notebook runs recompute the same fingerprints.
Each row receives a deterministic cache key based on the input and descriptor
settings. Failed-row reports include that key for traceability.

For a stronger published composition-descriptor baseline, install the optional
`materials-extra` dependencies and use matminer's Magpie descriptors through
`MagpieCompositionFeaturizer`:

```bash
python -m pip install "matgpr[materials-extra]"
```

```python
from matgpr import MagpieCompositionFeaturizer

magpie_featurizer = MagpieCompositionFeaturizer(
    formula_column="composition",
    properties=("Number", "AtomicWeight", "Electronegativity"),
    statistics=("mean", "range", "avg_dev"),
    errors="coerce",
)

magpie_features = magpie_featurizer.fit_transform(data)
failed_formulas = magpie_featurizer.failed_
```

Magpie descriptors summarize elemental-property tables with statistics such as
mean, range, mode, and average deviation. They are useful as a stronger
composition-only baseline for inorganic materials, but they add an optional
dependency and can create a larger feature space than the lightweight built-in
composition descriptors.

### 3.2 Crystal Structure Fingerprints

For crystal structures, `matgpr` accepts `pymatgen.Structure` objects,
structure-file paths, or CIF strings. The built-in descriptors are lightweight
global structure features intended for small tabular GPR workflows.

```python
from matgpr import StructureFeaturizer, append_structure_fingerprints

structure_data = append_structure_fingerprints(
    data,
    structure_column="structure",
    errors="coerce",
)

structure_featurizer = StructureFeaturizer(
    structure_column="structure",
    errors="coerce",
    cache_dir="fingerprint_cache",
)

structure_features = structure_featurizer.fit_transform(data)
failed_structures = structure_featurizer.failed_
structure_feature_names = structure_featurizer.get_feature_names_out()
```

The default structure descriptors include sorted log lattice lengths, sorted
cosines of lattice angles, log volume per atom, and density. These descriptors
are not a replacement for local-environment descriptors such as SOAP or MBTR,
but they provide a fast structure-aware baseline without extra heavy
dependencies.

Useful functions:

| Function | Purpose |
| --- | --- |
| `structure_fingerprint(structure)` | Featurizes one `pymatgen.Structure`, structure file, or CIF string. |
| `featurize_structures(structures, errors="raise")` | Featurizes many structures and returns features plus failed rows. |
| `append_structure_fingerprints(data, structure_column="structure")` | Appends structure descriptors to a dataframe. |
| `structure_feature_names(...)` | Returns validated descriptor names, optionally with a prefix. |

### 3.3 Molecule and Polymer Fingerprints

For organic molecules and polymers, `matgpr` uses RDKit.

Molecule SMILES are canonicalized directly before fingerprinting. Polymer
repeat-unit SMILES must contain exactly two `[*]` dummy atoms. By default,
`matgpr` builds a cyclic trimer surrogate:

1. Repeat the unit three times.
2. Connect adjacent repeat units through the two `[*]` neighbors.
3. Preserve the dummy-end bond order.
4. Close the final repeat unit back to the first repeat unit.
5. Remove all `[*]` atoms.
6. RDKit-canonicalize the cyclic trimer.
7. Compute fingerprints or descriptors.

```python
from matgpr import featurize_smiles, append_smiles_features

polymer_result = featurize_smiles(
    data["polymer_smiles"],
    smiles_type="polymer",
    fingerprint_type="morgan+descriptors",
    n_bits=256,
    column_prefix="polymer",
    errors="coerce",
)

molecule_result = featurize_smiles(
    data["solvent_smiles"],
    smiles_type="molecule",
    fingerprint_type="morgan+descriptors",
    n_bits=256,
    column_prefix="solvent",
    errors="coerce",
)

model_data = pd.concat(
    [
        data.reset_index(drop=True),
        polymer_result.canonical_smiles,
        molecule_result.canonical_smiles,
        polymer_result.features,
        molecule_result.features,
    ],
    axis=1,
).dropna().reset_index(drop=True)
```

Supported `fingerprint_type` values:

| Value | Meaning |
| --- | --- |
| `"morgan"` | Morgan circular fingerprint bits. |
| `"rdkit"` | RDKit topological fingerprint bits. |
| `"maccs"` | MACCS keys. |
| `"descriptors"` | Selected RDKit molecular descriptors. |
| `"morgan+descriptors"` | Morgan bits plus descriptor columns. |
| `"rdkit+descriptors"` | RDKit bits plus descriptor columns. |
| `"maccs+descriptors"` | MACCS keys plus descriptor columns. |

Useful functions:

| Function | Purpose |
| --- | --- |
| `canonicalize_molecule_smiles(smiles)` | Canonicalizes one molecule SMILES. |
| `canonicalize_polymer_smiles(smiles, repeat_units=3)` | Builds and canonicalizes the cyclic polymer surrogate. |
| `fingerprint_smiles(smiles, smiles_type="molecule")` | Featurizes one SMILES string. |
| `featurize_smiles(smiles_values, errors="raise")` | Featurizes many SMILES strings and records failures. |
| `append_smiles_features(data, smiles_column=...)` | Appends RDKit features to an existing dataframe. |

For scikit-learn-style workflows:

```python
from matgpr import PolymerSmilesFeaturizer, SmilesFeaturizer

polymer_featurizer = PolymerSmilesFeaturizer(
    smiles_column="polymer_smiles",
    fingerprint_type="morgan+descriptors",
    n_bits=256,
    errors="coerce",
    cache_dir="fingerprint_cache",
)
polymer_features = polymer_featurizer.fit_transform(data)

solvent_featurizer = SmilesFeaturizer(
    smiles_column="solvent_smiles",
    fingerprint_type="descriptors",
    cache_dir="fingerprint_cache",
)
solvent_features = solvent_featurizer.fit_transform(data)
```

`SmilesFeaturizer` and `PolymerSmilesFeaturizer` store the most recent
canonical SMILES in `canonical_smiles_` and failed rows in `failed_`.
They also expose `cache_keys_` and `cache_hit_` when caching is enabled.

## 4. Prepare Features and Targets

For dataframe workflows:

```python
target_column = "target_property"
X, y = separate_features_target(model_data, target_column)

X_train, X_test, y_train, y_test = split_train_test(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
```

For scikit-learn pipelines, identify feature types and build a reusable
preprocessor:

```python
numeric_features, categorical_features = identify_feature_types(X_train)

preprocessor = build_preprocessor(
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    scaler="standard",
    numeric_imputation="median",
    categorical_imputation="most_frequent",
)
```

Important rule: fit preprocessors only on training data. Reuse the fitted
preprocessor for validation, test, and prediction data.

## 5. Standard GPR Models

### 5.1 Scikit-Learn Baseline

Use scikit-learn GPR when you want a compact baseline, simple grid search, or
easy integration with scikit-learn pipelines.

```python
from sklearn.pipeline import Pipeline

from matgpr import build_sklearn_gpr_model

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "gpr",
            build_sklearn_gpr_model(
                kernel="matern",
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
y_test_pred, y_test_std = model.predict(X_test, return_std=True)
metrics = regression_metrics(y_test, y_test_pred)
```

Available helper functions:

| Function | Purpose |
| --- | --- |
| `build_sklearn_gpr_kernel(name="matern")` | Builds RBF, Matern, ARD, Tanimoto, element-fraction, or structure kernels. |
| `build_sklearn_gpr_model(...)` | Returns an unfitted `GaussianProcessRegressor`. |
| `build_sklearn_gpr_grid_search(...)` | Returns a `GridSearchCV` over common GPR kernels and settings. |

### 5.2 MatGPR Estimator API

Use `MatGPRRegressor` when you want GPyTorch uncertainty handling with a
scikit-learn-style estimator interface.

```python
from matgpr import MatGPRRegressor

model = MatGPRRegressor(
    kernel="matern",
    ard=True,
    lr=0.03,
    training_iter=1000,
    initial_noise=0.05,
    standardize_y=True,
    random_state=42,
)

model.fit(X_train_array, y_train.to_numpy())
y_test_pred, y_test_std = model.predict(X_test_array, return_std=True)
metrics = regression_metrics(y_test, y_test_pred)
```

Estimator-level missing-value handling is available when quick modeling or
candidate-pool prediction needs a local policy:

```python
model = MatGPRRegressor(
    missing="impute",
    imputation_strategy="median",
    training_iter=1000,
    random_state=42,
)

model.fit(X_train_array, y_train.to_numpy())
print(model.missing_report_.to_dict())

candidate_report = model.summarize_prediction_missing_values(X_candidate_array)
print(candidate_report.to_dict())

y_candidate, y_candidate_std = model.predict(
    X_candidate_array,
    return_std=True,
)
```

Use `missing="error"` when missing values should fail loudly, `missing="drop"`
when incomplete training rows should be removed, and `missing="impute"` when
numeric feature values should be filled from the training fold. Targets are not
imputed; missing targets are rejected by `missing="error"` and dropped by
`missing="drop"` or `missing="impute"`. Prediction-time missing features are
supported by `missing="impute"` because the fitted training imputer can be
reused without leaking test or candidate-pool information.

Useful fitted attributes:

| Attribute | Meaning |
| --- | --- |
| `result_` | Full `GPyTorchGPRResult` object. |
| `model_`, `likelihood_` | Fitted GPyTorch objects. |
| `loss_history_` | Training loss by optimizer iteration. |
| `target_mean_`, `target_std_` | Target standardization values. |
| `missing_report_` | Fitted `MissingValueReport` describing rejected, dropped, or imputed training rows. |

For prediction-time audits, use
`model.summarize_prediction_missing_values(X_candidate)` before calling
`predict`. The method returns a `MissingValueReport` without changing estimator
state.

For confidence intervals:

```python
prediction = model.predict_distribution(X_test_array, confidence_level=0.95)
```

`prediction` contains `mean`, `std`, `lower`, and `upper`.

`MatGPRRegressor` follows the scikit-learn estimator interface and can be used
inside pipelines, grid searches, and column transformers. For example, combine
composition descriptors and process variables before fitting GPR:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matgpr import CompositionFeaturizer, MatGPRRegressor

features = ColumnTransformer(
    transformers=[
        (
            "composition",
            CompositionFeaturizer(
                properties=("atomic_number",),
                statistics=("fwm", "max"),
                return_dataframe=False,
            ),
            ["formula"],
        ),
        ("process", "passthrough", ["temperature_k", "load_n"]),
    ]
)

model = Pipeline(
    steps=[
        ("features", features),
        ("scale", StandardScaler()),
        ("gpr", MatGPRRegressor(training_iter=1000, random_state=42)),
    ]
)

model.fit(train_data, y_train)
y_test_pred, y_test_std = model.predict(test_data, return_std=True)
```

### 5.3 GPyTorch Exact GPR

Use GPyTorch GPR when you want the `PhysicsInformedMean` API, direct access to
training loss, target standardization, and flexible uncertainty handling.

```python
from sklearn.preprocessing import StandardScaler

from matgpr import fit_gpytorch_gpr

feature_columns = X_train.select_dtypes(include="number").columns.tolist()

scaler = StandardScaler()
X_train_array = scaler.fit_transform(X_train[feature_columns])
X_test_array = scaler.transform(X_test[feature_columns])

result = fit_gpytorch_gpr(
    X_train_array,
    y_train.to_numpy(),
    kernel="matern",
    ard=True,
    lr=0.03,
    training_iter=1000,
    initial_noise=0.05,
    standardize_y=True,
    verbose=True,
)

prediction = result.predict(X_test_array, confidence_level=0.95)
metrics = regression_metrics(y_test, prediction.mean)
```

`prediction` is a `GPyTorchPrediction` object:

| Attribute | Meaning |
| --- | --- |
| `mean` | Predictive mean in original target units. |
| `std` | Predictive standard deviation in original target units. |
| `lower`, `upper` | Confidence bounds when `confidence_level` is supplied. |

### 5.4 Multitask GPyTorch GPR

Use `MultitaskGPRRegressor` or `fit_multitask_gpytorch_gpr` when the same
material rows have multiple related target properties and every target is
observed for every row. Use `SparseMultitaskGPRRegressor` or
`fit_sparse_multitask_gpytorch_gpr` when the target matrix has `NaN` entries
for unobserved task values. Both forms learn a shared input-space kernel and a
task covariance, so correlated properties can share statistical strength:

```text
cov[f_i(x), f_j(x')] = k_x(x, x') k_task(i, j)
```

Example:

```python
from matgpr import MultitaskGPRRegressor

target_columns = ["strength_mpa", "ductility_percent"]

model = MultitaskGPRRegressor(
    task_names=target_columns,
    task_covar_rank=1,
    kernel="matern",
    training_iter=1000,
    verbose=False,
)

model.fit(X_train_array, train_data[target_columns])
prediction = model.predict_distribution(X_test_array, confidence_level=0.95)
```

The prediction arrays have shape `(n_samples, n_tasks)` in the same order as
`model.task_names_`. Report per-task metrics and uncertainty diagnostics rather
than only an averaged score:

```python
from matgpr import evaluate_multitask_train_test_split

validation = evaluate_multitask_train_test_split(
    model,
    X_array,
    data[target_columns],
    test_size=0.2,
    random_state=7,
    model_name="multitask_gpr",
)

validation.task_metrics
validation.predictions
```

`validation.task_metrics` contains one row per split and task with RMSE, MAE,
R2, Pearson \(r\), sample count, and uncertainty diagnostics when predictive
standard deviations are available.

For incomplete target matrices, keep unobserved values as `NaN`:

```python
sparse_model = SparseMultitaskGPRRegressor(
    task_names=target_columns,
    task_covar_rank=1,
    kernel="matern",
    noise_mode="task",
    initial_task_noises={name: 0.1 for name in target_columns},
    training_iter=1000,
    min_observations_per_task=2,
    verbose=False,
)

sparse_model.fit(X_train_array, train_data[target_columns])
sparse_prediction = sparse_model.predict_distribution(
    X_test_array,
    confidence_level=0.95,
)
sparse_model.task_observation_counts_
```

The sparse estimator preserves partially observed rows and learns one task
covariance across all finite target entries. Use `noise_mode="shared"` for one
global observation-noise variance, or `noise_mode="task"` for one learned noise
variance per target property. Use `noise_mode="known"` with
`known_noise_variance` when each observed entry has a reported or estimated
measurement variance in original target units. See
[Sparse Multitask Noise](sparse_multitask_noise_design.md) for details.

Example with known per-observation variances:

```python
known_noise_variance = reported_std[target_columns] ** 2
known_noise_variance = known_noise_variance.mask(train_data[target_columns].isna())

known_noise_model = SparseMultitaskGPRRegressor(
    task_names=target_columns,
    noise_mode="known",
    known_noise_variance=known_noise_variance,
    training_iter=1000,
    verbose=False,
)

known_noise_model.fit(X_train_array, train_data[target_columns])
```

Use the sparse validation helper when held-out target matrices also contain
unobserved values:

```python
from matgpr import evaluate_sparse_multitask_train_test_split

sparse_validation = evaluate_sparse_multitask_train_test_split(
    sparse_model,
    X_array,
    data[target_columns],
    test_size=0.2,
    random_state=7,
    model_name="sparse_multitask_gpr",
)

sparse_validation.task_metrics
sparse_validation.observed_predictions
```

Sparse metrics are calculated only on observed target entries. The full
prediction table keeps an `observed` column so users can separate parity-plot
rows from predictions made for unmeasured tasks. For a complete sparse
workflow, see the [Sparse Multitask Walkthrough](sparse_multitask_walkthrough.md).

### 5.5 Multi-Fidelity GPR

Use `MultiFidelityGPRRegressor` when low-fidelity data, such as simulations or
screening measurements, are available alongside scarce high-fidelity
measurements. The first `matgpr` multi-fidelity model learns:

```text
y_high(x) = rho * y_low(x) + intercept + delta(x)
```

where `delta(x)` is a GPR correction trained on high-fidelity residuals.

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

If low-fidelity values are not available at prediction points, fit an internal
low-fidelity surrogate:

```python
model.fit(
    X_high,
    y_high,
    X_low=X_simulation,
    y_low=y_simulation,
)

prediction = model.predict_distribution(X_test, confidence_level=0.95)
```

For joint co-kriging-ready datasets with row-wise fidelity labels, validate the
data table with `prepare_multifidelity_observations`:

```python
from matgpr import prepare_multifidelity_observations


observations = prepare_multifidelity_observations(
    X=X_all,
    y=y_all,
    fidelity=fidelity_labels,
    fidelity_order=("simulation", "experiment"),
    target_fidelity="experiment",
    sample_id=material_ids,
)

observations.fidelity_observation_counts
```

This container preserves explicit fidelity order, target fidelity, optional
sample identifiers, feature names, and known per-observation noise variances.
The delta model still accepts direct high-fidelity and low-fidelity arrays; the
observation container is the input contract for joint co-kriging models.

For the first two-level co-kriging API, use:

```python
from matgpr import CoKrigingGPRRegressor


cokriging_model = CoKrigingGPRRegressor(
    fidelity_order=("simulation", "experiment"),
    target_fidelity="experiment",
    training_iter=1000,
    random_state=7,
)

cokriging_model.fit(X_all, y_all, fidelity=fidelity_labels)
cokriging_prediction = cokriging_model.predict_distribution(
    X_test,
    confidence_level=0.95,
)
```

This initial co-kriging model supports exactly two fidelity levels, learns one
constant autoregressive coefficient `rho`, and uses one shared learned
observation-noise term. Known-noise and per-fidelity noise modes are planned
extensions.

For low-data studies, use `multifidelity_learning_curve` so the number of
high-fidelity training points is varied while the low-fidelity source is kept
fixed:

```python
from matgpr import multifidelity_learning_curve


lc_result = multifidelity_learning_curve(
    {"delta multi-fidelity GPR": model},
    X_high,
    y_high,
    low_fidelity_high=simulation_at_high_points,
    train_size_start=10,
    train_size_stop=100,
    train_size_step=10,
    n_splits=20,
    random_state=42,
)
```

Report validation metrics on held-out high-fidelity data and compare against a
high-fidelity-only GPR baseline. See [Multi-Fidelity GPR](multifidelity_gpr.md)
for assumptions and reporting guidance.

## 6. Physics-Aware Kernels

Physics can also enter through the covariance function. The kernel controls
which materials are considered similar. This is especially important for
molecular and polymer fingerprints, where Euclidean distance is often less
natural than overlap-based similarity, for inorganic compositions where the
amount of elemental substitution is physically meaningful, and for crystal
structures where lattice geometry and packing should affect similarity.

### 6.1 Tanimoto Kernel for Fingerprints

Use `TanimotoKernel` for binary or non-negative count fingerprints, such as
Morgan fingerprints from RDKit.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel

from matgpr import TanimotoKernel

kernel = (
    ConstantKernel(1.0)
    * TanimotoKernel()
    + WhiteKernel(noise_level=1.0)
)

model = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42,
)

model.fit(X_train_fingerprints, y_train)
y_test_pred, y_test_std = model.predict(X_test_fingerprints, return_std=True)
```

For convenience, `build_sklearn_gpr_kernel("tanimoto")` returns the same
Tanimoto-plus-noise structure.

### 6.2 Element-Fraction Kernel for Inorganic Compositions

Use `ElementFractionKernel` when rows are elemental composition vectors, such as
the output of `featurize_element_fractions` or `append_element_fractions`.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel

from matgpr import ElementFractionKernel

kernel = (
    ConstantKernel(1.0)
    * ElementFractionKernel(metric="l1", length_scale=1.0)
    + WhiteKernel(noise_level=1.0)
)

model = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42,
)

model.fit(X_train_element_fractions, y_train)
y_test_pred, y_test_std = model.predict(X_test_element_fractions, return_std=True)
```

The default `metric="l1"` compares the total fraction of elements that must be
substituted to move from one composition to another. Use `metric="l2"` when a
smoother Euclidean distance over the composition simplex is preferred.

For convenience, `build_sklearn_gpr_kernel("composition")` and
`build_sklearn_gpr_kernel("element_fraction")` return an element-fraction
composition kernel with white noise.

### 6.3 Structure Kernel for Crystal Geometry

Use `StructureFeatureKernel` for continuous structure descriptors, such as the
output of `featurize_structures`, `append_structure_fingerprints`, or
`StructureFeaturizer`.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel

from matgpr import StructureFeatureKernel

kernel = (
    ConstantKernel(1.0)
    * StructureFeatureKernel(metric="l2", length_scale=1.0)
    + WhiteKernel(noise_level=1.0)
)

model = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42,
)

model.fit(X_train_structure_features, y_train)
y_test_pred, y_test_std = model.predict(X_test_structure_features, return_std=True)
```

The default `metric="l2"` gives an RBF kernel over global structure descriptors.
Use `feature_scales` or a scikit-learn scaler when structure features have very
different units or ranges.

For convenience, `build_sklearn_gpr_kernel("structure")` returns a
structure-feature kernel with white noise.

### 6.4 Mixed Fingerprint and Physics Kernels

For mixed feature matrices, use `FeatureSubsetKernel` to apply different kernels
to different column groups. This lets a fingerprint kernel act on molecular bits
while a continuous kernel acts on physics descriptors.

```python
from sklearn.gaussian_process.kernels import RBF

from matgpr import FeatureSubsetKernel, TanimotoKernel, build_additive_kernel

fingerprint_columns = list(range(0, 2048))
physics_columns = [2048, 2049]

kernel = build_additive_kernel(
    FeatureSubsetKernel(TanimotoKernel(), columns=fingerprint_columns),
    FeatureSubsetKernel(RBF(length_scale=1.0), columns=physics_columns),
)
```

Use additive kernels when different feature blocks contribute complementary
effects. Use product kernels when similarity should be high only when both
feature blocks are similar.

## 7. Target Transforms and Physics Residuals

Some materials properties are easier to model after a target transformation.
Common examples include log diffusivity, log conductivity, Arrhenius-linearized
rates, bounded efficiencies, and residuals relative to a simple physics
baseline.

`matgpr` target transforms expose a small common interface:

- `fit(y)` estimates transform parameters when needed.
- `transform(y)` maps observed targets into model space.
- `fit_transform(y)` does both steps.
- `inverse_transform(y_transformed)` maps point values back to original units.
- `inverse_prediction(prediction, ...)` maps predictive means, standard
  deviations, and confidence bounds back to original units.

### 7.1 Log Targets

Use `LogTargetTransform` for positive targets such as diffusivity,
conductivity, rate constants, and permeability-like properties. Predictive
standard deviations are inverted using log-normal moments.

```python
from matgpr import LogTargetTransform, fit_gpytorch_gpr

target_transform = LogTargetTransform(offset=0.0)
y_train_model = target_transform.fit_transform(y_train)

result = fit_gpytorch_gpr(
    X_train_array,
    y_train_model,
    kernel="matern",
    ard=True,
    standardize_y=True,
)

prediction_model = result.predict(X_test_array, confidence_level=0.95)
prediction = target_transform.inverse_prediction(prediction_model)
```

### 7.2 Bounded Targets

Use `BoundedTargetTransform` for targets with known finite physical limits,
such as efficiencies, fractions, probabilities, normalized phase fractions, or
bounded scores. The transform maps the target interval to an unconstrained
logit scale before fitting GPR.

```python
from matgpr import BoundedTargetTransform, fit_gpytorch_gpr

target_transform = BoundedTargetTransform(
    lower_bound=0.0,
    upper_bound=100.0,
)
y_train_model = target_transform.fit_transform(y_train)

result = fit_gpytorch_gpr(
    X_train_array,
    y_train_model,
    kernel="matern",
    ard=True,
)

prediction_model = result.predict(X_test_array, confidence_level=0.95)
prediction = target_transform.inverse_prediction(prediction_model)
```

`inverse_prediction` converts predictive intervals by applying the inverse
logit transform to the interval bounds. Predictive means and standard
deviations are estimated with Gauss-Hermite quadrature over the
logistic-normal distribution, so uncertainty remains on the original bounded
scale.

### 7.3 Materials-Property Presets

For common materials targets, use a documented preset instead of choosing the
transform from memory:

```python
from matgpr import (
    make_materials_target_transform,
    search_target_transform_specs,
    summarize_target_transform_specs,
)

summary = summarize_target_transform_specs()
transport_presets = search_target_transform_specs(tag="transport")

target_transform = make_materials_target_transform("diffusivity")
y_train_model = target_transform.fit_transform(y_train)
```

Useful presets include:

| Preset | Transform | Typical use |
| --- | --- | --- |
| `efficiency_percent` | bounded 0 to 100 | OPV PCE, conversion percent, yield percent |
| `fraction` | bounded 0 to 1 | phase fraction, volume fraction, probability |
| `diffusivity` | log | solvent, ion, or gas diffusion coefficients |
| `permeability` | log | gas and membrane permeability |
| `conductivity` | log | electrical, ionic, and thermal conductivity |
| `band_gap_ev` | log with small offset | nonnegative electronic band gaps |
| `energy_above_hull` | log with small offset | nonnegative stability above the convex hull |
| `modulus`, `strength`, `hardness` | log | positive mechanical properties |
| `formation_energy`, `binding_energy` | standardize | signed energy targets |

Aliases are supported, so `make_materials_target_transform("pce")` returns the
same bounded transform as `efficiency_percent`, and
`make_materials_target_transform("diffusion-coefficient")` returns the
`diffusivity` preset.

Preset defaults can be overridden when the dataset has a narrower known domain:

```python
target_transform = make_materials_target_transform(
    "efficiency_percent",
    upper_bound=35.0,
)
```

The presets are not automatic truth. They are documented defaults. Always
check the target units, sign convention, possible zero values, and physical
domain before choosing a transform.

### 7.4 Standardized Targets

`fit_gpytorch_gpr(..., standardize_y=True)` already standardizes internally.
Use `StandardizedTargetTransform` when you need explicit control outside the
model, for example when comparing multiple estimators with the same transformed
target.

```python
target_transform = StandardizedTargetTransform()
y_train_model = target_transform.fit_transform(y_train)

result = fit_gpytorch_gpr(
    X_train_array,
    y_train_model,
    standardize_y=False,
)

prediction_model = result.predict(X_test_array, confidence_level=0.95)
prediction = target_transform.inverse_prediction(prediction_model)
```

### 7.5 Physics Residual Modeling

Physics residual modeling is a simple alternative to changing the GP mean
function. A baseline equation predicts the coarse physical trend, and GPR learns
the residual:

```text
residual = measured_property - physics_baseline
```

```python
from matgpr import PhysicsResidualTransform

target_transform = PhysicsResidualTransform()

baseline_train = physics_baseline(train_data)
baseline_test = physics_baseline(test_data)

y_train_residual = target_transform.fit_transform(
    y_train,
    baseline=baseline_train,
)

result = fit_gpytorch_gpr(
    X_train_array,
    y_train_residual,
    kernel="matern",
    ard=True,
)

residual_prediction = result.predict(X_test_array, confidence_level=0.95)
prediction = target_transform.inverse_prediction(
    residual_prediction,
    baseline=baseline_test,
)
```

This approach is useful when the physics baseline is not differentiable, lives
outside PyTorch, comes from a lookup table, or should remain completely fixed.

## 8. Physics Constraints With Virtual Observations

Some physics knowledge is easier to express as soft anchor observations than as
a GP mean function. Examples include a known zero-response limit, a saturation
value, or a local monotonic trend with respect to temperature, time, loading, or
composition.

`matgpr` provides virtual-observation utilities for this case. They augment the
training set with physics-derived rows:

```text
X_aug = [X_observed, X_virtual]
y_aug = [y_observed, y_virtual]
```

For a known limit in feature column `j`:

```text
x_virtual = x_reference with x_j = x_limit
y_virtual = y_limit(x_virtual)
```

For a monotonic trend:

```text
x_virtual = x_reference + delta e_j
y_virtual = y_reference + direction * minimum_slope * delta
```

where `direction` is `+1` for increasing trends and `-1` for decreasing trends.
The `noise_std` assigned to virtual observations controls how strongly the
anchor influences the model.

```python
from matgpr import (
    KnownLimitConstraint,
    MonotonicTrendConstraint,
    append_virtual_observations,
    build_sklearn_gpr_model,
)

zero_time = KnownLimitConstraint(
    feature="time_s",
    limit_value=0.0,
    target_value=0.0,
    noise_std=0.02,
    label="zero_time_limit",
)

temperature_trend = MonotonicTrendConstraint(
    feature="temperature_k",
    direction="increasing",
    step=25.0,
    minimum_slope=0.0,
    feature_max=1000.0,
    noise_std=0.2,
)

limit_observations = zero_time.generate(X_train)
trend_observations = temperature_trend.generate(X_train, y_train)

augmented = append_virtual_observations(
    X_train,
    y_train,
    limit_observations,
    trend_observations,
    base_alpha=1e-8,
)

model = build_sklearn_gpr_model(
    n_features=augmented.X.shape[1],
    alpha=augmented.alpha,
)
model.fit(augmented.X, augmented.y)
```

For scikit-learn GPR, pass `augmented.alpha` to use one observation variance
per row. This lets real observations and virtual physics anchors have different
noise levels. For GPyTorch workflows, the augmented `X` and `y` can still be
used directly, but the current exact-GPR helper uses one learned Gaussian noise
level for all rows.

Use virtual observations carefully:

- Keep anchor values physically defensible.
- Use a larger `noise_std` when the constraint is approximate.
- Report how many virtual observations were added.
- Validate against held-out experimental data, not only augmented training
  error.
- Do not claim global monotonicity unless using a formal derivative-constrained
  GP.

### 8.1 Derivative-Constrained GPR

When physics gives slope information, use derivative-constrained GPR instead
of converting the slope into ordinary function-value anchors. In this model,
the GP is trained on both function observations and derivative observations:

```text
y_i = f(x_i) + epsilon_i
g_m = df(z_m) / dz_{q_m} + eta_m
```

For an RBF kernel,

```text
k(x, x') = sigma_f^2 exp(-0.5 sum_d ((x_d - x'_d) / ell_d)^2)
```

`matgpr` uses the exact joint covariance between function values and
derivatives:

```text
cov[f(x), df(z)/dz_j] =
    k(x, z) (x_j - z_j) / ell_j^2

cov[df(x)/dx_i, df(z)/dz_j] =
    k(x, z) [1(i = j) / ell_i^2
             - (x_i - z_i)(x_j - z_j) / (ell_i^2 ell_j^2)]
```

This is useful for materials trends such as increasing diffusivity with
temperature, decreasing viscosity with temperature, positive time dependence,
or a known near-zero derivative in a saturation regime.

```python
from matgpr import (
    MonotonicDerivativeConstraint,
    fit_derivative_constrained_gpr,
)

temperature_slope = MonotonicDerivativeConstraint(
    feature="temperature_k",
    direction="increasing",
    minimum_slope=0.01,
    noise_std=0.1,
)

derivative_observations = temperature_slope.generate(X_train)

result = fit_derivative_constrained_gpr(
    X_train,
    y_train,
    derivative_observations,
    length_scale=None,
    signal_variance=1.0,
    value_noise_std=0.05,
    standardize_y=True,
    optimize_hyperparameters=True,
)

prediction = result.predict(X_test, confidence_level=0.95)
```

You can also pass measured or equation-derived slopes directly:

```python
from matgpr import DerivativeObservationSet

derivative_observations = DerivativeObservationSet(
    X=slope_anchor_features,
    feature_indices=feature_columns.index("temperature_k"),
    derivative_values=d_property_d_temperature,
    noise_std=0.2,
)
```

Important details:

- Derivatives must be with respect to the same feature scale used in `X`.
- If `X` is standardized, transform derivative values to that standardized
  feature scale before fitting. For `x_scaled = (x - mean) / scale`, use
  `dy / dx_scaled = scale * dy / dx`.
- `noise_std` controls how strongly derivative observations influence the
  posterior.
- Monotonic derivative observations are soft equality observations of the
  local slope; they encourage a trend near anchor points, but they still do not
  prove global monotonicity everywhere.

### 8.2 Physics-Aware Noise Models

Published materials datasets often mix measurements from different papers,
instruments, simulation levels, replicate batches, or experimental regimes.
Treating every row as equally noisy can make GPR over-trust uncertain sources.

Physics-aware noise profiles define one observation noise standard deviation
per row:

```text
y_i = f(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma_i^2)
alpha_i = sigma_i^2
```

`matgpr` stores these values in `ObservationNoiseProfile`. Use
`profile.alpha` with scikit-learn GPR and `profile.noise_std` with
`fit_derivative_constrained_gpr`.

For source-dependent noise:

```python
from matgpr import SourceNoiseModel

source_noise = SourceNoiseModel(
    source_noise_std={
        "experiment": 0.05,
        "simulation": 0.20,
        "literature_estimate": 0.50,
    },
    default_noise_std=0.30,
    unknown="default",
)

source_profile = source_noise.profile(train_data["data_source"])
```

For replicate-aware noise:

```python
from matgpr import ReplicateNoiseModel

replicate_noise = ReplicateNoiseModel(min_noise_std=0.02)
replicate_profile = replicate_noise.fit_profile(
    train_data["target_property"],
    train_data["sample_id"],
)
```

For feature-dependent heteroscedastic noise:

```python
from matgpr import FeatureNoiseModel

feature_noise = FeatureNoiseModel(
    noise_std_function=lambda X: 0.02 + 0.0001 * np.maximum(X[:, temperature_col] - 300.0, 0.0),
    label="temperature_noise",
)

feature_profile = feature_noise.profile(X_train_array)
```

Independent noise components can be combined in quadrature:

```python
from matgpr import combine_noise_profiles, build_sklearn_gpr_model

noise_profile = combine_noise_profiles(
    source_profile,
    replicate_profile,
    feature_profile,
)

model = build_sklearn_gpr_model(
    n_features=X_train_array.shape[1],
    alpha=noise_profile.alpha,
)
model.fit(X_train_array, y_train)
```

For derivative-constrained GPR:

```python
result = fit_derivative_constrained_gpr(
    X_train_array,
    y_train,
    derivative_observations,
    value_noise_std=noise_profile.noise_std,
)
```

Guidance:

- Use source noise when rows come from different papers, instruments, or
  simulation levels.
- Use replicate noise when repeated measurements exist for the same material
  or condition.
- Use feature noise when uncertainty is known to grow in a physical regime,
  such as high temperature, high load, low signal, or extreme composition.
- Keep noise values in target units.
- Report the assumed noise model when publishing model results.

### 8.3 Learned Heteroscedastic GPR

Use `fit_heteroscedastic_gpr` when the observation noise is not known in
advance but appears to vary across the materials space. The current
implementation is a practical two-stage model:

```text
y_i = f_signal(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma_noise^2(x_i))

f_signal(x) ~ GP(m(x), k_signal(x, x'))
log(sigma_noise^2(x)) ~ GP(m_noise(x), k_noise(x, x'))
```

The signal GP is fit first. Then a second GP is fit to
`log(residual^2 + residual_variance_floor)`. At prediction time, `matgpr`
returns the signal mean, latent signal uncertainty, learned noise uncertainty,
and total uncertainty:

```text
sigma_total^2(x) = sigma_latent^2(x) + sigma_noise^2(x)
```

Example:

```python
from matgpr import fit_heteroscedastic_gpr

result = fit_heteroscedastic_gpr(
    X_train_array,
    y_train,
    signal_kernel="matern",
    noise_kernel="matern",
    signal_training_iter=1000,
    noise_training_iter=500,
    residual_variance_floor=1e-8,
    verbose=False,
)

prediction = result.predict(X_test_array, confidence_level=0.95)

y_mean = prediction.mean
y_total_std = prediction.std
y_signal_std = prediction.latent_std
y_noise_std = prediction.noise_std
```

This can also be combined with a physics-informed mean function:

```python
result = fit_heteroscedastic_gpr(
    X_train_array,
    y_train,
    signal_mean_module=physics_mean_function,
    signal_training_iter=1000,
    noise_training_iter=500,
    verbose=False,
)
```

Use learned heteroscedastic GPR when:

- measurement quality changes with composition, temperature, concentration, or
  processing condition,
- some regions of descriptor space are systematically harder to predict,
- uncertainty calibration is important for Bayesian optimization or candidate
  prioritization,
- no reliable per-row experimental uncertainty is available.

Use explicit `SourceNoiseModel`, `ReplicateNoiseModel`, or `FeatureNoiseModel`
instead when uncertainty estimates are known from metadata, replicate
measurements, or a trusted physical noise equation.

Report that this is a residual-noise GP approximation, not a full joint
variational heteroscedastic likelihood. Also report the residual variance floor,
signal kernel, noise kernel, and train/test protocol.

## 9. Physics-Informed GPR

Physics-informed GPR introduces physics through the GP mean function. The GP
then learns the residual between the physics equation and the observed data.

Conceptually:

```text
y_i = m_phys(x_i; theta) + f_residual(x_i) + epsilon_i

f_residual(x) ~ GP(0, k(x, x'))
epsilon_i ~ Normal(0, sigma_n^2)
```

where:

| Symbol | Meaning |
| --- | --- |
| `y_i` | Observed target value. |
| `m_phys(x_i; theta)` | Physics-informed mean equation. |
| `theta` | Fixed or learnable physical parameters. |
| `f_residual(x_i)` | Data-driven GP residual. |
| `k(x, x')` | Kernel covariance over the full feature space. |
| `sigma_n^2` | Learned observation noise. |

### 9.1 How Physics Is Introduced

In `matgpr`, physics is introduced with `PhysicsInformedMean`.

The user supplies:

1. A Python equation.
2. A mapping from physics feature names to feature-column indices.
3. Learnable physical parameters and initial values.
4. Optional fixed physical constants.
5. Optional positive constraints for parameters that must remain positive.
6. Optional feature means and standard deviations so equations can be written
   in original physical units even when the model input is scaled.

The equation has this signature:

```python
def equation(features, parameters):
    ...
    return predicted_mean
```

`features` is a dictionary of tensors. `parameters` is a dictionary of tensors.
The equation must return one mean value per sample.

For high-level workflows, `PhysicsInformedGPRRegressor` builds the
`PhysicsInformedMean` internally and exposes `fit`, `predict`, `score`,
`get_params`, and `set_params`.

```python
from matgpr import PhysicsInformedGPRRegressor


def physics_mean(features, parameters):
    return parameters["offset"] + parameters["slope"] * features["physics_descriptor"]


model = PhysicsInformedGPRRegressor(
    equation=physics_mean,
    feature_indices={"physics_descriptor": 0},
    learnable_parameters={"offset": 0.0, "slope": 1.0},
    positive_parameters=("slope",),
    training_iter=1000,
    random_state=42,
)

model.fit(X_train_scaled, y_train)
y_test_pred, y_test_std = model.predict(X_test_scaled, return_std=True)
learned_parameters = model.learned_physics_parameters_
```

### 9.2 Example 1: Arrhenius Mean Function

For a temperature-dependent transport property:

```text
log10(D) = log10(D0) - Ea / (ln(10) R T)
```

where:

| Parameter | Meaning | Learned? |
| --- | --- | --- |
| `log_d0` | Pre-exponential term in log10 units. | Yes |
| `activation_energy_kj_mol` | Activation energy. | Yes, positive |
| `R` | Gas constant. | Fixed in equation |

Code:

```python
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from matgpr import PhysicsInformedMean, fit_gpytorch_gpr

def arrhenius_mean(features, parameters):
    gas_constant = 8.314462618
    temperature = torch.clamp(features["temperature_k"], min=1.0)
    return parameters["log_d0"] - (
        parameters["activation_energy_kj_mol"] * 1000.0
    ) / (np.log(10.0) * gas_constant * temperature)


feature_columns = fingerprint_columns + ["temperature_k"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[feature_columns])
X_test_scaled = scaler.transform(test_data[feature_columns])

feature_means = dict(zip(feature_columns, scaler.mean_))
feature_stds = dict(zip(feature_columns, scaler.scale_))

mean_module = PhysicsInformedMean(
    equation=arrhenius_mean,
    feature_indices={"temperature_k": feature_columns.index("temperature_k")},
    learnable_parameters={
        "log_d0": float(np.percentile(train_data["log10_diffusivity"], 75)),
        "activation_energy_kj_mol": 10.0,
    },
    positive_parameters=("activation_energy_kj_mol",),
    feature_means={"temperature_k": feature_means["temperature_k"]},
    feature_stds={"temperature_k": feature_stds["temperature_k"]},
)

result = fit_gpytorch_gpr(
    X_train_scaled,
    train_data["log10_diffusivity"].to_numpy(),
    mean_module=mean_module,
    kernel="matern",
    ard=True,
    standardize_y=True,
    training_iter=1000,
)

prediction = result.predict(X_test_scaled, confidence_level=0.95)
learned_parameters = result.model.mean_module.current_parameter_values()
```

Why pass `feature_means` and `feature_stds`? The model receives scaled
features, but the Arrhenius equation should use temperature in Kelvin.
`PhysicsInformedMean` converts selected scaled columns back to original units
before evaluating the equation.

### 9.3 Example 2: Degeneracy and Binding Mean Function

For OPV-like materials, a simple physics-informed mean can encode the idea that
larger degeneracy can improve accessible pathways, while stronger binding can
penalize charge separation:

```text
PCE = beta0 + beta_deg log(1 + degeneracy) - beta_bind binding_energy
```

where:

| Parameter | Meaning | Learned? |
| --- | --- | --- |
| `beta0` | Baseline efficiency. | Yes |
| `beta_deg` | Strength of degeneracy benefit. | Yes, positive |
| `beta_bind` | Strength of binding penalty. | Yes, positive |

Code:

```python
def degeneracy_binding_mean(features, parameters):
    degeneracy = torch.clamp(features["degeneracy"], min=0.0)
    binding = features["binding_energy"]
    return (
        parameters["beta0"]
        + parameters["beta_deg"] * torch.log1p(degeneracy)
        - parameters["beta_bind"] * binding
    )


physics_features = ["degeneracy", "binding_energy"]
feature_columns = descriptor_columns + physics_features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[feature_columns])
X_test_scaled = scaler.transform(test_data[feature_columns])

feature_means = dict(zip(feature_columns, scaler.mean_))
feature_stds = dict(zip(feature_columns, scaler.scale_))

mean_module = PhysicsInformedMean(
    equation=degeneracy_binding_mean,
    feature_indices={name: feature_columns.index(name) for name in physics_features},
    learnable_parameters={
        "beta0": float(np.median(train_data["pce"])),
        "beta_deg": 1.0,
        "beta_bind": 1.0,
    },
    positive_parameters=("beta_deg", "beta_bind"),
    feature_means={name: feature_means[name] for name in physics_features},
    feature_stds={name: feature_stds[name] for name in physics_features},
)

result = fit_gpytorch_gpr(
    X_train_scaled,
    train_data["pce"].to_numpy(),
    mean_module=mean_module,
    kernel="matern",
    ard=False,
    training_iter=1000,
)
```

### 9.4 Reusable Physics Equation Templates

For common materials trends, `matgpr` includes reusable equation templates. A
template stores:

- the equation callable,
- the required physics feature names,
- default learnable parameter initial values,
- positive-parameter constraints,
- fixed constants such as the gas constant,
- short documentation for reporting.

Available templates include:

| Template | Equation form | Typical use |
| --- | --- | --- |
| `arrhenius_rate` | \(A \exp(-E_a / RT)\) | Diffusion, conductivity, reaction or transport rates |
| `arrhenius_sqrt_time` | \(b + \sqrt{A \exp(-E_a / RT)t}\) | Oxidation, diffusion depth, aging |
| `power_law` | \(b + c x^n\) | Load, rate, concentration, or field scaling |
| `hall_petch` | \(\sigma_0 + k d^{-1/2}\) | Strength, hardness, yield stress with grain size |
| `free_volume_exponential` | \(b + A \exp(-B/f_v)\) | Polymer diffusion, permeability, mobility |
| `rule_of_mixtures` | \((1-\phi)y_m + \phi y_i + \gamma\phi(1-\phi)\) | Composites, alloys, blends |

Use `get_physics_equation_template` when you want a documented starting point:

```python
from matgpr import (
    describe_physics_equation_template,
    get_physics_equation_template,
    search_physics_equation_templates,
    summarize_physics_equation_templates,
    fit_gpytorch_gpr,
)


template_table = summarize_physics_equation_templates()
transport_templates = search_physics_equation_templates(query="transport")
template = get_physics_equation_template("arrhenius_rate")
metadata = describe_physics_equation_template("arrhenius_rate")

feature_metadata = template.feature_specs()
parameter_metadata = template.parameter_specs()
assumptions = metadata["assumptions"]

mean_module = template.build_mean_function(
    feature_indices={"temperature_k": feature_columns.index("temperature_k")},
    learnable_parameter_overrides={
        "prefactor": 1.0,
        "activation_energy": 25_000.0,
    },
    feature_means={"temperature_k": feature_means["temperature_k"]},
    feature_stds={"temperature_k": feature_stds["temperature_k"]},
)

result = fit_gpytorch_gpr(
    X_train_scaled,
    y_train,
    mean_module=mean_module,
    kernel="matern",
    ard=True,
    training_iter=1000,
)

learned_parameters = result.model.mean_module.current_parameter_values()
```

The template feature names are canonical names used inside the equation. They
do not need to match dataframe column names. For example, a hardness notebook
can map an experimental load column to the `power_law` template:

```python
template = get_physics_equation_template("power_law")

mean_module = template.build_mean_function(
    feature_indices={"driving_variable": feature_columns.index("load_n")},
    learnable_parameter_overrides={
        "coefficient": 1.0,
        "exponent": -0.1,
        "offset": float(y_train.mean()),
    },
)
```

Use the metadata when writing reports. It states which features are required,
their expected units, which parameters are learned during GPR training, which
constants are fixed, and which assumptions justify using the equation.

The templates are not proof that a system obeys the equation. Treat them as
transparent inductive biases, then validate them against standard GPR with
learning curves, held-out parity plots, uncertainty calibration, and learned
parameter checks.

### 9.5 Learned vs Fixed Parameters

Use `learnable_parameters` when a parameter should be optimized from data:

```python
learnable_parameters={"slope": 1.0, "intercept": 0.0}
```

Use `positive_parameters` when a learned parameter must stay positive:

```python
positive_parameters=("slope",)
```

Use `fixed_parameters` when a constant should be passed to the equation but not
optimized:

```python
mean_module = PhysicsInformedMean(
    equation=my_equation,
    feature_indices={"temperature_k": 0},
    learnable_parameters={"activation_energy": 50.0},
    positive_parameters=("activation_energy",),
    fixed_parameters={"gas_constant": 8.314462618},
)
```

Inside the equation:

```python
def my_equation(features, parameters):
    R = parameters["gas_constant"]
    ...
```

After training, inspect learned values:

```python
result.model.mean_module.current_parameter_values()
```

### 9.6 Practical Guidance for Physics Equations

Good physics-informed mean functions should be:

- Simple enough to be explainable.
- Written in the same units as the target.
- Based on columns available at prediction time.
- Helpful in low-data regimes.
- Flexible enough that the GP residual can correct imperfect physics.

Start with one or two physics terms. Add complexity only when learning curves,
test-set metrics, and uncertainty calibration improve.

## 10. Model Analysis

### 10.1 Reusable Validation Workflows

Use the validation helpers when you want notebook workflows to become
reproducible package-level analysis.

For one held-out split:

```python
from matgpr import evaluate_train_test_split


validation = evaluate_train_test_split(
    model,
    X,
    y,
    test_size=0.10,
    random_state=42,
    model_name="physics-informed GPR",
)

metrics = validation.metrics_frame()
predictions = validation.predictions
```

`predictions` contains `split`, `sample_position`, `sample_label`, `y_true`,
`y_pred`, and optional `y_std`, so it can be passed directly to parity-plot
code.

For 10-fold cross-validation:

```python
from matgpr import cross_validate_regressor


cv_result = cross_validate_regressor(
    model,
    X_train,
    y_train,
    cv=10,
    random_state=43,
    model_name="physics-informed GPR",
)

fold_metrics = cv_result.fold_metrics
cv_summary = cv_result.summary(metric_columns=["test_RMSE", "test_R2", "test_r"])
out_of_fold_predictions = cv_result.predictions
```

For learning curves:

```python
from matgpr import learning_curve


lc_result = learning_curve(
    {
        "standard GPR": standard_model,
        "physics-informed GPR": physics_model,
    },
    X,
    y,
    train_size_start=10,
    train_size_stop=100,
    train_size_step=10,
    train_size_unit="percent",
    n_splits=20,
    test_size=0.30,
    random_state=44,
    metrics=("RMSE", "R2", "MAE", "r"),
    metric_splits=("test",),
)

learning_curve_rows = lc_result.runs
learning_curve_summary = lc_result.summary(metrics=("RMSE", "R2"), splits="test")
```

These utilities clone the estimator for each split, return train/test metrics,
and include uncertainty diagnostics when the estimator supports
`predict(..., return_std=True)`.

Use `metric_splits=("train", "test")` when train and held-out learning curves
should be compared. The result table always includes `train_size`,
`train_size_percent`, and `n_train`; choose the displayed x-axis in
`plot_learning_curve(...)`.

For multi-fidelity GPR, use `multifidelity_learning_curve(...)` with
`low_fidelity_high=...` or `X_low=..., y_low=...`. It returns the same
`LearningCurveResult` shape as `learning_curve(...)`, plus fitted fidelity-map
columns such as `rho` and `intercept` and component prediction columns when
`store_predictions=True`.

Use the reporting helpers to explain how much of each prediction comes from the
scaled low-fidelity source versus the learned high-fidelity correction:

```python
from matgpr import decompose_multifidelity_prediction, summarize_multifidelity_components


component_rows = decompose_multifidelity_prediction(
    prediction,
    y_true=y_test,
    sample_labels=test_sample_ids,
    model_name="delta multi-fidelity GPR",
    split="test",
)
component_summary = summarize_multifidelity_components(component_rows)

# For learning-curve predictions:
learning_curve_component_summary = summarize_multifidelity_components(
    lc_result.predictions,
    group_by=("model", "split", "train_size_percent"),
)
```

### 10.2 Regression Metrics

```python
metrics = regression_metrics(y_test, prediction.mean)
print(metrics)
```

Returned metrics:

| Metric | Meaning |
| --- | --- |
| `R2` | Coefficient of determination. Higher is better. |
| `RMSE` | Root mean squared error in target units. Lower is better. |
| `MAE` | Mean absolute error in target units. Lower is better. |
| `r` | Pearson correlation coefficient. Higher absolute value indicates stronger linear association. |

For train/test summaries:

```python
metrics = train_test_regression_metrics(
    y_train,
    train_prediction.mean,
    y_test,
    test_prediction.mean,
)
```

### 10.3 Parity Plot With Uncertainty

```python
fig, ax = plot_parity(
    y_train,
    train_prediction.mean,
    y_train_std=train_prediction.std,
    y_test_true=y_test,
    y_test_pred=test_prediction.mean,
    y_test_std=test_prediction.std,
    title="GPR parity plot",
    xlabel="Measured property",
    ylabel="Predicted property",
    save_path="figures/parity.png",
)
```

### 10.4 Uncertainty Diagnostics

GPR models return a predictive mean and standard deviation. Use the standard
regression metrics for point-prediction accuracy, then separately check whether
the predictive uncertainty is calibrated and useful.

```python
diagnostics = uncertainty_diagnostics(
    y_test,
    test_prediction.mean,
    test_prediction.std,
    confidence_level=0.95,
)
print(diagnostics)
```

Important uncertainty diagnostics:

| Diagnostic | Meaning |
| --- | --- |
| `observed_coverage` | Fraction of points inside the requested Gaussian prediction interval. |
| `coverage_error` | Observed coverage minus expected coverage. Values near zero are preferred. |
| `NLPD` | Gaussian negative log predictive density. Lower is better. |
| `mean_standardized_residual` | Average residual divided by predictive standard deviation. Values near zero suggest low bias. |
| `std_standardized_residual` | Spread of standardized residuals. Values near one suggest calibrated standard deviations. |
| `uncertainty_error_spearman` | Rank correlation between predictive standard deviation and absolute error. Positive values mean larger uncertainties tend to flag larger errors. |

For calibration plots, compare observed interval coverage against expected
coverage across several confidence levels:

```python
fig, ax, curve = plot_uncertainty_calibration(
    y_test,
    test_prediction.mean,
    test_prediction.std,
    save_path="figures/uncertainty_calibration.png",
)
```

To check whether high-uncertainty predictions are also high-error predictions:

```python
fig, ax, diagnostics = plot_uncertainty_vs_error(
    y_test,
    test_prediction.mean,
    test_prediction.std,
    save_path="figures/uncertainty_vs_error.png",
)
```

For tabular workflows, `calibration_curve(...)` returns the same coverage
information as a dataframe, which is useful for reports and benchmark tables.

### 10.5 Learning Curves

For train-size experiments, collect rows with columns such as `train_size`,
`n_train`, `model`, and `test_R2`.

```python
fig, ax, summary = plot_learning_curve(
    lc_result.runs,
    metric="RMSE",
    split="test",
    x_axis="percent",
    title="Test RMSE learning curve",
)
```

To show train and test curves together, request both splits:

```python
fig, ax, summary = plot_learning_curve(
    lc_result.runs,
    metric="RMSE",
    split=("train", "test"),
    x_axis="count",
    title="Train/test RMSE learning curve",
)
```

Use `metric="RMSE"`, `metric="R2"`, `metric="MAE"`, or `metric="r"` to choose
the plotted metric. Use `metric_col="test_RMSE"` or another explicit column
name when plotting custom metrics.

### 10.6 90/10 Validation With 10-Fold Cross-Validation

After selecting the best model family from learning curves, use a conventional
validation protocol before fitting the final production model:

1. Split the full dataset into 90 percent training and 10 percent held-out test.
2. Run 10-fold cross-validation inside the 90 percent training partition.
3. Refit the selected model on the full 90 percent training partition.
4. Plot cross-validation statistics and train/test parity with uncertainty bars.

The reusable validation utilities keep this workflow compact:

```python
from sklearn.model_selection import train_test_split
from matgpr import cross_validate_regressor, evaluate_train_test_split


train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.10,
    random_state=42,
)

cv_result = cross_validate_regressor(
    model,
    X.iloc[train_idx],
    y.iloc[train_idx],
    cv=10,
    random_state=43,
)

validation = evaluate_train_test_split(
    model,
    X,
    y,
    train_indices=train_idx,
    test_indices=test_idx,
)

cv_summary = cv_result.summary(metric_columns=["test_RMSE", "test_R2", "test_r"])
parity_predictions = validation.predictions
```

For very small datasets, use a smaller `cv` value if each fold would contain
too few validation samples.

### 10.7 PCA

```python
from matgpr import fit_pca, summarize_pca, transform_pca, plot_pca_scree, plot_pca_scores

train_scores, pca, pca_scaler = fit_pca(X_train_numeric, n_components=5, scale=True)
test_scores = transform_pca(X_test_numeric, pca, scaler=pca_scaler)

pca_summary = summarize_pca(pca)
plot_pca_scree(pca)
plot_pca_scores(train_scores, test_scores=test_scores)
```

### 10.8 SHAP Analysis

`matgpr` does not require SHAP internally, but the example notebooks use SHAP
for production-model interpretation. For compact tabular datasets, run SHAP on
all selected model features. For large fingerprint datasets, use a tractable
candidate set such as physics features, descriptor columns, and the most
target-correlated fingerprint bits.

```python
import shap

selected_feature_columns = feature_columns
SHAP_FEATURES = selected_feature_columns[:30]
reference_values = model_data[feature_columns].median(numeric_only=True)
shap_background = shap.sample(model_data[SHAP_FEATURES], 40, random_state=42)
shap_explain = shap.sample(model_data[SHAP_FEATURES], 60, random_state=43)

def predict_for_shap(shap_frame):
    shap_frame = pd.DataFrame(shap_frame, columns=SHAP_FEATURES)
    full_frame = pd.DataFrame(
        np.repeat(reference_values.to_numpy()[None, :], len(shap_frame), axis=0),
        columns=feature_columns,
    )
    full_frame.loc[:, SHAP_FEATURES] = shap_frame.to_numpy()
    X_scaled = scaler.transform(full_frame[feature_columns])
    return result.predict(X_scaled, return_std=False).mean

explainer = shap.PermutationExplainer(predict_for_shap, shap_background)
shap_values = explainer(
    shap_explain,
    max_evals=2 * len(SHAP_FEATURES) + 1,
    batch_size=16,
)

shap_importance = pd.DataFrame(
    {
        "feature": SHAP_FEATURES,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }
).sort_values("mean_abs_shap", ascending=False)
```

For GPR, SHAP can be computationally expensive. For publication workflows,
combine SHAP with domain checks, correlation analysis, and sensitivity plots.

## 11. Bayesian Optimization

After validating a GPR model, use Bayesian optimization when the goal is to
choose the next material, molecule, polymer, formulation, or experiment from a
finite candidate pool. The first `matgpr` Bayesian-optimization API uses
BoTorch as an optional backend:

```bash
python -m pip install "matgpr[bo]"
```

For finite candidate lists, featurize measured rows and candidate rows with the
same descriptor pipeline, then rank the candidates:

```python
from matgpr import (
    BOBenchmarkStrategy,
    CandidateConstraint,
    CandidateDuplicatePolicy,
    CandidateTrustRegion,
    build_cartesian_candidate_grid,
    build_composition_candidate_grid,
    compare_bo_strategies,
    exclude_existing_candidates,
    summarize_candidate_category_coverage,
    summarize_candidate_feature_coverage,
    summarize_candidate_pool,
    ObjectiveSpec,
    observation_noise_variance,
    rank_multi_objective_candidates,
    select_diverse_batch,
    select_sequential_multi_objective_batch,
    select_pareto_front,
    split_candidate_features,
    suggest_multi_objective_next_experiments,
    suggest_next_experiments,
    summarize_bo_recommendation_audit,
    log_bo_recommendations,
    log_selected_experiments,
    log_observations,
    plot_bo_benchmark_trace,
    plot_bo_campaign_progress,
    plot_bo_regret_trace,
    resume_bo_campaign,
    summarize_closed_loop_log,
)
```

Build candidate pools explicitly before ranking. Use Cartesian grids for
processing conditions or formulation choices:

```python
process_candidates = build_cartesian_candidate_grid(
    {
        "temperature_c": [60, 80, 100],
        "solvent": ["water", "ethanol"],
        "annealing_time_min": [10, 30],
    },
    fixed_values={"campaign": "screen_1"},
)
```

Use composition grids when the candidate space is an inorganic composition
simplex. The generated table includes reduced formulas, number of components,
and element-fraction columns:

```python
composition_candidates = build_composition_candidate_grid(
    ["Al", "Co", "Ni"],
    step=0.25,
    min_components=2,
    max_components=3,
)
```

Audit candidate pools before BO. The pool diagnostics report descriptor
completeness, duplicate keys, numeric feature ranges, and categorical metadata
diversity:

```python
candidate_diagnostics = summarize_candidate_pool(
    composition_candidates,
    feature_columns=("frac_Al", "frac_Co", "frac_Ni"),
    categorical_columns=("formula",),
    key_columns=("formula",),
)

candidate_diagnostics.overview_frame()
candidate_diagnostics.numeric_feature_frame()
candidate_diagnostics.duplicate_key_frame()
```

Remove rows that are already measured or already selected before BO:

```python
composition_candidates = exclude_existing_candidates(
    composition_candidates,
    measured_data,
    key_columns=("formula",),
)
```

Check whether the finite pool covers the measured feature space. This helps
identify extrapolative BO campaigns before acquisition values are trusted:

```python
feature_coverage = summarize_candidate_feature_coverage(
    composition_candidates,
    measured_data,
    feature_columns=("frac_Al", "frac_Co", "frac_Ni"),
)

category_coverage = summarize_candidate_category_coverage(
    process_candidates,
    measured_data,
    categorical_columns=("solvent",),
)
```

After adding descriptors or selecting numeric fraction/process columns, split
features from metadata:

```python
X_candidate_features, candidate_metadata = split_candidate_features(
    composition_candidates,
    feature_columns=("frac_Al", "frac_Co", "frac_Ni"),
)
```

Then rank the finite pool:

```python
bo_result = suggest_next_experiments(
    X_train=X_measured_features,
    y_train=y_measured,
    X_candidates=X_candidate_features,
    candidate_data=candidate_metadata,
    top_k=5,
    acquisition_function="log_expected_improvement",
    maximize=True,
)

recommendations = bo_result.recommendations
ranked_pool = bo_result.ranked_candidates
```

Summarize why candidates were recommended before sending them to the lab or
recording them in a campaign log:

```python
recommendation_audit = summarize_bo_recommendation_audit(
    bo_result,
    candidate_count=len(composition_candidates),
    identifier_columns=("candidate_id", "formula"),
)

recommendation_audit.overview_frame()
recommendation_audit.score_summary_frame()
recommendation_audit.policy_summary_frame()
recommendation_audit.recommendation_frame()
```

The audit tables explain acquisition-score ranges, posterior uncertainty,
constraint status, trust-region status, duplicate status, and batch-selection
order when those columns are present in the ranked candidate table.

When the next experiment must balance multiple goals, include tradeoff columns
such as cost, toxicity, synthesis difficulty, or degradation rate in
`candidate_metadata`, then rank the finite pool with objective definitions.
Each objective states which column to use, whether larger or smaller values are
better, and how much weight it receives in the scalarized score:

```python
multi_objective_ranked = rank_multi_objective_candidates(
    ranked_pool,
    objectives=[
        ObjectiveSpec(
            name="performance",
            column="matgpr_predicted_mean",
            goal="maximize",
            weight=0.6,
        ),
        ObjectiveSpec(
            name="cost",
            column="estimated_cost_usd_g",
            goal="minimize",
            weight=0.25,
        ),
        ObjectiveSpec(
            name="toxicity",
            column="toxicity_score",
            goal="minimize",
            weight=0.15,
        ),
    ],
    top_k=10,
)

pareto_candidates = select_pareto_front(
    multi_objective_ranked,
    objectives=[
        ObjectiveSpec("performance", "matgpr_predicted_mean", "maximize"),
        ObjectiveSpec("cost", "estimated_cost_usd_g", "minimize"),
        ObjectiveSpec("toxicity", "toxicity_score", "minimize"),
    ],
)
```

When two or more objectives are measured outcomes, use BoTorch multi-objective
Bayesian optimization. `matgpr` fits one independent GP per objective, converts
minimize-type objectives into BoTorch maximization space internally, and ranks
candidates by expected hypervolume improvement:

```python
multi_bo_result = suggest_multi_objective_next_experiments(
    X_train=X_measured_features,
    y_train=measured_data[["conductivity_s_cm", "degradation_rate"]],
    X_candidates=X_candidate_features,
    objective_directions=("maximize", "minimize"),
    candidate_data=candidate_metadata,
    reference_point=(0.0, 1.0),
    top_k=5,
    acquisition_function="q_log_expected_hypervolume_improvement",
)

multi_objective_recommendations = multi_bo_result.recommendations
```

The optional `reference_point` is provided in original objective units and
directions. For example, the second value above is a worse degradation rate.
If omitted, `matgpr` estimates a conservative reference point from the observed
training objectives. Recommendation tables include `matgpr_acquisition`,
`matgpr_predicted_pareto_front`, and one mean/std pair per objective, such as
`matgpr_predicted_mean_conductivity_s_cm` and
`matgpr_predicted_std_degradation_rate`. Supported multi-objective acquisition
functions are `"q_log_expected_hypervolume_improvement"`,
`"q_log_noisy_expected_hypervolume_improvement"`,
`"q_expected_hypervolume_improvement"`, and
`"q_noisy_expected_hypervolume_improvement"`.

For multi-experiment batches, use sequential hypervolume-aware selection. This
greedy strategy picks one candidate, treats it as pending, then recomputes the
multi-objective acquisition for the remaining pool:

```python
sequential_bo_result = suggest_multi_objective_next_experiments(
    X_train=X_measured_features,
    y_train=measured_data[["conductivity_s_cm", "degradation_rate"]],
    X_candidates=X_candidate_features,
    objective_directions=("maximize", "minimize"),
    candidate_data=candidate_metadata,
    top_k=4,
    batch_selection="sequential",
    acquisition_function="q_log_expected_hypervolume_improvement",
)

sequential_batch = sequential_bo_result.recommendations[
    [
        "candidate_id",
        "matgpr_batch_order",
        "matgpr_batch_score",
        "matgpr_acquisition",
    ]
]
```

`matgpr_batch_score` is the step-wise acquisition value after previously
selected candidates are treated as pending. `matgpr_acquisition` remains the
single-candidate acquisition value, which is useful for auditing why sequential
batch order may differ from the individual ranking.

When experimental rows have known measurement uncertainty, pass a target-noise
variance vector to the BoTorch surrogate. The variance should be in squared
target units. It can come from a reported variance column, a standard-deviation
column, a standard-error column, or replicate measurements:

```python
noise_variance = observation_noise_variance(
    measured_data,
    std_column="conductivity_std",
)

bo_result = suggest_next_experiments(
    X_train=X_measured_features,
    y_train=y_measured,
    X_candidates=X_candidate_features,
    candidate_data=candidate_metadata,
    noise_variance=noise_variance,
    acquisition_function="log_noisy_expected_improvement",
    top_k=5,
)
```

For replicate measurements, use a stable group label such as material ID,
composition ID, formulation ID, or experiment condition ID:

```python
noise_variance = observation_noise_variance(
    measured_data,
    replicate_group_column="material_id",
    target_column="conductivity_s_cm",
)
```

Add finite-pool feasibility constraints when only part of the candidate library
is experimentally realistic. Examples include synthesis-temperature windows,
composition limits, solvent restrictions, safety filters, or equipment limits:

```python
constraints = [
    CandidateConstraint(
        name="temperature_window",
        column="synthesis_temperature_c",
        lower_bound=25.0,
        upper_bound=120.0,
    ),
    CandidateConstraint(
        name="allowed_solvent",
        column="solvent_class",
        allowed_values=("green", "water", "alcohol"),
    ),
]

bo_result = suggest_next_experiments(
    X_train=X_measured_features,
    y_train=y_measured,
    X_candidates=X_candidate_features,
    candidate_data=candidate_metadata,
    constraints=constraints,
    constraint_policy="filter",
    top_k=5,
)
```

The returned tables include:

| Column | Meaning |
| --- | --- |
| `matgpr_rank` | Candidate rank by acquisition value. |
| `matgpr_feasible` | Whether the candidate satisfies the supplied constraints. |
| `matgpr_constraint_violations` | Semicolon-separated labels for failed constraints. |
| `matgpr_predicted_mean` | GP posterior mean in original target direction. |
| `matgpr_predicted_std` | GP posterior standard deviation. |
| `matgpr_acquisition` | Acquisition value used for ranking. |

Supported acquisition functions are `"log_expected_improvement"`,
`"log_noisy_expected_improvement"`, `"expected_improvement"`,
`"noisy_expected_improvement"`, `"probability_of_improvement"`, and
`"upper_confidence_bound"`. Prefer the log variants for numerical stability.
Use the noisy variants when the surrogate was fit with known observation noise.
Use `maximize=False` for targets where lower values are better, such as
degradation rate, diffusion barrier, cost, or toxicity. Use
`constraint_policy="annotate"` when you want to keep infeasible candidates in
the ranked table for auditing rather than filtering them out.

Use trust regions when a BO campaign should stay near known feasible chemistry,
composition, or processing space. Use duplicate policies when candidates may
already be measured, selected in a previous batch, or queued in a lab workflow:

```python
trust_region = CandidateTrustRegion(
    centers=X_measured_features,
    radius=2.0,
    feature_scales="std",
)

duplicate_policy = CandidateDuplicatePolicy(
    existing_candidates=measured_data,
    key_columns=("candidate_id",),
)

bo_result = suggest_next_experiments(
    X_train=X_measured_features,
    y_train=y_measured,
    X_candidates=X_candidate_features,
    candidate_data=candidate_metadata,
    trust_region=trust_region,
    trust_region_policy="filter",
    duplicate_policy=duplicate_policy,
    duplicate_policy_action="filter",
    top_k=5,
)
```

`CandidateTrustRegion` measures each candidate's distance to the nearest
center using `"euclidean"`, `"manhattan"`, or `"chebyshev"` distance. Set
`feature_scales="std"` when descriptor magnitudes differ. `CandidateDuplicatePolicy`
can match exact metadata keys with `key_columns`, near-duplicate descriptors
with `feature_tolerance`, or both. Use `"annotate"` policies to keep all rows
and add audit columns such as `matgpr_in_trust_region`,
`matgpr_trust_region_distance`, `matgpr_is_duplicate`,
`matgpr_duplicate_reason`, and `matgpr_duplicate_distance`.

When selecting several experiments at once, use diversity-aware batch selection
to avoid near-duplicate candidates:

```python
bo_result = suggest_next_experiments(
    X_train=X_measured_features,
    y_train=y_measured,
    X_candidates=X_candidate_features,
    candidate_data=candidate_metadata_with_descriptors,
    top_k=5,
    batch_selection="diverse",
    batch_feature_columns=("band_gap_ev", "formation_energy_ev_atom"),
    diversity_weight=0.5,
)

diverse_recommendations = bo_result.recommendations
```

You can also apply the batch selector directly to an already ranked candidate
table:

```python
diverse_recommendations = select_diverse_batch(
    bo_result.ranked_candidates,
    top_k=5,
    feature_columns=("band_gap_ev", "formation_energy_ev_atom"),
    diversity_weight=0.5,
)
```

Log each closed-loop step so recommendations, selected experiments, and
measured outcomes can be audited across campaign iterations:

```python
log_path = "results/bo_campaign_log.csv"

log_bo_recommendations(
    bo_result.recommendations,
    path=log_path,
    campaign_id="conductivity_screen",
    iteration=0,
    model_name="physics_informed_gpr",
    acquisition_function=bo_result.acquisition_function,
)

selected_experiments = diverse_recommendations.head(3)
log_selected_experiments(
    selected_experiments,
    path=log_path,
    campaign_id="conductivity_screen",
    iteration=0,
    selection_policy="diverse_top_3",
)

new_measurements = pd.DataFrame(
    {
        "candidate_id": selected_experiments["candidate_id"],
        "conductivity_s_cm": [0.18, 0.24, 0.21],
    }
)
log_observations(
    new_measurements,
    path=log_path,
    campaign_id="conductivity_screen",
    iteration=1,
    target_column="conductivity_s_cm",
)

campaign_summary = summarize_closed_loop_log(
    log_path,
    campaign_id="conductivity_screen",
    target_column="conductivity_s_cm",
)
```

Before committing to a closed-loop policy, benchmark candidate-ranking
strategies on historical finite-pool data where outcomes are already known:

```python
benchmark = compare_bo_strategies(
    historical_candidates,
    strategies=[
        BOBenchmarkStrategy(
            "expected_improvement",
            score_column="matgpr_acquisition",
        ),
        BOBenchmarkStrategy(
            "physics_prior",
            score_column="physics_score",
        ),
        BOBenchmarkStrategy("random"),
    ],
    target_column="measured_conductivity_s_cm",
    candidate_id_column="candidate_id",
    maximize=True,
    budget=20,
    n_repeats=25,
    random_state=42,
)

benchmark_history = benchmark.history
benchmark_summary = benchmark.summary_by_strategy()

plot_bo_benchmark_trace(benchmark_history)
plot_bo_regret_trace(benchmark_history)
```

`benchmark.history` records the best value found after each simulated
experiment. `benchmark.summary` gives one row per strategy and repeat, while
`benchmark.summary_by_strategy()` aggregates final best value, simple regret,
hit-optimum rate, and evaluations needed to find the optimum. This is an
offline replay against known outcomes; it is useful for comparing acquisition
or ranking ideas, but it should not be reported as a live prospective
closed-loop result.

At the start of a new session, rebuild campaign state from the log before
asking for the next recommendations:

```python
campaign_state = resume_bo_campaign(
    log_path,
    campaign_id="conductivity_screen",
    candidate_pool=composition_candidates,
    key_columns=("candidate_id",),
)

next_iteration = campaign_state.next_iteration
available_candidates = campaign_state.available_candidates
duplicate_policy = campaign_state.duplicate_policy()
```

`campaign_state.pending_experiments` contains selected candidates without
matching observations. `campaign_state.completed_experiments` contains the
latest observed rows by key. `campaign_state.available_candidates` removes
both pending and completed candidates from the finite pool, and
`campaign_state.duplicate_policy()` can be passed directly to
`suggest_next_experiments` to guard against accidental repeats.

For closed-loop reports, plot either the logged record counts or the best
measured target value so far:

```python
plot_bo_campaign_progress(log_path, campaign_id="conductivity_screen")

plot_bo_campaign_progress(
    log_path,
    campaign_id="conductivity_screen",
    target_column="conductivity_s_cm",
    maximize=True,
)
```

These workflows are best for materials informatics tasks where users have a
realistic library of synthesizable materials or feasible experimental
conditions, either loaded from a file or generated as a finite candidate grid.

## 12. Save Models and Results

Save fitted preprocessors, models, or full pipelines:

```python
from matgpr import save_artifact, load_artifact, log_experiment_result

save_artifact(scaler, "artifacts/scaler.joblib")
save_artifact(result, "artifacts/gpytorch_gpr_result.joblib")

loaded_result = load_artifact("artifacts/gpytorch_gpr_result.joblib")
```

Append metrics to a CSV log:

```python
log_experiment_result(
    metrics,
    metadata={
        "model": "physics_informed_gpr",
        "kernel": "matern",
        "train_fraction": 0.2,
        "random_state": 42,
    },
    path="results/experiment_log.csv",
)
```

## 13. Common Troubleshooting

### Invalid Polymer SMILES

Polymer SMILES must contain exactly two `[*]` atoms for
`smiles_type="polymer"`.

```python
result = featurize_smiles(
    data["polymer_smiles"],
    smiles_type="polymer",
    errors="coerce",
)
print(result.failed)
```

Use `errors="coerce"` during dataset cleaning to identify invalid rows, then
drop or manually repair them before final modeling.

### Feature Index Mistakes in Physics Equations

`feature_indices` must match the exact column order of the array passed to
`fit_gpytorch_gpr`.

```python
feature_columns = descriptor_columns + physics_features
feature_indices = {name: feature_columns.index(name) for name in physics_features}
```

If the model input is scaled, also pass the original feature means and standard
deviations for the physics columns.

### Target Standardization

`fit_gpytorch_gpr(..., standardize_y=True)` trains on standardized targets and
returns predictions in original target units. `PhysicsInformedMean` handles this
automatically when passed to `fit_gpytorch_gpr`.

### Small Datasets

For small materials datasets:

- Prefer simple physics equations.
- Use repeated random splits.
- Plot learning curves from low train fractions.
- Compare standard GPR and physics-informed GPR with the same features,
  kernel, and split.
- Report both mean performance and standard deviation across splits.

## 14. Minimal End-to-End Example

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

from matgpr import (
    normalize_column_names,
    replace_missing_placeholders,
    append_composition_fingerprints,
    split_train_test,
    fit_gpytorch_gpr,
    regression_metrics,
    plot_parity,
)

data = pd.read_pickle("dataset.pkl")
data = normalize_column_names(data)
data = replace_missing_placeholders(data)
data = append_composition_fingerprints(data, formula_column="formula", errors="coerce")
data = data.dropna().reset_index(drop=True)

target_column = "target_property"
feature_columns = [column for column in data.columns if column not in ["formula", target_column]]
X = data[feature_columns]
y = data[target_column]

X_train_df, X_test_df, y_train, y_test = split_train_test(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

result = fit_gpytorch_gpr(
    X_train,
    y_train.to_numpy(),
    kernel="matern",
    ard=True,
    training_iter=1000,
    standardize_y=True,
)

train_prediction = result.predict(X_train, confidence_level=0.95)
test_prediction = result.predict(X_test, confidence_level=0.95)

metrics = regression_metrics(y_test, test_prediction.mean)
print(metrics)

plot_parity(
    y_train,
    train_prediction.mean,
    y_train_std=train_prediction.std,
    y_test_true=y_test,
    y_test_pred=test_prediction.mean,
    y_test_std=test_prediction.std,
)
```

This example is intentionally compact. For full scientific workflows, see the
worked examples in `examples/`.
