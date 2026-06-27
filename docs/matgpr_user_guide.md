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
8. Analyze RMSE, R2, Pearson r, parity plots, learning curves, PCA, and feature
   effects.
9. Save models, preprocessors, metrics, and plots.

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
    CompositionFeaturizer,
    SmilesFeaturizer,
    PolymerSmilesFeaturizer,
    separate_features_target,
    split_train_test,
    identify_feature_types,
    build_preprocessor,
    MatGPRRegressor,
    PhysicsInformedGPRRegressor,
    build_sklearn_gpr_model,
    fit_gpytorch_gpr,
    regression_metrics,
    train_test_regression_metrics,
    plot_parity,
    plot_learning_curve,
    save_artifact,
)
```

## 2. Data Cleaning

Start by making column names stable and missing values explicit.

```python
data = pd.read_csv("dataset.csv")

data = normalize_column_names(data)
data = replace_missing_placeholders(data)
data = drop_duplicate_rows(data)

missing_report = summarize_missingness(data)
numeric_report = summarize_numeric_columns(data)
```

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
from matgpr import append_composition_fingerprints, featurize_compositions

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

Useful functions:

| Function | Purpose |
| --- | --- |
| `clean_formula(formula)` | Normalizes formula text before parsing. |
| `composition_fingerprint(formula)` | Featurizes one inorganic formula. |
| `featurize_compositions(formulas, errors="raise")` | Featurizes many formulas and returns features plus failed rows. |
| `append_composition_fingerprints(data, formula_column="composition")` | Appends descriptors to an existing dataframe. |

For scikit-learn-style workflows, use `CompositionFeaturizer`:

```python
from matgpr import CompositionFeaturizer

composition_featurizer = CompositionFeaturizer(
    formula_column="composition",
    errors="coerce",
)

composition_features = composition_featurizer.fit_transform(data)
failed_formulas = composition_featurizer.failed_
feature_names = composition_featurizer.get_feature_names_out()
```

### 3.2 Molecule and Polymer Fingerprints

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
)
polymer_features = polymer_featurizer.fit_transform(data)

solvent_featurizer = SmilesFeaturizer(
    smiles_column="solvent_smiles",
    fingerprint_type="descriptors",
)
solvent_features = solvent_featurizer.fit_transform(data)
```

`SmilesFeaturizer` and `PolymerSmilesFeaturizer` store the most recent
canonical SMILES in `canonical_smiles_` and failed rows in `failed_`.

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
| `build_sklearn_gpr_kernel(name="matern")` | Builds an RBF, Matern, ARD RBF, or ARD Matern kernel. |
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

Useful fitted attributes:

| Attribute | Meaning |
| --- | --- |
| `result_` | Full `GPyTorchGPRResult` object. |
| `model_`, `likelihood_` | Fitted GPyTorch objects. |
| `loss_history_` | Training loss by optimizer iteration. |
| `target_mean_`, `target_std_` | Target standardization values. |

For confidence intervals:

```python
prediction = model.predict_distribution(X_test_array, confidence_level=0.95)
```

`prediction` contains `mean`, `std`, `lower`, and `upper`.

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

## 6. Physics-Informed GPR

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

### 6.1 How Physics Is Introduced

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

### 6.2 Example 1: Arrhenius Mean Function

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

### 6.3 Example 2: Degeneracy and Binding Mean Function

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

### 6.4 Learned vs Fixed Parameters

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

### 6.5 Practical Guidance for Physics Equations

Good physics-informed mean functions should be:

- Simple enough to be explainable.
- Written in the same units as the target.
- Based on columns available at prediction time.
- Helpful in low-data regimes.
- Flexible enough that the GP residual can correct imperfect physics.

Start with one or two physics terms. Add complexity only when learning curves,
test-set metrics, and uncertainty calibration improve.

## 7. Model Analysis

### 7.1 Regression Metrics

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

### 7.2 Parity Plot With Uncertainty

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

### 7.3 Learning Curves

For repeated train-size experiments, collect rows with columns such as
`train_size`, `model`, and `test_R2`.

```python
fig, ax, summary = plot_learning_curve(
    results_df,
    train_size_col="train_size",
    metric_col="test_R2",
    model_col="model",
    title="Learning curve",
)
```

For RMSE learning curves, set `metric_col="test_RMSE"`.

### 7.4 90/10 Validation With 10-Fold Cross-Validation

After selecting the best model family from learning curves, use a conventional
validation protocol before fitting the final production model:

1. Split the full dataset into 90 percent training and 10 percent held-out test.
2. Run 10-fold cross-validation inside the 90 percent training partition.
3. Refit the selected model on the full 90 percent training partition.
4. Plot cross-validation statistics on the left and train/test parity with
   uncertainty bars on the right.

The exact fitting function depends on the notebook, but the structure should
look like this:

```python
from sklearn.model_selection import StratifiedKFold, train_test_split

N_CV_SPLITS = 10

validation_train, validation_test = train_test_split(
    model_data,
    test_size=0.10,
    random_state=42,
    stratify=target_strata(model_data, TARGET_COLUMN),
)

cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=43)
cv_records = []

for fold, (train_idx, val_idx) in enumerate(cv.split(validation_train, target_strata(validation_train, TARGET_COLUMN)), start=1):
    fold_train = validation_train.iloc[train_idx].reset_index(drop=True)
    fold_val = validation_train.iloc[val_idx].reset_index(drop=True)

    fitted = fit_model(best_model_key, fold_train)
    train_pred = predict_model(fitted, fold_train)
    val_pred = predict_model(fitted, fold_val)

    cv_records.append({"fold": fold, "split": "CV train", **regression_summary(fold_train[TARGET_COLUMN], train_pred.mean)})
    cv_records.append({"fold": fold, "split": "CV validation", **regression_summary(fold_val[TARGET_COLUMN], val_pred.mean)})

cv_results = pd.DataFrame(cv_records)
cv_summary = (
    cv_results
    .groupby("split")
    .agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r_mean=("r", "mean"),
        r_std=("r", "std"),
    )
    .reset_index()
)
```

For small datasets, make sure each stratification bin has enough samples for
10 folds. If not, reduce the number of bins while keeping `N_CV_SPLITS = 10`.

### 7.5 PCA

```python
from matgpr import fit_pca, summarize_pca, transform_pca, plot_pca_scree, plot_pca_scores

train_scores, pca, pca_scaler = fit_pca(X_train_numeric, n_components=5, scale=True)
test_scores = transform_pca(X_test_numeric, pca, scaler=pca_scaler)

pca_summary = summarize_pca(pca)
plot_pca_scree(pca)
plot_pca_scores(train_scores, test_scores=test_scores)
```

### 7.6 SHAP Analysis

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

## 8. Save Models and Results

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

## 9. Common Troubleshooting

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

## 10. Minimal End-to-End Example

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

data = pd.read_csv("hardness.csv")
data = normalize_column_names(data)
data = replace_missing_placeholders(data)
data = append_composition_fingerprints(data, formula_column="formula", errors="coerce")
data = data.dropna().reset_index(drop=True)

target_column = "hardness_gpa"
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
