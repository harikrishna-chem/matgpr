# Quickstart

This page shows the shortest practical path from a standard GPR baseline to a
physics-informed GPR model. It uses a tiny synthetic materials-aging dataset so
the workflow can be copied into a fresh notebook.

## 1. Install And Import

```bash
python -m pip install "matgpr[examples] @ git+https://github.com/harikrishna-chem/matgpr.git@main"
```

```python
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from matgpr import (
    MatGPRRegressor,
    PhysicsInformedGPRRegressor,
    plot_parity,
    regression_metrics,
)
```

## 2. Load Or Build A Dataframe

In your own project, replace this synthetic dataframe with a literature,
experimental, or simulation dataset.

```python
rng = np.random.default_rng(7)
n_samples = 80

temperature_k = rng.uniform(320.0, 620.0, n_samples)
time_h = rng.uniform(1.0, 80.0, n_samples)
composition_descriptor = rng.normal(0.0, 1.0, n_samples)

true_prefactor = 1.5e3
true_activation_energy = 18_000.0
gas_constant = 8.314462618

physics_signal = np.sqrt(
    true_prefactor
    * np.exp(-true_activation_energy / (gas_constant * temperature_k))
    * time_h
)
target = 0.15 + physics_signal + 0.08 * composition_descriptor + rng.normal(0.0, 0.04, n_samples)

data = pd.DataFrame(
    {
        "temperature_k": temperature_k,
        "time_h": time_h,
        "composition_descriptor": composition_descriptor,
        "aged_property": target,
    }
)
```

Use clear units and column names. Keep a short data card for real datasets:
source, target units, feature definitions, filtering, and train/test protocol.

## 3. Split Features And Target

```python
feature_columns = ["temperature_k", "time_h", "composition_descriptor"]
target_column = "aged_property"

train_data, test_data = train_test_split(
    data,
    test_size=0.25,
    random_state=42,
)

X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_test = test_data[feature_columns]
y_test = test_data[target_column]
```

For real studies, fit all preprocessing only on `X_train`, then apply the
fitted preprocessing to validation, test, and candidate data.

## 4. Fit A Standard GPR Baseline

Start with a non-physics model. It gives you a data-only baseline for accuracy,
uncertainty, and calibration.

```python
standard_model = MatGPRRegressor(
    kernel="matern",
    ard=True,
    training_iter=300,
    initial_noise=0.05,
    standardize_y=True,
    random_state=42,
)

standard_model.fit(X_train, y_train)
standard_prediction = standard_model.predict_distribution(
    X_test,
    confidence_level=0.95,
)

standard_metrics = regression_metrics(
    y_test,
    standard_prediction.mean,
)
standard_metrics
```

## 5. Define A Physics Mean

Physics-informed GPR in `matgpr` uses:

```text
target = physics mean + GP residual + noise
```

The equation below encodes square-root-time growth with an Arrhenius rate:

```python
def arrhenius_sqrt_time_mean(features, parameters):
    temperature = torch.clamp(features["temperature_k"], min=1.0)
    time = torch.clamp(features["time_h"], min=1e-8)

    prefactor = parameters["prefactor"]
    activation_energy = parameters["activation_energy"]
    offset = parameters["offset"]
    gas_constant = parameters["gas_constant"]

    rate = prefactor * torch.exp(-activation_energy / (gas_constant * temperature))
    return offset + torch.sqrt(torch.clamp(rate * time, min=0.0))
```

The equation receives dictionaries of torch tensors:

- `features` contains the columns listed in `physics_features`.
- `parameters` contains learnable and fixed physics parameters.

## 6. Fit Physics-Informed GPR

```python
physics_model = PhysicsInformedGPRRegressor(
    equation=arrhenius_sqrt_time_mean,
    physics_features=("temperature_k", "time_h"),
    learnable_parameters={
        "prefactor": 500.0,
        "activation_energy": 12_000.0,
        "offset": 0.1,
    },
    fixed_parameters={"gas_constant": 8.314462618},
    positive_parameters=("prefactor", "activation_energy"),
    kernel="matern",
    ard=True,
    training_iter=300,
    initial_noise=0.05,
    standardize_y=True,
    random_state=42,
)

physics_model.fit(X_train, y_train)
physics_prediction = physics_model.predict_distribution(
    X_test,
    confidence_level=0.95,
)

physics_metrics = regression_metrics(
    y_test,
    physics_prediction.mean,
)
physics_metrics
```

Inspect the learned physics parameters:

```python
physics_model.learned_physics_parameters_
```

The GP also learns kernel length scales, output scale, and Gaussian likelihood
noise. The physics parameters are optimized during the same GPyTorch training
loop; they are not fitted separately and frozen.

## 7. Compare Models

```python
comparison = pd.DataFrame(
    [
        {"model": "Standard GPR", **standard_metrics},
        {"model": "Physics-informed GPR", **physics_metrics},
    ]
).set_index("model")

comparison
```

```python
physics_train_prediction = physics_model.predict_distribution(X_train)

fig, ax = plot_parity(
    y_train,
    physics_train_prediction.mean,
    y_train_std=physics_train_prediction.std,
    y_test_true=y_test,
    y_test_pred=physics_prediction.mean,
    y_test_std=physics_prediction.std,
    title="Quickstart PI-GPR Test Parity",
    xlabel="Measured property",
    ylabel="Predicted property",
)
```

In a real benchmark, compare standard GPR and PI-GPR with the same split,
features, kernel family, training budget, and metrics. For low-data claims, use
repeated splits or learning curves rather than a single lucky split.

## 8. What To Report

When you publish or share a PI-GPR workflow, report:

- target name, units, and data source,
- feature columns and descriptor generation method,
- train/test split or cross-validation protocol,
- standard GPR baseline metrics,
- physics equation used for the mean function,
- features used by the equation,
- learned and fixed parameters,
- whether parameters were constrained positive,
- uncertainty metrics or parity plots with error bars,
- limitations and applicability domain.

## Next Pages

- [User Guide](matgpr_user_guide.md) for end-to-end workflows.
- [Physics-Informed GPR](physics_informed_gpr.md) for equations and reporting
  expectations.
- [Fingerprinting Options](fingerprinting_options.md) for descriptors.
- [API Reference](api/index.md) for exact signatures.
