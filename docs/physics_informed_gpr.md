# Physics-Informed GPR

Physics-informed GPR in `matgpr` introduces domain knowledge through the GP
mean function. The GP still learns residual structure and uncertainty from data,
but the prior mean is no longer an arbitrary constant.

## Model Form

A standard GPR model can be written as:

$$
y(\mathbf{x}) = m(\mathbf{x}) + f(\mathbf{x}) + \epsilon
$$

where:

- \(m(\mathbf{x})\) is the mean function.
- \(f(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x'}))\) is the
  learned residual process.
- \(\epsilon\) is observational noise.

In a standard model, \(m(\mathbf{x})\) is usually constant. In a
physics-informed model, `matgpr` lets the user define:

$$
m(\mathbf{x}) = g(\mathbf{x}_{phys}; \boldsymbol{\theta}, \mathbf{c})
$$

where:

- \(g\) is a user-supplied physics equation.
- \(\mathbf{x}_{phys}\) are the features needed by that equation.
- \(\boldsymbol{\theta}\) are learnable equation parameters.
- \(\mathbf{c}\) are fixed constants.

The model learns both the GP kernel/noise parameters and the selected physics
parameters during GPyTorch training.

## Why Put Physics In The Mean?

For materials datasets, especially low-data experimental datasets, a useful
physics mean function can:

- encode the expected trend before the GP sees many observations,
- reduce the burden on the kernel to rediscover known behavior,
- improve extrapolation when the equation is valid,
- make learned parameters easier to inspect than a fully black-box baseline.

The GP residual is still important. It captures systematic deviations caused by
missing descriptors, approximations in the equation, experimental noise, and
material-specific effects not included in the physics term.

## Implementation Pattern

Define an equation that accepts a feature dictionary and a parameter dictionary.
The parameter dictionary contains both learnable parameters and fixed
parameters:

```python
import torch


def arrhenius_equation(features, parameters):
    temperature_k = features["temperature_c"] + 273.15
    time_min = torch.clamp(features["time_min"], min=1e-8)

    prefactor = parameters["prefactor"]
    activation_energy = parameters["activation_energy"]
    gas_constant = parameters["gas_constant"]

    rate = prefactor * torch.exp(-activation_energy / (gas_constant * temperature_k))
    return torch.sqrt(torch.clamp(rate * time_min, min=1e-12))
```

Create a `PhysicsInformedMean`:

```python
from matgpr import PhysicsInformedMean

mean_function = PhysicsInformedMean(
    equation=arrhenius_equation,
    feature_indices={"temperature_c": 0, "time_min": 1},
    learnable_parameters={
        "prefactor": 1.0,
        "activation_energy": 50_000.0,
    },
    fixed_parameters={"gas_constant": 8.314},
    positive_parameters=("prefactor", "activation_energy"),
)
```

Then pass it to GPyTorch fitting:

```python
from matgpr import fit_gpytorch_gpr

result = fit_gpytorch_gpr(
    X_train,
    y_train,
    mean_module=mean_function,
    training_iter=1000,
)

prediction = result.predict(X_test, confidence_level=0.95)
learned_parameters = result.model.mean_module.current_parameter_values()
```

## Feature Scaling

Most GPR workflows scale features before training. A physics equation, however,
often expects original physical units. `PhysicsInformedMean` can recover original
feature values when feature means and standard deviations are supplied:

```python
mean_function = PhysicsInformedMean(
    equation=arrhenius_equation,
    feature_indices={"temperature_c": 0, "time_min": 1},
    learnable_parameters={"prefactor": 1.0, "activation_energy": 50_000.0},
    feature_means={"temperature_c": 150.0, "time_min": 60.0},
    feature_stds={"temperature_c": 25.0, "time_min": 15.0},
)
```

This keeps the GP numerically stable while preserving physical units inside the
equation.

## Choosing Learnable Parameters

Use learnable parameters for quantities that:

- are physically meaningful,
- are uncertain or dataset-specific,
- can be estimated from the available data,
- should be constrained to remain positive or within a sensible domain.

Keep constants fixed when they are known physical constants or when the dataset
is too small to support reliable estimation.

## Validation Expectations

When reporting a physics-informed model, include:

- the exact equation used for \(m(\mathbf{x})\),
- the features used by the equation,
- which parameters were learned and their initial values,
- which parameters were fixed,
- whether feature scaling was inverted inside the mean function,
- the train/test protocol,
- learning curves comparing standard GPR and PI-GPR,
- uncertainty diagnostics such as interval coverage and NLPD.

Do not claim that the model is physically correct only because it uses a physics
term. The equation is an inductive bias, and it should be checked against held
out data, uncertainty calibration, and domain expectations.
