# Safe Custom Equations

Safe custom-equation records define the JSON-safe specification and expression
validation layer for future user-defined physics-informed mean functions.

The current API supports:

- strict JSON-safe equation specifications,
- allowlisted expression parsing with Python `ast` and no arbitrary execution,
- NumPy preview evaluation with `evaluate_safe_equation_numpy(...)`,
- finite/non-finite output reporting,
- physics-mean and optional target-residual previews with
  `preview_safe_equation_mean(...)`,
- torch evaluation with `evaluate_safe_equation_torch(...)`,
- differentiable parameter initialization with
  `initialize_safe_equation_torch_parameters(...)`,
- trainable PI-GPR mean-function adapters with
  `build_safe_equation_mean_function(...)`.

Use this layer before model fitting to check that a proposed physics mean has
declared variables, finite defaults, supported functions, and finite outputs on
sample data.

```python
from matgpr import (
    SafeEquationParameter,
    SafeEquationSpec,
    SafeEquationVariable,
    preview_safe_equation_mean,
)

spec = SafeEquationSpec(
    name="custom_power_law",
    expression="coefficient * pow(x, exponent)",
    variables=[SafeEquationVariable("x", units="a.u.")],
    parameters=[
        SafeEquationParameter("coefficient", initial_value=2.0),
        SafeEquationParameter("exponent", initial_value=0.5),
    ],
)

preview = preview_safe_equation_mean(
    spec,
    {"x": [1.0, 4.0, 9.0]},
    target_values=[2.1, 3.8, 6.2],
)
preview.to_dict()
```

Use the torch evaluator when gradient flow is required:

```python
import torch

from matgpr import (
    build_safe_equation_mean_function,
    evaluate_safe_equation_torch,
    initialize_safe_equation_torch_parameters,
)

parameters = initialize_safe_equation_torch_parameters(spec)
mean = evaluate_safe_equation_torch(
    spec,
    {"x": torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)},
    parameter_values=parameters,
)
mean.sum().backward()
```

Build a trainable PI-GPR mean adapter when fitting with GPyTorch:

```python
mean_module = build_safe_equation_mean_function(
    spec,
    feature_indices={"x": 0},
)
```

The returned `SafeEquationMeanFunction` is compatible with
`fit_gpytorch_gpr(..., mean_module=mean_module)`. It uses the spec's
`learnable` flags to split parameters into trainable and fixed physics
parameters, stores adapter metadata with `to_dict()`, and evaluates the safe
equation through the torch interpreter during training.

::: matgpr.safe_equations
