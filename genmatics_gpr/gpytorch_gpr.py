from __future__ import annotations

from collections.abc import Callable, Mapping

import gpytorch
import numpy as np
import torch


class EquationMeanFunction(gpytorch.means.Mean):
    """Physics-informed mean function driven by a user-supplied equation.

    The equation is a Python callable that receives two dictionaries:
    ``features`` and ``parameters``. ``features`` maps user-selected column
    names to torch tensors, and ``parameters`` maps parameter names to learnable
    torch tensors.

    Example
    -------
    def oxidation_equation(features, parameters):
        temperature_k = features["temperature_c"] + 273.15
        time_min = torch.clamp(features["time_min"], min=1e-8)
        A = parameters["A"]
        Q = parameters["Q"]
        kp = A * torch.exp(-Q / (8.314 * temperature_k))
        return torch.sqrt(torch.clamp(kp * time_min, min=1e-12))
    """

    def __init__(
        self,
        *,
        equation: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor],
        column_indices: Mapping[str, int],
        parameter_initial_values: Mapping[str, float] | None = None,
        positive_parameters: tuple[str, ...] = (),
        feature_means: Mapping[str, float] | None = None,
        feature_stds: Mapping[str, float] | None = None,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ):
        super().__init__()
        self.equation = equation
        self.column_indices = dict(column_indices)
        self.positive_parameters = set(positive_parameters)
        self.feature_means = dict(feature_means or {})
        self.feature_stds = dict(feature_stds or {})
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

        for name, value in (parameter_initial_values or {}).items():
            initial_value = torch.tensor(float(value))
            if name in self.positive_parameters:
                if value <= 0:
                    raise ValueError(f"Positive parameter '{name}' must start above zero")
                initial_value = torch.log(initial_value)
            self.register_parameter(name=f"raw_{name}", parameter=torch.nn.Parameter(initial_value))

    def _parameter_value(self, name: str) -> torch.Tensor:
        value = getattr(self, f"raw_{name}")
        if name in self.positive_parameters:
            return torch.exp(value)
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = {}
        for name, index in self.column_indices.items():
            value = x[:, index]
            if name in self.feature_means or name in self.feature_stds:
                mean = self.feature_means.get(name, 0.0)
                std = self.feature_stds.get(name, 1.0)
                value = value * std + mean
            features[name] = value

        parameters = {
            name.removeprefix("raw_"): self._parameter_value(name.removeprefix("raw_"))
            for name, _ in self.named_parameters()
            if name.startswith("raw_")
        }
        mean = self.equation(features, parameters)
        return (mean - self.y_mean) / self.y_std


class ExactGPRModel(gpytorch.models.ExactGP):
    """Exact Gaussian Process Regression model using GPyTorch."""

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        *,
        kernel: str = "matern",
        ard_num_dims: int | None = None,
        mean_module=None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            _make_gpytorch_base_kernel(kernel, ard_num_dims=ard_num_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _make_gpytorch_base_kernel(
    name: str,
    *,
    ard_num_dims: int | None = None,
):
    normalized = name.lower()

    if normalized == "rbf":
        return gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    if normalized == "matern":
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    if normalized == "matern15":
        return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)

    raise ValueError("name must be one of: rbf, matern, matern15")


def _to_tensor(x, *, device: str = "cpu", dtype=torch.float64) -> torch.Tensor:
    """Convert numpy, pandas, or torch input to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


def train_gpytorch_gpr(
    X_train,
    y_train,
    *,
    kernel: str = "matern",
    ard: bool = True,
    mean_module=None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    device: str = "cpu",
    dtype=torch.float64,
    verbose: bool = True,
):
    """Train an exact GPR model using GPyTorch.

    ``mean_module`` may be ``None`` for a learned constant mean or an
    ``EquationMeanFunction`` for physics-informed modeling. If targets are
    standardized, the returned predictions are still converted back to original
    target units by ``predict_gpytorch_gpr``.
    """
    train_x = _to_tensor(X_train, device=device, dtype=dtype)
    train_y = _to_tensor(y_train, device=device, dtype=dtype).ravel()

    if standardize_y:
        y_mean = train_y.mean()
        y_std = train_y.std()
        if y_std.item() == 0:
            raise ValueError("y_train has zero standard deviation")
        train_y_model = (train_y - y_mean) / y_std
    else:
        y_mean = torch.tensor(0.0, dtype=dtype, device=device)
        y_std = torch.tensor(1.0, dtype=dtype, device=device)
        train_y_model = train_y

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
    if initial_noise is not None:
        likelihood.noise = initial_noise

    if mean_module is None:
        mean_module = gpytorch.means.ConstantMean()
    elif hasattr(mean_module, "y_mean") and hasattr(mean_module, "y_std"):
        mean_module.y_mean = y_mean
        mean_module.y_std = y_std
    mean_module = mean_module.to(device=device, dtype=dtype)

    model = ExactGPRModel(
        train_x,
        train_y_model,
        likelihood,
        kernel=kernel,
        ard_num_dims=train_x.shape[1] if ard else None,
        mean_module=mean_module,
    ).to(device=device, dtype=dtype)

    model.y_mean = y_mean
    model.y_std = y_std
    model.standardize_y = standardize_y

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y_model)
        loss.backward()
        optimizer.step()

        if verbose and ((i + 1) % 100 == 0 or i == 0):
            noise = likelihood.noise.item()
            outputscale = model.covar_module.outputscale.item()
            lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            print(
                f"Iter {i + 1:4d}/{training_iter} | "
                f"Loss: {loss.item():.4f} | "
                f"Noise: {noise:.4e} | "
                f"Outputscale: {outputscale:.4e} | "
                f"Mean lengthscale: {lengthscale.mean():.4e}"
            )

    return model, likelihood


def predict_gpytorch_gpr(
    model,
    likelihood,
    X,
    *,
    device: str = "cpu",
    dtype=torch.float64,
    return_std: bool = True,
):
    """Predict mean and optional standard deviation from a fitted GPyTorch GPR."""
    model.eval()
    likelihood.eval()
    test_x = _to_tensor(X, device=device, dtype=dtype)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        mean = pred_dist.mean
        std = pred_dist.stddev

    if hasattr(model, "y_mean") and hasattr(model, "y_std"):
        mean = mean * model.y_std + model.y_mean
        std = std * model.y_std

    mean = mean.detach().cpu().numpy()
    std = std.detach().cpu().numpy()

    if return_std:
        return mean, std
    return mean
