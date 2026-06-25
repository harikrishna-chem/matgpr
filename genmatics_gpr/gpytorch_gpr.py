from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F

TensorMap = dict[str, torch.Tensor]
PhysicsEquation = Callable[[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]], torch.Tensor]

__all__ = [
    "EquationMeanFunction",
    "ExactGPRModel",
    "GPyTorchGPRResult",
    "GPyTorchPrediction",
    "PhysicsEquation",
    "PhysicsInformedMean",
    "fit_gpytorch_gpr",
    "predict_gpytorch_gpr",
    "train_gpytorch_gpr",
]


@dataclass(frozen=True)
class GPyTorchPrediction:
    """Predictions from a fitted GPyTorch Gaussian-process model.

    Attributes
    ----------
    mean
        Predictive mean in the original target units.
    std
        Predictive standard deviation in the original target units. This is
        ``None`` when standard deviations were not requested.
    lower, upper
        Optional lower and upper confidence bounds in the original target
        units. These are populated when ``confidence_level`` is supplied.
    """

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


@dataclass(frozen=True)
class GPyTorchGPRResult:
    """Container returned by ``fit_gpytorch_gpr``.

    The result keeps the fitted model, likelihood, and training diagnostics
    together. Use ``predict`` for new inputs, or access ``model`` and
    ``likelihood`` directly when lower-level GPyTorch operations are needed.
    """

    model: ExactGPRModel
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    loss_history: list[float]
    target_mean: float
    target_std: float
    standardize_y: bool
    kernel: str
    ard: bool
    device: str
    dtype: torch.dtype

    def predict(
        self,
        X,
        *,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool = True,
    ) -> GPyTorchPrediction:
        """Predict target values for new samples.

        Parameters
        ----------
        X
            Feature matrix with the same column order used during fitting.
        return_std
            Whether to include predictive standard deviations.
        confidence_level
            Optional central confidence level, for example ``0.95`` for a
            95 percent interval.
        include_observation_noise
            If ``True``, uncertainty includes the fitted Gaussian likelihood
            noise. If ``False``, uncertainty comes from the latent function.
        """
        return _predict_gpytorch_gpr(
            self.model,
            self.likelihood,
            X,
            device=self.device,
            dtype=self.dtype,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )


class PhysicsInformedMean(gpytorch.means.Mean):
    """Physics-informed GP mean function backed by a user-supplied equation.

    A physics-informed Gaussian process can be viewed as
    ``observed response = mechanistic mean + data-driven residual``. This mean
    module evaluates the mechanistic equation, then lets the GP covariance
    model the residual structure left over by that equation.

    The equation receives two dictionaries:

    - ``features`` maps selected feature names to torch tensors.
    - ``parameters`` maps physical parameter names to torch tensors.

    Parameters listed in ``learnable_parameters`` are optimized jointly with
    the GP hyperparameters. Parameters listed in ``fixed_parameters`` stay
    constant but are still passed to the equation for clarity.
    """

    def __init__(
        self,
        *,
        equation: PhysicsEquation,
        feature_indices: Mapping[str, int] | None = None,
        column_indices: Mapping[str, int] | None = None,
        learnable_parameters: Mapping[str, float] | None = None,
        parameter_initial_values: Mapping[str, float] | None = None,
        positive_parameters: Sequence[str] = (),
        fixed_parameters: Mapping[str, float] | None = None,
        feature_means: Mapping[str, float] | None = None,
        feature_stds: Mapping[str, float] | None = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.equation = equation
        self.feature_indices = _resolve_feature_indices(feature_indices, column_indices)
        self.fixed_parameters = _validate_numeric_mapping(fixed_parameters or {}, "fixed_parameters")
        self.feature_means = _validate_numeric_mapping(feature_means or {}, "feature_means")
        self.feature_stds = _validate_positive_mapping(feature_stds or {}, "feature_stds")
        self.positive_parameters = set(positive_parameters)

        learnable_parameters = _resolve_learnable_parameters(
            learnable_parameters,
            parameter_initial_values,
        )
        self._learnable_parameter_names = tuple(learnable_parameters)
        self._validate_physics_parameter_names(learnable_parameters)

        target_std = float(target_std)
        if target_std <= 0:
            raise ValueError("target_std must be positive")
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(target_std))

        for name, initial_value in learnable_parameters.items():
            initial_value = float(initial_value)
            if name in self.positive_parameters:
                if initial_value <= 0:
                    raise ValueError(f"Positive parameter '{name}' must start above zero")
                raw_value = _inverse_softplus(initial_value)
            else:
                raw_value = torch.tensor(initial_value)
            self.register_parameter(
                name=f"raw_{name}",
                parameter=torch.nn.Parameter(raw_value),
            )

    @property
    def column_indices(self) -> dict[str, int]:
        """Backward-compatible alias for ``feature_indices``."""
        return dict(self.feature_indices)

    def set_target_standardization(
        self,
        mean: torch.Tensor | float,
        std: torch.Tensor | float,
    ) -> None:
        """Set the target transform used when training on standardized targets.

        Physics equations are usually written in original target units. When
        the GPR target is standardized for optimization, the mean function must
        apply the same target transform before returning values to GPyTorch.
        """
        std_value = torch.as_tensor(
            std,
            dtype=self.target_std.dtype,
            device=self.target_std.device,
        )
        if torch.any(std_value <= 0):
            raise ValueError("std must be positive")

        mean_value = torch.as_tensor(
            mean,
            dtype=self.target_mean.dtype,
            device=self.target_mean.device,
        )
        self.target_mean.copy_(mean_value)
        self.target_std.copy_(std_value)

    def current_parameter_values(self, *, detach: bool = True) -> dict[str, float | torch.Tensor]:
        """Return current physical parameter values.

        By default this returns Python floats for easy reporting. Set
        ``detach=False`` to inspect tensors that still participate in autograd.
        """
        values = self._physics_parameter_tensors(reference=self.target_mean)
        if not detach:
            return values
        return {name: value.detach().cpu().item() for name, value in values.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("PhysicsInformedMean expects a 2D feature tensor")
        self._validate_feature_width(x.shape[1])

        features = {
            name: self._feature_in_original_units(name, x[:, index])
            for name, index in self.feature_indices.items()
        }
        parameters = self._physics_parameter_tensors(reference=x)

        mean = self.equation(features, parameters)
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
        else:
            mean = mean.to(dtype=x.dtype, device=x.device)
        mean = mean.reshape(-1)

        if mean.shape[0] != x.shape[0]:
            raise ValueError(
                "Physics equation must return one mean value per sample; "
                f"received shape {tuple(mean.shape)} for {x.shape[0]} samples"
            )

        return (mean - self.target_mean.to(dtype=x.dtype, device=x.device)) / self.target_std.to(
            dtype=x.dtype,
            device=x.device,
        )

    def _feature_in_original_units(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self.feature_means and name not in self.feature_stds:
            return value
        mean = self.feature_means.get(name, 0.0)
        std = self.feature_stds.get(name, 1.0)
        return value * std + mean

    def _physics_parameter_tensors(self, *, reference: torch.Tensor) -> TensorMap:
        values: TensorMap = {
            name: torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
            for name, value in self.fixed_parameters.items()
        }
        for name in self._learnable_parameter_names:
            raw_value = getattr(self, f"raw_{name}").to(
                dtype=reference.dtype,
                device=reference.device,
            )
            if name in self.positive_parameters:
                values[name] = F.softplus(raw_value)
            else:
                values[name] = raw_value
        return values

    def _validate_feature_width(self, n_features: int) -> None:
        if not self.feature_indices:
            return
        max_index = max(self.feature_indices.values())
        if max_index >= n_features:
            raise ValueError(
                f"feature_indices references column {max_index}, but input has only "
                f"{n_features} feature columns"
            )

    def _validate_physics_parameter_names(self, learnable_parameters: Mapping[str, float]) -> None:
        duplicates = set(learnable_parameters).intersection(self.fixed_parameters)
        if duplicates:
            raise ValueError(f"Parameters cannot be both fixed and learnable: {sorted(duplicates)}")

        unknown_positive = self.positive_parameters.difference(learnable_parameters)
        if unknown_positive:
            raise ValueError(
                "positive_parameters must refer to learnable parameters; "
                f"unknown names: {sorted(unknown_positive)}"
            )

        for name in set(learnable_parameters).union(self.fixed_parameters):
            _validate_torch_parameter_name(name)


class EquationMeanFunction(PhysicsInformedMean):
    """Backward-compatible name for ``PhysicsInformedMean``."""


class ExactGPRModel(gpytorch.models.ExactGP):
    """Exact Gaussian Process Regression model using GPyTorch."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        kernel: str = "matern",
        ard_num_dims: int | None = None,
        mean_module: gpytorch.means.Mean | None = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            _make_gpytorch_base_kernel(kernel, ard_num_dims=ard_num_dims)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gpytorch_gpr(
    X_train,
    y_train,
    *,
    kernel: str = "matern",
    ard: bool = True,
    mean_module: gpytorch.means.Mean | None = None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> GPyTorchGPRResult:
    """Fit an exact GPyTorch Gaussian Process Regressor.

    Parameters
    ----------
    X_train, y_train
        Training features and target values. Pandas, NumPy, and torch inputs
        are accepted.
    kernel
        Base kernel name: ``"rbf"``, ``"matern"``, ``"matern15"``, or
        ``"matern05"``.
    ard
        If ``True``, learn one kernel length scale per feature.
    mean_module
        Optional mean module. Pass ``PhysicsInformedMean`` to encode a
        materials-physics equation as the GP prior mean.
    lr
        Adam learning rate.
    training_iter
        Number of optimization iterations.
    initial_noise
        Optional initial Gaussian likelihood noise.
    standardize_y
        Whether to standardize target values during optimization. Predictions
        are always returned in original target units.
    """
    _validate_training_options(lr=lr, training_iter=training_iter, log_every=log_every)
    train_x = _to_tensor(X_train, device=device, dtype=dtype)
    train_y = _to_tensor(y_train, device=device, dtype=dtype).reshape(-1)
    _validate_training_arrays(train_x, train_y)

    if standardize_y:
        target_mean = train_y.mean()
        target_std = train_y.std(unbiased=False)
        if target_std.item() <= 0:
            raise ValueError("y_train has zero standard deviation")
        train_y_model = (train_y - target_mean) / target_std
    else:
        target_mean = torch.tensor(0.0, dtype=dtype, device=device)
        target_std = torch.tensor(1.0, dtype=dtype, device=device)
        train_y_model = train_y

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
    if initial_noise is not None:
        if initial_noise <= 0:
            raise ValueError("initial_noise must be positive")
        likelihood.noise_covar.initialize(noise=initial_noise)

    mean_module = _prepare_mean_module(
        mean_module,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        dtype=dtype,
    )

    model = ExactGPRModel(
        train_x,
        train_y_model,
        likelihood,
        kernel=kernel,
        ard_num_dims=train_x.shape[1] if ard else None,
        mean_module=mean_module,
    ).to(device=device, dtype=dtype)

    model.target_mean = target_mean.detach()
    model.target_std = target_std.detach()
    model.standardize_y = standardize_y

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_history: list[float] = []

    for iteration in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -marginal_log_likelihood(output, train_y_model)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

        if verbose and _should_log_iteration(iteration, training_iter, log_every):
            _print_training_status(iteration, training_iter, loss, likelihood, model)

    model.training_loss_history = loss_history

    return GPyTorchGPRResult(
        model=model,
        likelihood=likelihood,
        loss_history=loss_history,
        target_mean=float(target_mean.detach().cpu().item()),
        target_std=float(target_std.detach().cpu().item()),
        standardize_y=standardize_y,
        kernel=kernel,
        ard=ard,
        device=device,
        dtype=dtype,
    )


def train_gpytorch_gpr(
    X_train,
    y_train,
    *,
    kernel: str = "matern",
    ard: bool = True,
    mean_module: gpytorch.means.Mean | None = None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
    return_result: bool = False,
):
    """Train an exact GPyTorch GPR model.

    This compatibility wrapper returns ``(model, likelihood)`` by default.
    Set ``return_result=True`` to receive the richer ``GPyTorchGPRResult`` with
    loss history, target-scaling metadata, and a ``predict`` method.
    """
    result = fit_gpytorch_gpr(
        X_train,
        y_train,
        kernel=kernel,
        ard=ard,
        mean_module=mean_module,
        lr=lr,
        training_iter=training_iter,
        initial_noise=initial_noise,
        standardize_y=standardize_y,
        device=device,
        dtype=dtype,
        verbose=verbose,
        log_every=log_every,
    )
    if return_result:
        return result
    return result.model, result.likelihood


def predict_gpytorch_gpr(
    model: ExactGPRModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    return_std: bool = True,
    confidence_level: float | None = None,
    include_observation_noise: bool = True,
    return_prediction: bool = False,
):
    """Predict with a fitted GPyTorch GPR model.

    By default this keeps the earlier helper behavior: it returns
    ``(mean, std)`` when ``return_std=True`` and only ``mean`` otherwise. Set
    ``return_prediction=True`` to receive a ``GPyTorchPrediction`` object with
    optional confidence intervals.
    """
    prediction = _predict_gpytorch_gpr(
        model,
        likelihood,
        X,
        device=device,
        dtype=dtype,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
    )
    if return_prediction:
        return prediction
    if return_std:
        return prediction.mean, prediction.std
    return prediction.mean


def _predict_gpytorch_gpr(
    model: ExactGPRModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X,
    *,
    device: str,
    dtype: torch.dtype,
    return_std: bool,
    confidence_level: float | None,
    include_observation_noise: bool,
) -> GPyTorchPrediction:
    _validate_confidence_level(confidence_level)
    model.eval()
    likelihood.eval()
    test_x = _to_tensor(X, device=device, dtype=dtype)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_distribution = model(test_x)
        prediction_distribution = (
            likelihood(latent_distribution) if include_observation_noise else latent_distribution
        )
        mean = prediction_distribution.mean
        std = prediction_distribution.stddev if return_std or confidence_level is not None else None

    if hasattr(model, "target_mean") and hasattr(model, "target_std"):
        target_mean = model.target_mean.to(dtype=mean.dtype, device=mean.device)
        target_std = model.target_std.to(dtype=mean.dtype, device=mean.device)
        mean = mean * target_std + target_mean
        if std is not None:
            std = std * target_std

    mean_array = mean.detach().cpu().numpy()
    std_array = None if std is None else std.detach().cpu().numpy()
    lower = None
    upper = None

    if confidence_level is not None:
        if std_array is None:
            raise ValueError("confidence_level requires return_std=True")
        z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        lower = mean_array - z_value * std_array
        upper = mean_array + z_value * std_array

    return GPyTorchPrediction(mean=mean_array, std=std_array, lower=lower, upper=upper)


def _make_gpytorch_base_kernel(
    name: str,
    *,
    ard_num_dims: int | None = None,
) -> gpytorch.kernels.Kernel:
    normalized = name.lower().replace("-", "").replace("_", "")

    if normalized in {"rbf", "squaredexponential"}:
        return gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    if normalized in {"matern", "matern25", "matern52"}:
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    if normalized in {"matern15", "matern32"}:
        return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
    if normalized in {"matern05", "matern12", "exponential"}:
        return gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)

    raise ValueError("kernel must be one of: rbf, matern, matern15, matern05")


def _prepare_mean_module(
    mean_module: gpytorch.means.Mean | None,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> gpytorch.means.Mean:
    mean_module = mean_module or gpytorch.means.ConstantMean()
    mean_module = mean_module.to(device=device, dtype=dtype)

    if hasattr(mean_module, "set_target_standardization"):
        mean_module.set_target_standardization(target_mean, target_std)

    return mean_module


def _to_tensor(x, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert NumPy, pandas, or torch input to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    array = np.asarray(x)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=dtype, device=device)


def _resolve_feature_indices(
    feature_indices: Mapping[str, int] | None,
    column_indices: Mapping[str, int] | None,
) -> dict[str, int]:
    if feature_indices is not None and column_indices is not None:
        raise ValueError("Use either feature_indices or column_indices, not both")
    resolved = dict(feature_indices if feature_indices is not None else column_indices or {})

    for name, index in resolved.items():
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Feature index for '{name}' must be a non-negative integer")
    return resolved


def _resolve_learnable_parameters(
    learnable_parameters: Mapping[str, float] | None,
    parameter_initial_values: Mapping[str, float] | None,
) -> dict[str, float]:
    if learnable_parameters is not None and parameter_initial_values is not None:
        raise ValueError("Use either learnable_parameters or parameter_initial_values, not both")
    if learnable_parameters is not None:
        return dict(learnable_parameters)
    return dict(parameter_initial_values or {})


def _validate_numeric_mapping(values: Mapping[str, float], mapping_name: str) -> dict[str, float]:
    validated = {}
    for name, value in values.items():
        try:
            validated[name] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{mapping_name} values must be numeric; got {name}={value!r}") from error
    return validated


def _validate_positive_mapping(values: Mapping[str, float], mapping_name: str) -> dict[str, float]:
    validated = _validate_numeric_mapping(values, mapping_name)
    non_positive = [name for name, value in validated.items() if value <= 0]
    if non_positive:
        raise ValueError(f"{mapping_name} values must be positive: {non_positive}")
    return validated


def _validate_torch_parameter_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Parameter names must be non-empty strings")
    if "." in name:
        raise ValueError(f"Parameter name '{name}' cannot contain '.'")
    if name.startswith("raw_"):
        raise ValueError("Parameter names should not start with 'raw_'")


def _inverse_softplus(value: float) -> torch.Tensor:
    value_tensor = torch.tensor(float(value))
    return value_tensor + torch.log(-torch.expm1(-value_tensor))


def _validate_training_options(*, lr: float, training_iter: int, log_every: int) -> None:
    if lr <= 0:
        raise ValueError("lr must be positive")
    if training_iter <= 0:
        raise ValueError("training_iter must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")


def _validate_training_arrays(train_x: torch.Tensor, train_y: torch.Tensor) -> None:
    if train_x.ndim != 2:
        raise ValueError("X_train must be a 2D feature matrix")
    if train_y.ndim != 1:
        raise ValueError("y_train must be one-dimensional")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("X_train and y_train must contain the same number of samples")
    if train_x.shape[0] < 2:
        raise ValueError("At least two training samples are required")


def _validate_confidence_level(confidence_level: float | None) -> None:
    if confidence_level is None:
        return
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")


def _should_log_iteration(iteration: int, training_iter: int, log_every: int) -> bool:
    return iteration == 0 or iteration + 1 == training_iter or (iteration + 1) % log_every == 0


def _print_training_status(
    iteration: int,
    training_iter: int,
    loss: torch.Tensor,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    model: ExactGPRModel,
) -> None:
    noise = likelihood.noise.item()
    outputscale = model.covar_module.outputscale.item()
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    print(
        f"Iter {iteration + 1:4d}/{training_iter} | "
        f"Loss: {loss.item():.4f} | "
        f"Noise: {noise:.4e} | "
        f"Outputscale: {outputscale:.4e} | "
        f"Mean lengthscale: {lengthscale.mean():.4e}"
    )
