from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import gpytorch
import numpy as np
import torch

from .gpytorch_gpr import (
    _make_gpytorch_base_kernel,
    _should_log_iteration,
    _to_tensor,
    _validate_confidence_level,
    _validate_training_options,
)
from .multitask_gpr import MultitaskGPyTorchPrediction

__all__ = [
    "ExactSparseMultitaskGPRModel",
    "SparseMultitaskGPyTorchResult",
    "SparseMultitaskObservationData",
    "fit_sparse_multitask_gpytorch_gpr",
    "prepare_sparse_multitask_observations",
    "predict_sparse_multitask_gpytorch_gpr",
    "train_sparse_multitask_gpytorch_gpr",
]


@dataclass(frozen=True)
class SparseMultitaskObservationData:
    """Observed entries extracted from a sparse multitask target matrix.

    Attributes
    ----------
    X_observed
        Feature rows repeated once for each observed task value.
    y_observed
        One-dimensional observed target values.
    task_indices
        Integer task index for each observed value.
    sample_indices
        Original sample-row index for each observed value.
    observed_mask
        Boolean matrix with shape ``(n_samples, n_tasks)`` marking finite
        target observations.
    task_names
        Task names in target-column order.
    task_observation_counts
        Number of observed values for each task.
    """

    X_observed: np.ndarray
    y_observed: np.ndarray
    task_indices: np.ndarray
    sample_indices: np.ndarray
    observed_mask: np.ndarray
    task_names: tuple[str, ...]
    task_observation_counts: np.ndarray


@dataclass(frozen=True)
class SparseMultitaskGPyTorchResult:
    """Container returned by :func:`fit_sparse_multitask_gpytorch_gpr`."""

    model: ExactSparseMultitaskGPRModel
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    observation_data: SparseMultitaskObservationData
    loss_history: list[float]
    target_mean: np.ndarray
    target_std: np.ndarray
    standardize_y: bool
    kernel: str
    ard: bool
    num_tasks: int
    task_names: tuple[str, ...]
    task_covar_rank: int
    device: str
    dtype: torch.dtype

    def predict(
        self,
        X,
        *,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool = True,
    ) -> MultitaskGPyTorchPrediction:
        """Predict all modeled tasks for new samples."""
        return _predict_sparse_multitask_gpytorch_gpr(
            self.model,
            self.likelihood,
            X,
            device=self.device,
            dtype=self.dtype,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )


class SparseTaskConstantMean(torch.nn.Module):
    """Learn one constant mean offset per task for sparse observations."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.register_parameter(
            name="task_means",
            param=torch.nn.Parameter(torch.zeros(int(num_tasks))),
        )

    def forward(self, x: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        del x
        return self.task_means[task_indices.long()]


class ExactSparseMultitaskGPRModel(gpytorch.models.ExactGP):
    """Exact sparse multitask GPR using observed ``(x, task)`` pairs.

    Unlike :class:`matgpr.ExactMultitaskGPRModel`, this model does not require
    every task to be observed for every sample. Each finite target entry is
    represented as one scalar observation with covariance

    ``k((x, i), (x', j)) = k_x(x, x') k_task(i, j)``.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_task_indices: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        num_tasks: int,
        task_covar_rank: int = 1,
        kernel: str = "matern",
        ard_num_dims: int | None = None,
        mean_module: torch.nn.Module | None = None,
    ):
        super().__init__((train_x, train_task_indices), train_y, likelihood)
        self.num_tasks = int(num_tasks)
        self.task_covar_rank = int(task_covar_rank)
        self.mean_module = mean_module or SparseTaskConstantMean(self.num_tasks)
        self.data_covar_module = gpytorch.kernels.ScaleKernel(
            _make_gpytorch_base_kernel(kernel, ard_num_dims=ard_num_dims)
        )
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=self.num_tasks,
            rank=self.task_covar_rank,
        )

    def forward(
        self,
        x: torch.Tensor,
        task_indices: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        task_indices = task_indices.long().reshape(-1)
        mean_x = _call_sparse_mean_module(self.mean_module, x, task_indices)
        covar_x = self.data_covar_module(x).mul(self.task_covar_module(task_indices))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def prepare_sparse_multitask_observations(
    X,
    y,
    *,
    task_names: Sequence[str] | None = None,
    min_observations_per_task: int = 2,
) -> SparseMultitaskObservationData:
    """Convert a target matrix with missing entries into observed task values.

    Missing targets are represented by ``NaN`` values in ``y``. Feature values
    must already be finite because every observed task for a row shares the
    same feature vector.
    """
    X_array = _to_2d_numpy(X, "X")
    y_array = _to_2d_numpy(y, "y", allow_nan=True)
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if y_array.shape[1] < 2:
        raise ValueError("Sparse multitask GPR requires at least two target columns")
    if np.isinf(y_array).any():
        raise ValueError("y must not contain infinite values")

    task_names_resolved = _resolve_sparse_task_names(task_names, y, y_array.shape[1])
    min_observations_per_task = _validate_min_observations_per_task(min_observations_per_task)
    observed_mask = np.isfinite(y_array)
    task_counts = observed_mask.sum(axis=0).astype(int)
    low_count_tasks = [
        name
        for name, count in zip(task_names_resolved, task_counts, strict=True)
        if count < min_observations_per_task
    ]
    if low_count_tasks:
        raise ValueError(
            "Each task must have at least "
            f"{min_observations_per_task} observed values; low-count task(s): "
            f"{low_count_tasks}"
        )

    sample_indices, task_indices = np.where(observed_mask)
    if sample_indices.size < 2:
        raise ValueError("At least two observed target values are required")

    return SparseMultitaskObservationData(
        X_observed=X_array[sample_indices].copy(),
        y_observed=y_array[sample_indices, task_indices].astype(float, copy=True),
        task_indices=task_indices.astype(int, copy=True),
        sample_indices=sample_indices.astype(int, copy=True),
        observed_mask=observed_mask.copy(),
        task_names=task_names_resolved,
        task_observation_counts=task_counts,
    )


def fit_sparse_multitask_gpytorch_gpr(
    X_train,
    y_train,
    *,
    task_names: Sequence[str] | None = None,
    task_covar_rank: int = 1,
    kernel: str = "matern",
    ard: bool = True,
    mean_module: torch.nn.Module | None = None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    min_observations_per_task: int = 2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> SparseMultitaskGPyTorchResult:
    """Fit exact sparse multitask GPR from a target matrix with ``NaN`` gaps."""
    _validate_training_options(lr=lr, training_iter=training_iter, log_every=log_every)
    observation_data = prepare_sparse_multitask_observations(
        X_train,
        y_train,
        task_names=task_names,
        min_observations_per_task=min_observations_per_task,
    )
    train_x = _to_tensor(observation_data.X_observed, device=device, dtype=dtype)
    train_task_indices = torch.as_tensor(
        observation_data.task_indices,
        dtype=torch.long,
        device=device,
    )
    train_y = _to_tensor(observation_data.y_observed, device=device, dtype=dtype)
    num_tasks = len(observation_data.task_names)
    task_covar_rank = _validate_task_covar_rank(task_covar_rank, num_tasks)

    target_mean, target_std = _sparse_target_statistics(
        train_y,
        train_task_indices,
        num_tasks=num_tasks,
        standardize_y=standardize_y,
    )
    train_y_model = (train_y - target_mean[train_task_indices]) / target_std[train_task_indices]

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
    _initialize_sparse_likelihood(likelihood, initial_noise=initial_noise)
    if mean_module is not None:
        mean_module = mean_module.to(device=device, dtype=dtype)

    model = ExactSparseMultitaskGPRModel(
        train_x,
        train_task_indices,
        train_y_model,
        likelihood,
        num_tasks=num_tasks,
        task_covar_rank=task_covar_rank,
        kernel=kernel,
        ard_num_dims=train_x.shape[1] if ard else None,
        mean_module=mean_module,
    ).to(device=device, dtype=dtype)
    model.target_mean = target_mean.detach()
    model.target_std = target_std.detach()
    model.standardize_y = standardize_y
    model.task_names = observation_data.task_names

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_history: list[float] = []

    for iteration in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x, train_task_indices)
        loss = -marginal_log_likelihood(output, train_y_model)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

        if verbose and _should_log_iteration(iteration, training_iter, log_every):
            _print_sparse_multitask_training_status(iteration, training_iter, loss, likelihood, model)

    model.training_loss_history = loss_history

    return SparseMultitaskGPyTorchResult(
        model=model,
        likelihood=likelihood,
        observation_data=observation_data,
        loss_history=loss_history,
        target_mean=target_mean.detach().cpu().numpy(),
        target_std=target_std.detach().cpu().numpy(),
        standardize_y=standardize_y,
        kernel=kernel,
        ard=ard,
        num_tasks=num_tasks,
        task_names=observation_data.task_names,
        task_covar_rank=task_covar_rank,
        device=device,
        dtype=dtype,
    )


def train_sparse_multitask_gpytorch_gpr(
    X_train,
    y_train,
    *,
    task_names: Sequence[str] | None = None,
    task_covar_rank: int = 1,
    kernel: str = "matern",
    ard: bool = True,
    mean_module: torch.nn.Module | None = None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    min_observations_per_task: int = 2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
    return_result: bool = False,
):
    """Train sparse multitask GPR and return ``(model, likelihood)`` by default."""
    result = fit_sparse_multitask_gpytorch_gpr(
        X_train,
        y_train,
        task_names=task_names,
        task_covar_rank=task_covar_rank,
        kernel=kernel,
        ard=ard,
        mean_module=mean_module,
        lr=lr,
        training_iter=training_iter,
        initial_noise=initial_noise,
        standardize_y=standardize_y,
        min_observations_per_task=min_observations_per_task,
        device=device,
        dtype=dtype,
        verbose=verbose,
        log_every=log_every,
    )
    if return_result:
        return result
    return result.model, result.likelihood


def predict_sparse_multitask_gpytorch_gpr(
    model: ExactSparseMultitaskGPRModel,
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
    """Predict every modeled task for each new feature row."""
    prediction = _predict_sparse_multitask_gpytorch_gpr(
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


def _predict_sparse_multitask_gpytorch_gpr(
    model: ExactSparseMultitaskGPRModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X,
    *,
    device: str,
    dtype: torch.dtype,
    return_std: bool,
    confidence_level: float | None,
    include_observation_noise: bool,
) -> MultitaskGPyTorchPrediction:
    _validate_confidence_level(confidence_level)
    model.eval()
    likelihood.eval()
    test_x = _to_tensor(X, device=device, dtype=dtype)
    _validate_sparse_prediction_features(test_x, model)
    n_samples = test_x.shape[0]
    num_tasks = int(model.num_tasks)
    pred_x = test_x.repeat_interleave(num_tasks, dim=0)
    pred_task_indices = torch.arange(num_tasks, device=device, dtype=torch.long).repeat(n_samples)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_distribution = model(pred_x, pred_task_indices)
        prediction_distribution = (
            likelihood(latent_distribution) if include_observation_noise else latent_distribution
        )
        mean = prediction_distribution.mean
        std = prediction_distribution.stddev if return_std or confidence_level is not None else None

    mean = mean.reshape(n_samples, num_tasks)
    if std is not None:
        std = std.reshape(n_samples, num_tasks)

    if hasattr(model, "target_mean") and hasattr(model, "target_std"):
        target_mean = model.target_mean.to(dtype=mean.dtype, device=mean.device).reshape(1, -1)
        target_std = model.target_std.to(dtype=mean.dtype, device=mean.device).reshape(1, -1)
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

    return MultitaskGPyTorchPrediction(
        mean=mean_array,
        std=std_array,
        lower=lower,
        upper=upper,
        task_names=getattr(model, "task_names", ()),
    )


def _sparse_target_statistics(
    train_y: torch.Tensor,
    train_task_indices: torch.Tensor,
    *,
    num_tasks: int,
    standardize_y: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not standardize_y:
        return (
            torch.zeros(num_tasks, dtype=train_y.dtype, device=train_y.device),
            torch.ones(num_tasks, dtype=train_y.dtype, device=train_y.device),
        )

    target_mean = torch.zeros(num_tasks, dtype=train_y.dtype, device=train_y.device)
    target_std = torch.ones(num_tasks, dtype=train_y.dtype, device=train_y.device)
    zero_std_tasks: list[int] = []
    for task_index in range(num_tasks):
        values = train_y[train_task_indices == task_index]
        target_mean[task_index] = values.mean()
        target_std[task_index] = values.std(unbiased=False)
        if target_std[task_index] <= 0:
            zero_std_tasks.append(task_index)
    if zero_std_tasks:
        raise ValueError(f"y_train has zero standard deviation for task index(es): {zero_std_tasks}")
    return target_mean, target_std


def _call_sparse_mean_module(
    mean_module: torch.nn.Module,
    x: torch.Tensor,
    task_indices: torch.Tensor,
) -> torch.Tensor:
    try:
        return mean_module(x, task_indices)
    except TypeError:
        return mean_module(x)


def _to_2d_numpy(values, name: str, *, allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    finite_or_nan = np.isfinite(array) | np.isnan(array)
    if allow_nan:
        if not finite_or_nan.all():
            raise ValueError(f"{name} contains unsupported non-finite values")
    elif not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _resolve_sparse_task_names(
    task_names: Sequence[str] | None,
    y,
    num_tasks: int,
) -> tuple[str, ...]:
    if task_names is None and hasattr(y, "columns"):
        task_names = tuple(str(column) for column in y.columns)
    if task_names is None:
        return tuple(f"task_{index}" for index in range(num_tasks))
    resolved = tuple(str(name) for name in task_names)
    if len(resolved) != num_tasks:
        raise ValueError(f"task_names must contain {num_tasks} names; got {len(resolved)}")
    if any(not name for name in resolved):
        raise ValueError("task_names must be non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError("task_names must be unique")
    return resolved


def _validate_task_covar_rank(task_covar_rank: int, num_tasks: int) -> int:
    if not isinstance(task_covar_rank, int):
        raise ValueError("task_covar_rank must be an integer")
    if task_covar_rank < 1:
        raise ValueError("task_covar_rank must be at least 1")
    if task_covar_rank > num_tasks:
        raise ValueError("task_covar_rank cannot be larger than the number of tasks")
    return task_covar_rank


def _validate_min_observations_per_task(value: int) -> int:
    if not isinstance(value, int):
        raise ValueError("min_observations_per_task must be an integer")
    if value < 1:
        raise ValueError("min_observations_per_task must be at least 1")
    return value


def _initialize_sparse_likelihood(
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    *,
    initial_noise: float | None,
) -> None:
    if initial_noise is None:
        return
    if initial_noise <= 0:
        raise ValueError("initial_noise must be positive")
    likelihood.initialize(noise=float(initial_noise))


def _validate_sparse_prediction_features(
    test_x: torch.Tensor,
    model: ExactSparseMultitaskGPRModel,
) -> None:
    if test_x.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    train_x = model.train_inputs[0]
    if test_x.shape[1] != train_x.shape[1]:
        raise ValueError(
            f"X has {test_x.shape[1]} features, but the model was fitted with "
            f"{train_x.shape[1]} features"
        )
    if not torch.isfinite(test_x).all():
        raise ValueError("X must contain only finite values")


def _print_sparse_multitask_training_status(
    iteration: int,
    training_iter: int,
    loss: torch.Tensor,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    model: ExactSparseMultitaskGPRModel,
) -> None:
    noise = float(likelihood.noise.detach().cpu().reshape(-1).mean().item())
    outputscale = float(model.data_covar_module.outputscale.detach().cpu().item())
    lengthscale = model.data_covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    print(
        f"Iter {iteration + 1:4d}/{training_iter} | "
        f"Loss: {loss.item():.4f} | "
        f"Noise: {noise:.4e} | "
        f"Outputscale: {outputscale:.4e} | "
        f"Mean lengthscale: {lengthscale.mean():.4e}"
    )
