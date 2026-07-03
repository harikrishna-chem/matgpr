from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    likelihood: gpytorch.likelihoods.Likelihood
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
    noise_mode: str
    standardized_task_noise_variance: np.ndarray
    task_noise_variance: np.ndarray
    task_noise_std: np.ndarray
    standardized_observation_noise_variance: np.ndarray | None
    observation_noise_variance: np.ndarray | None
    device: str
    dtype: torch.dtype

    def predict(
        self,
        X,
        *,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool = True,
        prediction_noise_variance=None,
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
            prediction_noise_variance=prediction_noise_variance,
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
        likelihood: gpytorch.likelihoods.Likelihood,
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
    initial_task_noises: Sequence[float] | Mapping[str, float] | None = None,
    known_noise_variance=None,
    noise_mode: str = "shared",
    noise_lower_bound: float = 1e-4,
    standardize_y: bool = True,
    min_observations_per_task: int = 2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> SparseMultitaskGPyTorchResult:
    """Fit exact sparse multitask GPR from a target matrix with ``NaN`` gaps.

    Set ``noise_mode="shared"`` for one learned observation-noise variance
    across all observed task values, or ``noise_mode="task"`` for one learned
    noise variance per target task. Use ``noise_mode="known"`` with
    ``known_noise_variance`` when each observed target entry has a known
    measurement-noise variance in original target units.
    """
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
    observation_noise_variance, standardized_observation_noise_variance = _resolve_sparse_known_noise_variance(
        known_noise_variance,
        observation_data=observation_data,
        target_std=target_std.detach().cpu().numpy(),
    )

    noise_mode = _validate_sparse_noise_mode(noise_mode)
    likelihood = _make_sparse_likelihood(
        noise_mode=noise_mode,
        num_tasks=num_tasks,
        task_names=observation_data.task_names,
        initial_noise=initial_noise,
        initial_task_noises=initial_task_noises,
        standardized_known_noise_variance=standardized_observation_noise_variance,
        noise_lower_bound=noise_lower_bound,
        device=device,
        dtype=dtype,
    )
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
    model.noise_mode = noise_mode
    model.standardized_observation_noise_variance = (
        None
        if standardized_observation_noise_variance is None
        else standardized_observation_noise_variance.copy()
    )
    model.observation_noise_variance = (
        None if observation_noise_variance is None else observation_noise_variance.copy()
    )

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_history: list[float] = []

    for iteration in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x, train_task_indices)
        loss = -_sparse_marginal_log_likelihood(
            marginal_log_likelihood,
            output,
            train_y_model,
            train_task_indices,
            likelihood=likelihood,
        )
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

        if verbose and _should_log_iteration(iteration, training_iter, log_every):
            _print_sparse_multitask_training_status(iteration, training_iter, loss, likelihood, model)

    model.training_loss_history = loss_history
    standardized_noise_variance = _standardized_sparse_task_noise_variance(
        likelihood,
        noise_mode=noise_mode,
        num_tasks=num_tasks,
        task_indices=observation_data.task_indices,
        standardized_observation_noise_variance=standardized_observation_noise_variance,
    )
    task_noise_variance = standardized_noise_variance * target_std.detach().cpu().numpy() ** 2
    task_noise_std = np.sqrt(task_noise_variance)
    model.standardized_task_noise_variance = standardized_noise_variance.copy()
    model.task_noise_variance = task_noise_variance.copy()
    model.task_noise_std = task_noise_std.copy()

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
        noise_mode=noise_mode,
        standardized_task_noise_variance=standardized_noise_variance,
        task_noise_variance=task_noise_variance,
        task_noise_std=task_noise_std,
        standardized_observation_noise_variance=standardized_observation_noise_variance,
        observation_noise_variance=observation_noise_variance,
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
    initial_task_noises: Sequence[float] | Mapping[str, float] | None = None,
    known_noise_variance=None,
    noise_mode: str = "shared",
    noise_lower_bound: float = 1e-4,
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
        initial_task_noises=initial_task_noises,
        known_noise_variance=known_noise_variance,
        noise_mode=noise_mode,
        noise_lower_bound=noise_lower_bound,
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
    likelihood: gpytorch.likelihoods.Likelihood,
    X,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    return_std: bool = True,
    confidence_level: float | None = None,
    include_observation_noise: bool = True,
    prediction_noise_variance=None,
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
        prediction_noise_variance=prediction_noise_variance,
    )
    if return_prediction:
        return prediction
    if return_std:
        return prediction.mean, prediction.std
    return prediction.mean


def _predict_sparse_multitask_gpytorch_gpr(
    model: ExactSparseMultitaskGPRModel,
    likelihood: gpytorch.likelihoods.Likelihood,
    X,
    *,
    device: str,
    dtype: torch.dtype,
    return_std: bool,
    confidence_level: float | None,
    include_observation_noise: bool,
    prediction_noise_variance,
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
    standardized_prediction_noise_variance = None
    if include_observation_noise:
        standardized_prediction_noise_variance = _resolve_sparse_prediction_noise_variance(
            prediction_noise_variance,
            likelihood=likelihood,
            model=model,
            n_samples=n_samples,
            num_tasks=num_tasks,
            pred_task_indices=pred_task_indices,
            device=device,
            dtype=dtype,
        )

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_distribution = model(pred_x, pred_task_indices)
        prediction_distribution = (
            _apply_sparse_likelihood(
                likelihood,
                latent_distribution,
                pred_task_indices,
                standardized_prediction_noise_variance=standardized_prediction_noise_variance,
            )
            if include_observation_noise
            else latent_distribution
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


def _validate_sparse_noise_mode(noise_mode: str) -> str:
    normalized = str(noise_mode).strip().lower()
    aliases = {
        "shared": "shared",
        "global": "shared",
        "task": "task",
        "taskwise": "task",
        "task-wise": "task",
        "per_task": "task",
        "per-task": "task",
        "known": "known",
        "fixed": "known",
        "fixed_noise": "known",
        "fixed-noise": "known",
        "observed": "known",
        "per_observation": "known",
        "per-observation": "known",
    }
    if normalized not in aliases:
        raise ValueError("noise_mode must be one of: 'shared', 'task', or 'known'")
    return aliases[normalized]


def _validate_noise_lower_bound(noise_lower_bound: float) -> float:
    value = float(noise_lower_bound)
    if value <= 0:
        raise ValueError("noise_lower_bound must be positive")
    return value


def _make_sparse_likelihood(
    *,
    noise_mode: str,
    num_tasks: int,
    task_names: Sequence[str],
    initial_noise: float | None,
    initial_task_noises: Sequence[float] | Mapping[str, float] | None,
    standardized_known_noise_variance: np.ndarray | None,
    noise_lower_bound: float,
    device: str,
    dtype: torch.dtype,
) -> gpytorch.likelihoods.Likelihood:
    noise_lower_bound = _validate_noise_lower_bound(noise_lower_bound)
    noise_constraint = gpytorch.constraints.GreaterThan(noise_lower_bound)

    if noise_mode == "shared":
        if initial_task_noises is not None:
            raise ValueError("initial_task_noises can only be used with noise_mode='task'")
        if standardized_known_noise_variance is not None:
            raise ValueError("known_noise_variance can only be used with noise_mode='known'")
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint,
        ).to(device=device, dtype=dtype)
        _initialize_sparse_likelihood(likelihood, initial_noise=initial_noise)
        return likelihood

    if noise_mode == "known":
        if initial_task_noises is not None:
            raise ValueError("initial_task_noises can only be used with noise_mode='task'")
        if standardized_known_noise_variance is None:
            raise ValueError("known_noise_variance is required when noise_mode='known'")
        return gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.as_tensor(standardized_known_noise_variance, dtype=dtype, device=device),
            learn_additional_noise=False,
        ).to(device=device, dtype=dtype)

    if standardized_known_noise_variance is not None:
        raise ValueError("known_noise_variance can only be used with noise_mode='known'")

    likelihood = gpytorch.likelihoods.HadamardGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=noise_constraint,
    ).to(device=device, dtype=dtype)
    initial_values = _resolve_initial_task_noises(
        initial_task_noises,
        task_names=task_names,
        fallback_initial_noise=initial_noise,
    )
    if initial_values is not None:
        likelihood.initialize(
            noise=torch.as_tensor(initial_values, dtype=dtype, device=device),
        )
    return likelihood


def _resolve_sparse_known_noise_variance(
    known_noise_variance,
    *,
    observation_data: SparseMultitaskObservationData,
    target_std: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if known_noise_variance is None:
        return None, None

    values = _extract_sparse_known_noise_values(
        known_noise_variance,
        observed_mask=observation_data.observed_mask,
        n_observations=observation_data.y_observed.shape[0],
    )
    values = _validate_initial_noise_values(values, name="known_noise_variance")
    task_scale = np.asarray(target_std, dtype=float)[observation_data.task_indices] ** 2
    standardized_values = values / task_scale
    return values, standardized_values


def _extract_sparse_known_noise_values(
    known_noise_variance,
    *,
    observed_mask: np.ndarray,
    n_observations: int,
) -> np.ndarray:
    values = np.asarray(known_noise_variance, dtype=float)
    if values.ndim == 0:
        return np.full(n_observations, float(values), dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1)
        if values.shape[0] != n_observations:
            raise ValueError(
                "1D known_noise_variance must contain one value per observed target; "
                f"expected {n_observations}, got {values.shape[0]}"
            )
        return values
    if values.ndim == 2:
        if values.shape != observed_mask.shape:
            raise ValueError(
                "2D known_noise_variance must have the same shape as y_train; "
                f"expected {observed_mask.shape}, got {values.shape}"
            )
        return values[observed_mask]
    raise ValueError("known_noise_variance must be a scalar, 1D vector, or 2D target-shaped matrix")


def _initialize_sparse_likelihood(
    likelihood: gpytorch.likelihoods.Likelihood,
    *,
    initial_noise: float | None,
) -> None:
    if initial_noise is None:
        return
    if initial_noise <= 0:
        raise ValueError("initial_noise must be positive")
    likelihood.initialize(noise=float(initial_noise))


def _resolve_initial_task_noises(
    initial_task_noises: Sequence[float] | Mapping[str, float] | None,
    *,
    task_names: Sequence[str],
    fallback_initial_noise: float | None,
) -> np.ndarray | None:
    if initial_task_noises is None:
        if fallback_initial_noise is None:
            return None
        if fallback_initial_noise <= 0:
            raise ValueError("initial_noise must be positive")
        return np.full(len(task_names), float(fallback_initial_noise), dtype=float)

    if isinstance(initial_task_noises, Mapping):
        return _initial_task_noises_from_mapping(initial_task_noises, task_names=task_names)

    values = np.asarray(initial_task_noises, dtype=float).reshape(-1)
    if values.shape[0] != len(task_names):
        raise ValueError(
            "initial_task_noises must contain one value per task; "
            f"expected {len(task_names)}, got {values.shape[0]}"
        )
    return _validate_initial_noise_values(values, name="initial_task_noises")


def _initial_task_noises_from_mapping(
    initial_task_noises: Mapping[str, float],
    *,
    task_names: Sequence[str],
) -> np.ndarray:
    task_names = tuple(str(name) for name in task_names)
    provided = {str(name): value for name, value in initial_task_noises.items()}
    missing = [name for name in task_names if name not in provided]
    unknown = sorted(set(provided).difference(task_names))
    if missing or unknown:
        message_parts = []
        if missing:
            message_parts.append(f"missing task(s): {missing}")
        if unknown:
            message_parts.append(f"unknown task(s): {unknown}")
        raise ValueError("initial_task_noises mapping must match task_names; " + "; ".join(message_parts))
    values = np.asarray([provided[name] for name in task_names], dtype=float)
    return _validate_initial_noise_values(values, name="initial_task_noises")


def _validate_initial_noise_values(values: np.ndarray, *, name: str) -> np.ndarray:
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values <= 0):
        raise ValueError(f"{name} must contain only positive values")
    return values.astype(float, copy=True)


def _sparse_marginal_log_likelihood(
    marginal_log_likelihood: gpytorch.mlls.ExactMarginalLogLikelihood,
    output: gpytorch.distributions.MultivariateNormal,
    train_y: torch.Tensor,
    train_task_indices: torch.Tensor,
    *,
    likelihood: gpytorch.likelihoods.Likelihood,
) -> torch.Tensor:
    if _likelihood_uses_task_indices(likelihood):
        return marginal_log_likelihood(
            output,
            train_y,
            _task_likelihood_inputs(train_task_indices),
        )
    return marginal_log_likelihood(output, train_y)


def _apply_sparse_likelihood(
    likelihood: gpytorch.likelihoods.Likelihood,
    latent_distribution: gpytorch.distributions.MultivariateNormal,
    task_indices: torch.Tensor,
    *,
    standardized_prediction_noise_variance: torch.Tensor | None = None,
) -> gpytorch.distributions.MultivariateNormal:
    if isinstance(likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
        if standardized_prediction_noise_variance is None:
            return likelihood(latent_distribution)
        return likelihood(latent_distribution, noise=standardized_prediction_noise_variance)
    if _likelihood_uses_task_indices(likelihood):
        return likelihood(latent_distribution, _task_likelihood_inputs(task_indices))
    return likelihood(latent_distribution)


def _likelihood_uses_task_indices(likelihood: gpytorch.likelihoods.Likelihood) -> bool:
    return isinstance(likelihood, gpytorch.likelihoods.HadamardGaussianLikelihood)


def _task_likelihood_inputs(task_indices: torch.Tensor) -> torch.Tensor:
    return task_indices.long().reshape(-1, 1)


def _standardized_sparse_task_noise_variance(
    likelihood: gpytorch.likelihoods.Likelihood,
    *,
    noise_mode: str,
    num_tasks: int,
    task_indices: np.ndarray,
    standardized_observation_noise_variance: np.ndarray | None,
) -> np.ndarray:
    if noise_mode == "known":
        if standardized_observation_noise_variance is None:
            raise ValueError("known sparse likelihood requires observation noise values")
        return _mean_noise_variance_by_task(
            standardized_observation_noise_variance,
            task_indices=task_indices,
            num_tasks=num_tasks,
        )
    noise = likelihood.noise.detach().cpu().numpy().reshape(-1).astype(float)
    if noise_mode == "shared":
        if noise.size != 1:
            raise ValueError("Shared sparse likelihood must expose one noise value")
        return np.full(num_tasks, float(noise[0]), dtype=float)
    if noise.size != num_tasks:
        raise ValueError(
            "Task sparse likelihood must expose one noise value per task; "
            f"expected {num_tasks}, got {noise.size}"
        )
    return noise.copy()


def _mean_noise_variance_by_task(
    noise_variance: np.ndarray,
    *,
    task_indices: np.ndarray,
    num_tasks: int,
) -> np.ndarray:
    noise_variance = np.asarray(noise_variance, dtype=float).reshape(-1)
    task_indices = np.asarray(task_indices, dtype=int).reshape(-1)
    if noise_variance.shape[0] != task_indices.shape[0]:
        raise ValueError("noise_variance and task_indices must contain the same number of values")
    summaries = np.zeros(num_tasks, dtype=float)
    for task_index in range(num_tasks):
        task_values = noise_variance[task_indices == task_index]
        if task_values.size == 0:
            raise ValueError(f"No noise values available for task index {task_index}")
        summaries[task_index] = float(task_values.mean())
    return summaries


def _resolve_sparse_prediction_noise_variance(
    prediction_noise_variance,
    *,
    likelihood: gpytorch.likelihoods.Likelihood,
    model: ExactSparseMultitaskGPRModel,
    n_samples: int,
    num_tasks: int,
    pred_task_indices: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not isinstance(likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
        if prediction_noise_variance is not None:
            raise ValueError("prediction_noise_variance can only be used with noise_mode='known'")
        return None

    if prediction_noise_variance is None:
        if not hasattr(model, "standardized_task_noise_variance"):
            return None
        task_noise = np.asarray(model.standardized_task_noise_variance, dtype=float).reshape(-1)
        values = task_noise[pred_task_indices.detach().cpu().numpy()]
        return torch.as_tensor(values, dtype=dtype, device=device)

    values = _extract_sparse_prediction_noise_values(
        prediction_noise_variance,
        n_samples=n_samples,
        num_tasks=num_tasks,
    )
    values = _validate_initial_noise_values(values, name="prediction_noise_variance")
    target_std = model.target_std.detach().cpu().numpy().reshape(-1)
    task_scale = target_std[pred_task_indices.detach().cpu().numpy()] ** 2
    standardized_values = values / task_scale
    return torch.as_tensor(standardized_values, dtype=dtype, device=device)


def _extract_sparse_prediction_noise_values(
    prediction_noise_variance,
    *,
    n_samples: int,
    num_tasks: int,
) -> np.ndarray:
    expected = n_samples * num_tasks
    values = np.asarray(prediction_noise_variance, dtype=float)
    if values.ndim == 0:
        return np.full(expected, float(values), dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1)
        if values.shape[0] == num_tasks:
            return np.tile(values, n_samples)
        if values.shape[0] == expected:
            return values
        raise ValueError(
            "1D prediction_noise_variance must contain either one value per "
            f"task ({num_tasks}) or one dense prediction value ({expected}); "
            f"got {values.shape[0]}"
        )
    if values.ndim == 2:
        if values.shape != (n_samples, num_tasks):
            raise ValueError(
                "2D prediction_noise_variance must have shape "
                f"({n_samples}, {num_tasks}); got {values.shape}"
            )
        return values.reshape(-1)
    raise ValueError("prediction_noise_variance must be a scalar, 1D vector, or 2D matrix")


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
    likelihood: gpytorch.likelihoods.Likelihood,
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
