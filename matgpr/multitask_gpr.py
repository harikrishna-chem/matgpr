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

__all__ = [
    "ExactMultitaskGPRModel",
    "MultitaskGPyTorchPrediction",
    "MultitaskGPyTorchResult",
    "fit_multitask_gpytorch_gpr",
    "predict_multitask_gpytorch_gpr",
    "train_multitask_gpytorch_gpr",
]


@dataclass(frozen=True)
class MultitaskGPyTorchPrediction:
    """Predictions from a fitted multitask GPyTorch GPR model.

    Attributes
    ----------
    mean
        Predictive means with shape ``(n_samples, n_tasks)`` in the original
        target units.
    std
        Predictive standard deviations with shape ``(n_samples, n_tasks)`` in
        the original target units. This is ``None`` when standard deviations
        were not requested.
    lower, upper
        Optional lower and upper confidence bounds with shape
        ``(n_samples, n_tasks)``. These are populated when
        ``confidence_level`` is supplied.
    task_names
        Names of the modeled tasks in target-column order.
    """

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    task_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultitaskGPyTorchResult:
    """Container returned by :func:`fit_multitask_gpytorch_gpr`.

    This result keeps the fitted multitask GP, likelihood, task metadata, and
    training diagnostics together. Use :meth:`predict` for new samples, or
    access ``model`` and ``likelihood`` directly for lower-level GPyTorch
    operations.
    """

    model: ExactMultitaskGPRModel
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood
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
        """Predict all modeled tasks for new samples.

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
            If ``True``, uncertainty includes the fitted multitask likelihood
            noise. If ``False``, uncertainty comes from the latent multitask GP.
        """
        return _predict_multitask_gpytorch_gpr(
            self.model,
            self.likelihood,
            X,
            device=self.device,
            dtype=self.dtype,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )


class ExactMultitaskGPRModel(gpytorch.models.ExactGP):
    """Exact multitask Gaussian Process Regression model using GPyTorch.

    The model uses a shared input-space kernel and a learned task covariance.
    This is appropriate when every training row has all target tasks observed,
    for example multiple measured properties for the same material candidate.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        *,
        num_tasks: int,
        task_covar_rank: int = 1,
        kernel: str = "matern",
        ard_num_dims: int | None = None,
        mean_module: gpytorch.means.Mean | None = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = int(num_tasks)
        self.task_covar_rank = int(task_covar_rank)
        self.mean_module = mean_module or gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=self.num_tasks,
        )
        data_covar_module = gpytorch.kernels.ScaleKernel(
            _make_gpytorch_base_kernel(kernel, ard_num_dims=ard_num_dims)
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_covar_module=data_covar_module,
            num_tasks=self.num_tasks,
            rank=self.task_covar_rank,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def fit_multitask_gpytorch_gpr(
    X_train,
    y_train,
    *,
    task_names: Sequence[str] | None = None,
    task_covar_rank: int = 1,
    kernel: str = "matern",
    ard: bool = True,
    mean_module: gpytorch.means.Mean | None = None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    initial_task_noises: Sequence[float] | None = None,
    standardize_y: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> MultitaskGPyTorchResult:
    """Fit an exact multitask GPyTorch Gaussian Process Regressor.

    Parameters
    ----------
    X_train
        Two-dimensional feature matrix. Pandas, NumPy, and torch inputs are
        accepted.
    y_train
        Two-dimensional target matrix with shape ``(n_samples, n_tasks)``.
        This first multitask implementation expects every task to be observed
        for every training sample.
    task_names
        Optional names for the target columns. If omitted, tasks are named
        ``task_0``, ``task_1``, and so on.
    task_covar_rank
        Rank of the learned inter-task covariance factor. Use ``1`` as a
        conservative low-data default; larger values are more flexible but need
        more data.
    kernel
        Base input-space kernel name: ``"rbf"``, ``"matern"``,
        ``"matern15"``, or ``"matern05"``.
    ard
        If ``True``, learn one input length scale per feature.
    mean_module
        Optional multitask mean module. By default, the model uses one learned
        constant mean per task.
    lr
        Adam learning rate.
    training_iter
        Number of optimization iterations.
    initial_noise
        Optional shared likelihood-noise initialization.
    initial_task_noises
        Optional per-task likelihood-noise initialization in target-column
        order.
    standardize_y
        Whether to standardize each target task during optimization.
        Predictions are always returned in original target units.
    """
    _validate_training_options(lr=lr, training_iter=training_iter, log_every=log_every)
    train_x = _to_tensor(X_train, device=device, dtype=dtype)
    train_y = _to_tensor(y_train, device=device, dtype=dtype)
    _validate_multitask_training_arrays(train_x, train_y)

    num_tasks = train_y.shape[1]
    task_names_resolved = _resolve_task_names(task_names, num_tasks)
    task_covar_rank = _validate_task_covar_rank(task_covar_rank, num_tasks)

    if standardize_y:
        target_mean = train_y.mean(dim=0)
        target_std = train_y.std(dim=0, unbiased=False)
        zero_std_mask = target_std <= 0
        if torch.any(zero_std_mask):
            zero_std_tasks = [
                name
                for name, is_zero in zip(task_names_resolved, zero_std_mask.detach().cpu(), strict=True)
                if bool(is_zero)
            ]
            raise ValueError(f"y_train has zero standard deviation for task(s): {zero_std_tasks}")
        train_y_model = (train_y - target_mean) / target_std
    else:
        target_mean = torch.zeros(num_tasks, dtype=dtype, device=device)
        target_std = torch.ones(num_tasks, dtype=dtype, device=device)
        train_y_model = train_y

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
    ).to(device=device, dtype=dtype)
    _initialize_multitask_likelihood(
        likelihood,
        initial_noise=initial_noise,
        initial_task_noises=initial_task_noises,
        num_tasks=num_tasks,
        device=device,
        dtype=dtype,
    )

    if mean_module is not None:
        mean_module = mean_module.to(device=device, dtype=dtype)

    model = ExactMultitaskGPRModel(
        train_x,
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
    model.task_names = task_names_resolved

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
            _print_multitask_training_status(iteration, training_iter, loss, likelihood, model)

    model.training_loss_history = loss_history

    return MultitaskGPyTorchResult(
        model=model,
        likelihood=likelihood,
        loss_history=loss_history,
        target_mean=target_mean.detach().cpu().numpy(),
        target_std=target_std.detach().cpu().numpy(),
        standardize_y=standardize_y,
        kernel=kernel,
        ard=ard,
        num_tasks=num_tasks,
        task_names=task_names_resolved,
        task_covar_rank=task_covar_rank,
        device=device,
        dtype=dtype,
    )


def train_multitask_gpytorch_gpr(
    X_train,
    y_train,
    *,
    task_names: Sequence[str] | None = None,
    task_covar_rank: int = 1,
    kernel: str = "matern",
    ard: bool = True,
    mean_module: gpytorch.means.Mean | None = None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    initial_task_noises: Sequence[float] | None = None,
    standardize_y: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
    return_result: bool = False,
):
    """Train an exact multitask GPyTorch GPR model.

    This compatibility wrapper returns ``(model, likelihood)`` by default.
    Set ``return_result=True`` to receive the richer
    :class:`MultitaskGPyTorchResult`.
    """
    result = fit_multitask_gpytorch_gpr(
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
        standardize_y=standardize_y,
        device=device,
        dtype=dtype,
        verbose=verbose,
        log_every=log_every,
    )
    if return_result:
        return result
    return result.model, result.likelihood


def predict_multitask_gpytorch_gpr(
    model: ExactMultitaskGPRModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    X,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    return_std: bool = True,
    confidence_level: float | None = None,
    include_observation_noise: bool = True,
    return_prediction: bool = False,
):
    """Predict all tasks with a fitted multitask GPyTorch GPR model.

    By default this returns ``(mean, std)`` when ``return_std=True`` and only
    ``mean`` otherwise. Set ``return_prediction=True`` to receive a
    :class:`MultitaskGPyTorchPrediction` object with task names and optional
    confidence intervals.
    """
    prediction = _predict_multitask_gpytorch_gpr(
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


def _predict_multitask_gpytorch_gpr(
    model: ExactMultitaskGPRModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
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
    _validate_prediction_features(test_x, model)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_distribution = model(test_x)
        prediction_distribution = (
            likelihood(latent_distribution) if include_observation_noise else latent_distribution
        )
        mean = prediction_distribution.mean
        std = prediction_distribution.stddev if return_std or confidence_level is not None else None

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


def _validate_multitask_training_arrays(train_x: torch.Tensor, train_y: torch.Tensor) -> None:
    if train_x.ndim != 2:
        raise ValueError("X_train must be a 2D feature matrix")
    if train_y.ndim != 2:
        raise ValueError("y_train must be a 2D target matrix with shape (n_samples, n_tasks)")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("X_train and y_train must contain the same number of samples")
    if train_x.shape[0] < 2:
        raise ValueError(
            f"At least two training samples are required; got n_samples = {train_x.shape[0]}"
        )
    if train_y.shape[1] < 2:
        raise ValueError(
            "Multitask GPR requires at least two target tasks; "
            f"got n_tasks = {train_y.shape[1]}"
        )
    if not torch.isfinite(train_x).all():
        raise ValueError("X_train must contain only finite values")
    if not torch.isfinite(train_y).all():
        raise ValueError(
            "y_train must contain only finite values. This multitask implementation "
            "expects complete target observations."
        )


def _validate_prediction_features(test_x: torch.Tensor, model: ExactMultitaskGPRModel) -> None:
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


def _resolve_task_names(task_names: Sequence[str] | None, num_tasks: int) -> tuple[str, ...]:
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


def _initialize_multitask_likelihood(
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    *,
    initial_noise: float | None,
    initial_task_noises: Sequence[float] | None,
    num_tasks: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    if initial_noise is not None:
        if initial_noise <= 0:
            raise ValueError("initial_noise must be positive")
        likelihood.initialize(noise=float(initial_noise))

    if initial_task_noises is None:
        return
    task_noises = torch.as_tensor(initial_task_noises, dtype=dtype, device=device).reshape(-1)
    if task_noises.shape[0] != num_tasks:
        raise ValueError(
            f"initial_task_noises must contain {num_tasks} values; got {task_noises.shape[0]}"
        )
    if torch.any(task_noises <= 0):
        raise ValueError("initial_task_noises must all be positive")
    likelihood.initialize(task_noises=task_noises)


def _print_multitask_training_status(
    iteration: int,
    training_iter: int,
    loss: torch.Tensor,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    model: ExactMultitaskGPRModel,
) -> None:
    shared_noise = float(likelihood.noise.detach().cpu().reshape(-1).mean().item())
    task_noises = likelihood.task_noises.detach().cpu().numpy()
    data_kernel = model.covar_module.data_covar_module
    outputscale = float(data_kernel.outputscale.detach().cpu().item())
    lengthscale = data_kernel.base_kernel.lengthscale.detach().cpu().numpy()
    print(
        f"Iter {iteration + 1:4d}/{training_iter} | "
        f"Loss: {loss.item():.4f} | "
        f"Shared noise: {shared_noise:.4e} | "
        f"Mean task noise: {task_noises.mean():.4e} | "
        f"Outputscale: {outputscale:.4e} | "
        f"Mean lengthscale: {lengthscale.mean():.4e}"
    )
