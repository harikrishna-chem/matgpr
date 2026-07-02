from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import gpytorch
import numpy as np
import torch

from .gpytorch_gpr import GPyTorchGPRResult, fit_gpytorch_gpr

__all__ = [
    "HeteroscedasticGPRPrediction",
    "HeteroscedasticGPRResult",
    "fit_heteroscedastic_gpr",
]


@dataclass(frozen=True)
class HeteroscedasticGPRPrediction:
    """Predictions from a two-stage heteroscedastic Gaussian-process model.

    Attributes
    ----------
    mean
        Predictive mean from the signal GP in original target units.
    std
        Total predictive standard deviation in original target units,
        combining latent signal uncertainty and learned observation noise. This
        is ``None`` when standard deviations and confidence intervals were not
        requested.
    lower, upper
        Optional lower and upper Gaussian confidence bounds built from
        ``mean`` and ``std``.
    latent_std
        Latent-function standard deviation from the signal GP, excluding
        observation noise. This is ``None`` when ``std`` is ``None``.
    noise_std
        Learned input-dependent observation-noise standard deviation.
    noise_variance
        Learned input-dependent observation-noise variance.
    log_noise_variance
        Predicted log observation-noise variance from the residual-noise GP.
    """

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    latent_std: np.ndarray | None = None
    noise_std: np.ndarray | None = None
    noise_variance: np.ndarray | None = None
    log_noise_variance: np.ndarray | None = None


@dataclass(frozen=True)
class HeteroscedasticGPRResult:
    """Container returned by ``fit_heteroscedastic_gpr``.

    This model is a practical two-stage approximation:

    1. Fit a signal GP for the target.
    2. Fit a second GP to the log squared signal residuals.

    The second GP learns input-dependent observation-noise variance. Prediction
    combines latent signal uncertainty with the learned noise variance. This is
    useful for materials datasets where measurement quality, synthesis route,
    composition region, or descriptor coverage changes across the design
    space. It is not a full joint variational heteroscedastic likelihood.
    """

    signal_result: GPyTorchGPRResult
    noise_result: GPyTorchGPRResult
    residuals: np.ndarray
    log_noise_variance_targets: np.ndarray
    residual_variance_floor: float
    noise_variance_floor: float
    signal_kernel: str
    noise_kernel: str
    ard: bool
    device: str
    dtype: torch.dtype

    @property
    def model(self):
        """Fitted signal GP model."""
        return self.signal_result.model

    @property
    def likelihood(self):
        """Fitted signal GP likelihood."""
        return self.signal_result.likelihood

    def predict(
        self,
        X,
        *,
        return_std: bool = True,
        confidence_level: float | None = None,
    ) -> HeteroscedasticGPRPrediction:
        """Predict target values with learned input-dependent noise.

        Parameters
        ----------
        X
            Feature matrix with the same column order used during fitting.
        return_std
            Whether to include total and latent predictive standard deviations.
        confidence_level
            Optional central confidence level, for example ``0.95`` for a
            95 percent interval.
        """
        _validate_confidence_level(confidence_level)
        need_std = return_std or confidence_level is not None

        signal_prediction = self.signal_result.predict(
            X,
            return_std=need_std,
            confidence_level=None,
            include_observation_noise=False,
        )
        noise_prediction = self.noise_result.predict(
            X,
            return_std=False,
            confidence_level=None,
            include_observation_noise=False,
        )

        log_noise_variance = np.asarray(noise_prediction.mean, dtype=float)
        noise_variance = np.maximum(np.exp(log_noise_variance), self.noise_variance_floor)
        noise_std = np.sqrt(noise_variance)

        latent_std = None
        total_std = None
        lower = None
        upper = None

        if need_std:
            latent_std = np.asarray(signal_prediction.std, dtype=float)
            total_std = np.sqrt(latent_std**2 + noise_variance)

        if confidence_level is not None:
            z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
            lower = signal_prediction.mean - z_value * total_std
            upper = signal_prediction.mean + z_value * total_std

        return HeteroscedasticGPRPrediction(
            mean=signal_prediction.mean,
            std=total_std,
            lower=lower,
            upper=upper,
            latent_std=latent_std,
            noise_std=noise_std,
            noise_variance=noise_variance,
            log_noise_variance=log_noise_variance,
        )


def fit_heteroscedastic_gpr(
    X_train,
    y_train,
    *,
    signal_kernel: str = "matern",
    noise_kernel: str = "matern",
    ard: bool = True,
    signal_mean_module: gpytorch.means.Mean | None = None,
    signal_lr: float = 0.01,
    noise_lr: float = 0.01,
    signal_training_iter: int = 1000,
    noise_training_iter: int = 500,
    signal_initial_noise: float | None = 0.1,
    noise_initial_noise: float | None = 0.05,
    standardize_y: bool = True,
    standardize_noise_targets: bool = True,
    residual_variance_floor: float = 1e-8,
    noise_variance_floor: float | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> HeteroscedasticGPRResult:
    """Fit a two-stage heteroscedastic Gaussian Process Regressor.

    Parameters
    ----------
    X_train, y_train
        Training features and target values. Pandas, NumPy, and torch inputs
        are accepted.
    signal_kernel, noise_kernel
        Kernel names for the target signal GP and log-noise GP. Supported names
        match ``fit_gpytorch_gpr``: ``"rbf"``, ``"matern"``,
        ``"matern15"``, and ``"matern05"``.
    ard
        If ``True``, learn one kernel length scale per feature for both GPs.
    signal_mean_module
        Optional signal mean module. Pass ``PhysicsInformedMean`` to combine a
        mechanistic prior mean with learned heteroscedastic noise.
    signal_lr, noise_lr
        Adam learning rates for the signal GP and log-noise GP.
    signal_training_iter, noise_training_iter
        Optimization iterations for the signal GP and log-noise GP.
    signal_initial_noise, noise_initial_noise
        Optional initial Gaussian likelihood noise values for each GP.
    standardize_y
        Whether to standardize target values while fitting the signal GP.
    standardize_noise_targets
        Whether to standardize log-noise targets while fitting the noise GP.
        If the log-noise targets are constant, standardization is disabled
        automatically.
    residual_variance_floor
        Positive floor added to squared residuals before taking the logarithm.
        This protects against exact interpolation and defines the smallest
        learnable residual variance in original target units squared.
    noise_variance_floor
        Optional positive floor applied to predicted observation-noise
        variances. Defaults to ``residual_variance_floor``.
    device, dtype
        Torch device and dtype used for both GP fits.
    verbose, log_every
        Training logging controls passed to both GP fits.
    """
    residual_variance_floor = float(residual_variance_floor)
    if residual_variance_floor <= 0:
        raise ValueError("residual_variance_floor must be positive")

    if noise_variance_floor is None:
        noise_variance_floor = residual_variance_floor
    noise_variance_floor = float(noise_variance_floor)
    if noise_variance_floor <= 0:
        raise ValueError("noise_variance_floor must be positive")

    y_train_array = _as_1d_numpy_array(y_train, name="y_train")

    signal_result = fit_gpytorch_gpr(
        X_train,
        y_train_array,
        kernel=signal_kernel,
        ard=ard,
        mean_module=signal_mean_module,
        lr=signal_lr,
        training_iter=signal_training_iter,
        initial_noise=signal_initial_noise,
        standardize_y=standardize_y,
        device=device,
        dtype=dtype,
        verbose=verbose,
        log_every=log_every,
    )

    in_sample_signal = signal_result.predict(
        X_train,
        return_std=False,
        confidence_level=None,
        include_observation_noise=False,
    ).mean
    residuals = y_train_array - np.asarray(in_sample_signal, dtype=float)
    log_noise_variance_targets = np.log(residuals**2 + residual_variance_floor)

    standardize_log_noise = bool(standardize_noise_targets)
    if np.std(log_noise_variance_targets) <= 0:
        standardize_log_noise = False

    noise_result = fit_gpytorch_gpr(
        X_train,
        log_noise_variance_targets,
        kernel=noise_kernel,
        ard=ard,
        mean_module=None,
        lr=noise_lr,
        training_iter=noise_training_iter,
        initial_noise=noise_initial_noise,
        standardize_y=standardize_log_noise,
        device=device,
        dtype=dtype,
        verbose=verbose,
        log_every=log_every,
    )

    return HeteroscedasticGPRResult(
        signal_result=signal_result,
        noise_result=noise_result,
        residuals=residuals,
        log_noise_variance_targets=log_noise_variance_targets,
        residual_variance_floor=residual_variance_floor,
        noise_variance_floor=noise_variance_floor,
        signal_kernel=signal_kernel,
        noise_kernel=noise_kernel,
        ard=ard,
        device=device,
        dtype=dtype,
    )


def _as_1d_numpy_array(values, *, name: str) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim > 1 and 1 not in array.shape:
        raise ValueError(f"{name} must be one-dimensional")
    array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_confidence_level(confidence_level: float | None) -> None:
    if confidence_level is None:
        return
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
