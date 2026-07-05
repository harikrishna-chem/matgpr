from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, column_or_1d

from .gpytorch_gpr import GPyTorchGPRResult, fit_gpytorch_gpr

__all__ = [
    "DeltaMultiFidelityGPRResult",
    "MultiFidelityGPRPrediction",
    "MultiFidelityGPRRegressor",
    "fit_delta_multifidelity_gpr",
]


@dataclass(frozen=True)
class MultiFidelityGPRPrediction:
    """Prediction from a delta multi-fidelity Gaussian-process model.

    Attributes
    ----------
    mean, std, lower, upper
        High-fidelity predictive mean, standard deviation, and optional
        confidence interval in original high-fidelity target units.
    low_fidelity_mean, low_fidelity_std
        Low-fidelity values or low-fidelity surrogate predictions used in the
        autoregressive correction.
    correction_mean, correction_std
        GP correction term ``delta(x)`` and its uncertainty.
    rho, intercept
        Fitted linear mapping from low fidelity to high fidelity.
    """

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    low_fidelity_mean: np.ndarray | None = None
    low_fidelity_std: np.ndarray | None = None
    correction_mean: np.ndarray | None = None
    correction_std: np.ndarray | None = None
    rho: float = 1.0
    intercept: float = 0.0


@dataclass(frozen=True)
class DeltaMultiFidelityGPRResult:
    """Fitted two-stage delta multi-fidelity GPR result."""

    correction_model: GPyTorchGPRResult
    low_fidelity_model: GPyTorchGPRResult | None
    rho: float
    intercept: float
    high_fidelity_target_mean: float
    high_fidelity_target_std: float
    correction_target: np.ndarray
    low_fidelity_at_high: np.ndarray
    fit_intercept: bool
    include_low_fidelity_uncertainty: bool
    device: str
    dtype: torch.dtype

    def predict(
        self,
        X,
        *,
        low_fidelity=None,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool = True,
        include_low_fidelity_uncertainty: bool | None = None,
    ) -> MultiFidelityGPRPrediction:
        """Predict high-fidelity values for new samples."""
        return _predict_delta_multifidelity_gpr(
            self,
            X,
            low_fidelity=low_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
            include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        )


def fit_delta_multifidelity_gpr(
    X_high,
    y_high,
    *,
    low_fidelity_high=None,
    X_low=None,
    y_low=None,
    fit_intercept: bool = True,
    correction_kernel: str = "matern",
    low_fidelity_kernel: str = "matern",
    ard: bool = True,
    lr: float = 0.01,
    training_iter: int = 1000,
    low_fidelity_training_iter: int | None = None,
    correction_initial_noise: float | None = 0.1,
    low_fidelity_initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    include_low_fidelity_uncertainty: bool = True,
    device: str = "cpu",
    dtype: torch.dtype | str = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> DeltaMultiFidelityGPRResult:
    """Fit a delta multi-fidelity GPR model.

    The model uses the autoregressive correction

    ``y_high(x) = rho * y_low(x) + intercept + delta(x)``,

    where ``delta(x)`` is modeled by a standard exact GPR. Provide
    ``low_fidelity_high`` when low-fidelity values are already available at the
    high-fidelity training points. Alternatively, provide ``X_low`` and
    ``y_low`` to fit a low-fidelity GPR surrogate first.
    """
    dtype = _resolve_torch_dtype(dtype)
    X_high_array = _to_2d_numpy(X_high, "X_high")
    y_high_array = _to_1d_numpy(y_high, "y_high")
    check_consistent_length(X_high_array, y_high_array)
    _validate_feature_width_match(X_high_array, X_low, "X_high", "X_low")

    low_model: GPyTorchGPRResult | None = None
    if low_fidelity_high is None:
        if X_low is None or y_low is None:
            raise ValueError(
                "Provide low_fidelity_high, or provide both X_low and y_low to fit "
                "a low-fidelity surrogate"
            )
        low_model = fit_gpytorch_gpr(
            X_low,
            y_low,
            kernel=low_fidelity_kernel,
            ard=ard,
            lr=lr,
            training_iter=low_fidelity_training_iter or training_iter,
            initial_noise=low_fidelity_initial_noise,
            standardize_y=standardize_y,
            device=device,
            dtype=dtype,
            verbose=verbose,
            log_every=log_every,
        )
        low_prediction = low_model.predict(
            X_high_array,
            return_std=True,
            include_observation_noise=False,
        )
        low_fidelity_high_array = low_prediction.mean
    else:
        if X_low is not None or y_low is not None:
            raise ValueError("Use either low_fidelity_high or X_low/y_low, not both")
        low_fidelity_high_array = _to_1d_numpy(low_fidelity_high, "low_fidelity_high")

    check_consistent_length(X_high_array, low_fidelity_high_array)

    rho, intercept = _fit_linear_fidelity_map(
        low_fidelity_high_array,
        y_high_array,
        fit_intercept=fit_intercept,
    )
    correction_target = y_high_array - (rho * low_fidelity_high_array + intercept)
    standardize_correction = standardize_y and np.std(correction_target) > 0.0
    correction_model = fit_gpytorch_gpr(
        X_high_array,
        correction_target,
        kernel=correction_kernel,
        ard=ard,
        lr=lr,
        training_iter=training_iter,
        initial_noise=correction_initial_noise,
        standardize_y=standardize_correction,
        device=device,
        dtype=dtype,
        verbose=verbose,
        log_every=log_every,
    )

    return DeltaMultiFidelityGPRResult(
        correction_model=correction_model,
        low_fidelity_model=low_model,
        rho=float(rho),
        intercept=float(intercept),
        high_fidelity_target_mean=float(y_high_array.mean()),
        high_fidelity_target_std=float(y_high_array.std(ddof=0)),
        correction_target=correction_target.copy(),
        low_fidelity_at_high=low_fidelity_high_array.copy(),
        fit_intercept=fit_intercept,
        include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        device=device,
        dtype=dtype,
    )


class MultiFidelityGPRRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-style delta multi-fidelity GPR estimator.

    The estimator models scarce high-fidelity measurements by combining
    low-fidelity values with a GPR correction. Pass ``low_fidelity`` to
    :meth:`fit` and :meth:`predict` when low-fidelity values are externally
    available. If ``X_low`` and ``y_low`` are passed to :meth:`fit`, the
    estimator fits an internal low-fidelity surrogate and can predict without
    explicit low-fidelity values.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        correction_kernel: str = "matern",
        low_fidelity_kernel: str = "matern",
        ard: bool = True,
        lr: float = 0.01,
        training_iter: int = 1000,
        low_fidelity_training_iter: int | None = None,
        correction_initial_noise: float | None = 0.1,
        low_fidelity_initial_noise: float | None = 0.1,
        standardize_y: bool = True,
        include_low_fidelity_uncertainty: bool = True,
        device: str = "cpu",
        dtype: str | torch.dtype = "float64",
        verbose: bool = False,
        log_every: int = 100,
        include_observation_noise: bool = True,
        random_state: int | None = None,
    ):
        self.fit_intercept = fit_intercept
        self.correction_kernel = correction_kernel
        self.low_fidelity_kernel = low_fidelity_kernel
        self.ard = ard
        self.lr = lr
        self.training_iter = training_iter
        self.low_fidelity_training_iter = low_fidelity_training_iter
        self.correction_initial_noise = correction_initial_noise
        self.low_fidelity_initial_noise = low_fidelity_initial_noise
        self.standardize_y = standardize_y
        self.include_low_fidelity_uncertainty = include_low_fidelity_uncertainty
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.log_every = log_every
        self.include_observation_noise = include_observation_noise
        self.random_state = random_state

    def fit(self, X, y, *, low_fidelity=None, X_low=None, y_low=None):
        """Fit the multi-fidelity GPR model."""
        _seed_torch(self.random_state)
        X_checked = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        y_checked = column_or_1d(
            check_array(
                y,
                ensure_2d=False,
                dtype="numeric",
                ensure_all_finite=True,
                input_name="y",
            ),
            warn=True,
        )
        check_consistent_length(X_checked, y_checked)
        self.n_features_in_ = X_checked.shape[1]

        self.result_ = fit_delta_multifidelity_gpr(
            X_checked,
            y_checked,
            low_fidelity_high=low_fidelity,
            X_low=X_low,
            y_low=y_low,
            fit_intercept=self.fit_intercept,
            correction_kernel=self.correction_kernel,
            low_fidelity_kernel=self.low_fidelity_kernel,
            ard=self.ard,
            lr=self.lr,
            training_iter=self.training_iter,
            low_fidelity_training_iter=self.low_fidelity_training_iter,
            correction_initial_noise=self.correction_initial_noise,
            low_fidelity_initial_noise=self.low_fidelity_initial_noise,
            standardize_y=self.standardize_y,
            include_low_fidelity_uncertainty=self.include_low_fidelity_uncertainty,
            device=self.device,
            dtype=_resolve_torch_dtype(self.dtype),
            verbose=self.verbose,
            log_every=self.log_every,
        )
        self.correction_model_ = self.result_.correction_model
        self.low_fidelity_model_ = self.result_.low_fidelity_model
        self.rho_ = self.result_.rho
        self.intercept_ = self.result_.intercept
        self.correction_target_ = self.result_.correction_target.copy()
        self.low_fidelity_at_high_ = self.result_.low_fidelity_at_high.copy()
        return self

    def predict(
        self,
        X,
        *,
        low_fidelity=None,
        return_std: bool = False,
        include_observation_noise: bool | None = None,
        include_low_fidelity_uncertainty: bool | None = None,
    ):
        """Predict high-fidelity values."""
        prediction = self.predict_distribution(
            X,
            low_fidelity=low_fidelity,
            return_std=return_std,
            include_observation_noise=include_observation_noise,
            include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        )
        if return_std:
            return prediction.mean, prediction.std
        return prediction.mean

    def predict_distribution(
        self,
        X,
        *,
        low_fidelity=None,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool | None = None,
        include_low_fidelity_uncertainty: bool | None = None,
    ) -> MultiFidelityGPRPrediction:
        """Return high-fidelity predictive mean, uncertainty, and components."""
        check_is_fitted(self, "result_")
        X_checked = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_} features"
            )
        if include_observation_noise is None:
            include_observation_noise = self.include_observation_noise

        return self.result_.predict(
            X_checked,
            low_fidelity=low_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
            include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        )

    def score(self, X, y, sample_weight=None, *, low_fidelity=None) -> float:
        """Return R2 for high-fidelity predictions."""
        return r2_score(
            y,
            self.predict(X, low_fidelity=low_fidelity),
            sample_weight=sample_weight,
        )


def _predict_delta_multifidelity_gpr(
    result: DeltaMultiFidelityGPRResult,
    X,
    *,
    low_fidelity,
    return_std: bool,
    confidence_level: float | None,
    include_observation_noise: bool,
    include_low_fidelity_uncertainty: bool | None,
) -> MultiFidelityGPRPrediction:
    _validate_confidence_level(confidence_level)
    X_array = _to_2d_numpy(X, "X")
    low_mean, low_std = _resolve_prediction_low_fidelity(
        result,
        X_array,
        low_fidelity=low_fidelity,
        return_std=return_std or confidence_level is not None,
    )
    correction_prediction = result.correction_model.predict(
        X_array,
        return_std=return_std or confidence_level is not None,
        include_observation_noise=include_observation_noise,
    )
    mean = result.rho * low_mean + result.intercept + correction_prediction.mean

    correction_std = correction_prediction.std
    std = None
    if return_std or confidence_level is not None:
        if correction_std is None:
            raise ValueError("Internal correction prediction did not return standard deviations")
        include_low = (
            result.include_low_fidelity_uncertainty
            if include_low_fidelity_uncertainty is None
            else bool(include_low_fidelity_uncertainty)
        )
        variance = correction_std**2
        if include_low and low_std is not None:
            variance = variance + (result.rho * low_std) ** 2
        std = np.sqrt(np.maximum(variance, 0.0))

    lower = None
    upper = None
    if confidence_level is not None:
        if std is None:
            raise ValueError("confidence_level requires return_std=True")
        z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        lower = mean - z_value * std
        upper = mean + z_value * std

    return MultiFidelityGPRPrediction(
        mean=mean,
        std=std,
        lower=lower,
        upper=upper,
        low_fidelity_mean=low_mean,
        low_fidelity_std=low_std,
        correction_mean=correction_prediction.mean,
        correction_std=correction_std,
        rho=result.rho,
        intercept=result.intercept,
    )


def _resolve_prediction_low_fidelity(
    result: DeltaMultiFidelityGPRResult,
    X: np.ndarray,
    *,
    low_fidelity,
    return_std: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if low_fidelity is not None:
        values = _to_1d_numpy(low_fidelity, "low_fidelity")
        check_consistent_length(X, values)
        return values, None
    if result.low_fidelity_model is None:
        raise ValueError(
            "low_fidelity is required for prediction because no low-fidelity "
            "surrogate was fitted"
        )
    prediction = result.low_fidelity_model.predict(
        X,
        return_std=return_std,
        include_observation_noise=False,
    )
    return prediction.mean, prediction.std


def _fit_linear_fidelity_map(
    low_fidelity: np.ndarray,
    high_fidelity: np.ndarray,
    *,
    fit_intercept: bool,
) -> tuple[float, float]:
    if np.std(low_fidelity) <= 0:
        raise ValueError("low_fidelity values must have nonzero variance")
    if fit_intercept:
        design = np.column_stack([low_fidelity, np.ones_like(low_fidelity)])
        rho, intercept = np.linalg.lstsq(design, high_fidelity, rcond=None)[0]
    else:
        rho = float(np.dot(low_fidelity, high_fidelity) / np.dot(low_fidelity, low_fidelity))
        intercept = 0.0
    return float(rho), float(intercept)


def _to_2d_numpy(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_1d_numpy(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_feature_width_match(X_high: np.ndarray, X_low, high_name: str, low_name: str) -> None:
    if X_low is None:
        return
    X_low_array = _to_2d_numpy(X_low, low_name)
    if X_high.shape[1] != X_low_array.shape[1]:
        raise ValueError(
            f"{high_name} has {X_high.shape[1]} features, but {low_name} has "
            f"{X_low_array.shape[1]} features"
        )


def _validate_confidence_level(confidence_level: float | None) -> None:
    if confidence_level is None:
        return
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower()
    if normalized in {"float64", "double", "torch.float64"}:
        return torch.float64
    if normalized in {"float32", "single", "torch.float32"}:
        return torch.float32
    raise ValueError("dtype must be 'float64', 'float32', or a torch dtype")


def _seed_torch(random_state: int | None) -> None:
    if random_state is not None:
        torch.manual_seed(int(random_state))
