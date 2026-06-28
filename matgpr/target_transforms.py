from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gpytorch_gpr import GPyTorchPrediction

__all__ = [
    "IdentityTargetTransform",
    "LogTargetTransform",
    "PhysicsResidualTransform",
    "StandardizedTargetTransform",
    "make_target_transform",
]


@dataclass
class IdentityTargetTransform:
    """No-op target transform with the same interface as other transforms."""

    def fit(self, y, **kwargs):
        """Validate target values and return the transform."""
        _to_1d_finite(y, "y")
        return self

    def transform(self, y, **kwargs) -> np.ndarray:
        """Return target values unchanged."""
        return _to_1d_finite(y, "y")

    def fit_transform(self, y, **kwargs) -> np.ndarray:
        """Fit the transform and return transformed target values."""
        self.fit(y, **kwargs)
        return self.transform(y, **kwargs)

    def inverse_transform(self, y_transformed, **kwargs) -> np.ndarray:
        """Return transformed target values unchanged."""
        return _to_1d_finite(y_transformed, "y_transformed")

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return predictive standard deviations unchanged."""
        _to_1d_finite(mean_transformed, "mean_transformed")
        return _to_nonnegative_std(std_transformed)

    def inverse_prediction(self, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
        """Return a prediction object in the original target scale."""
        return _inverse_prediction_with_arrays(self, prediction, **kwargs)


@dataclass
class StandardizedTargetTransform:
    """Standardize target values using training-set mean and standard deviation."""

    mean_: float | None = field(default=None, init=False)
    scale_: float | None = field(default=None, init=False)

    def fit(self, y, **kwargs):
        """Estimate the mean and standard deviation from training targets."""
        values = _to_1d_finite(y, "y")
        scale = float(np.std(values, ddof=0))
        if scale <= 0:
            raise ValueError("Cannot standardize a constant target")
        self.mean_ = float(np.mean(values))
        self.scale_ = scale
        return self

    def transform(self, y, **kwargs) -> np.ndarray:
        """Return standardized target values."""
        self._require_fitted()
        values = _to_1d_finite(y, "y")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, y, **kwargs) -> np.ndarray:
        """Fit the transform and return standardized target values."""
        self.fit(y, **kwargs)
        return self.transform(y, **kwargs)

    def inverse_transform(self, y_transformed, **kwargs) -> np.ndarray:
        """Return standardized values in the original target scale."""
        self._require_fitted()
        values = _to_1d_finite(y_transformed, "y_transformed")
        return values * self.scale_ + self.mean_

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return predictive standard deviations in the original target scale."""
        self._require_fitted()
        _to_1d_finite(mean_transformed, "mean_transformed")
        return _to_nonnegative_std(std_transformed) * self.scale_

    def inverse_prediction(self, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
        """Return a prediction object in the original target scale."""
        return _inverse_prediction_with_arrays(self, prediction, **kwargs)

    def _require_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardizedTargetTransform must be fitted before use")


@dataclass
class LogTargetTransform:
    """Log-transform positive targets and invert Gaussian predictions.

    The transform is ``z = log(y + offset)``. Predictive means and standard
    deviations are inverted with log-normal moments, which is more appropriate
    than simply exponentiating the transformed-space mean.
    """

    offset: float = 0.0

    def fit(self, y, **kwargs):
        """Validate that all shifted target values are positive."""
        self._validate_shifted_target(y, "y")
        return self

    def transform(self, y, **kwargs) -> np.ndarray:
        """Return log-transformed target values."""
        values = self._validate_shifted_target(y, "y")
        return np.log(values + self.offset)

    def fit_transform(self, y, **kwargs) -> np.ndarray:
        """Fit the transform and return log-transformed target values."""
        self.fit(y, **kwargs)
        return self.transform(y, **kwargs)

    def inverse_transform(self, y_transformed, **kwargs) -> np.ndarray:
        """Return log-scale values in the original target scale."""
        values = _to_1d_finite(y_transformed, "y_transformed")
        return np.exp(values) - self.offset

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return original-scale standard deviations from log-normal moments."""
        mean = _to_1d_finite(mean_transformed, "mean_transformed")
        std = _to_nonnegative_std(std_transformed)
        _validate_same_length(mean, std, "mean_transformed", "std_transformed")
        variance = std**2
        return np.sqrt(np.expm1(variance) * np.exp(2.0 * mean + variance))

    def inverse_prediction(self, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
        """Return a log-space prediction object in the original target scale."""
        mean = _to_1d_finite(prediction.mean, "prediction.mean")
        std = None
        if prediction.std is not None:
            std = self.inverse_std(mean, prediction.std)

        lower = None
        if prediction.lower is not None:
            lower = self.inverse_transform(prediction.lower)

        upper = None
        if prediction.upper is not None:
            upper = self.inverse_transform(prediction.upper)

        if prediction.std is not None:
            original_mean = np.exp(mean + 0.5 * _to_nonnegative_std(prediction.std) ** 2) - self.offset
        else:
            original_mean = self.inverse_transform(mean)

        return GPyTorchPrediction(mean=original_mean, std=std, lower=lower, upper=upper)

    def _validate_shifted_target(self, y, name: str) -> np.ndarray:
        values = _to_1d_finite(y, name)
        if np.any(values + self.offset <= 0):
            raise ValueError(f"{name} + offset must contain only positive values")
        return values


@dataclass
class PhysicsResidualTransform:
    """Model residuals relative to a physics baseline.

    This transform supports workflows where a simple physics model provides a
    baseline prediction and GPR learns the residual:
    ``residual = observed_target - physics_baseline``.
    """

    baseline_name: str = "physics_baseline"

    def fit(self, y, *, baseline):
        """Validate target and baseline arrays."""
        values = _to_1d_finite(y, "y")
        baseline = _to_1d_finite(baseline, self.baseline_name)
        _validate_same_length(values, baseline, "y", self.baseline_name)
        return self

    def transform(self, y, *, baseline) -> np.ndarray:
        """Return residual target values."""
        values = _to_1d_finite(y, "y")
        baseline = _to_1d_finite(baseline, self.baseline_name)
        _validate_same_length(values, baseline, "y", self.baseline_name)
        return values - baseline

    def fit_transform(self, y, *, baseline) -> np.ndarray:
        """Fit the transform and return residual target values."""
        self.fit(y, baseline=baseline)
        return self.transform(y, baseline=baseline)

    def inverse_transform(self, y_transformed, *, baseline) -> np.ndarray:
        """Return residual predictions in the original target scale."""
        residual = _to_1d_finite(y_transformed, "y_transformed")
        baseline = _to_1d_finite(baseline, self.baseline_name)
        _validate_same_length(residual, baseline, "y_transformed", self.baseline_name)
        return residual + baseline

    def inverse_std(self, mean_transformed, std_transformed, **kwargs) -> np.ndarray:
        """Return residual predictive standard deviations unchanged."""
        _to_1d_finite(mean_transformed, "mean_transformed")
        return _to_nonnegative_std(std_transformed)

    def inverse_prediction(self, prediction: GPyTorchPrediction, *, baseline) -> GPyTorchPrediction:
        """Add a physics baseline back to a residual prediction object."""
        return _inverse_prediction_with_arrays(self, prediction, baseline=baseline)


def make_target_transform(name: str, **kwargs):
    """Create a target transform by name.

    Parameters
    ----------
    name
        One of ``"identity"``, ``"standard"``, ``"standardized"``, ``"log"``,
        ``"residual"``, or ``"physics_residual"``.
    """
    normalized = name.lower().replace("-", "_")
    if normalized in {"identity", "none", "passthrough"}:
        return IdentityTargetTransform(**kwargs)
    if normalized in {"standard", "standardized", "zscore", "z_score"}:
        return StandardizedTargetTransform(**kwargs)
    if normalized in {"log", "logarithmic"}:
        return LogTargetTransform(**kwargs)
    if normalized in {"residual", "physics_residual"}:
        return PhysicsResidualTransform(**kwargs)
    raise ValueError("name must be one of: identity, standard, log, residual")


def _inverse_prediction_with_arrays(transform, prediction: GPyTorchPrediction, **kwargs) -> GPyTorchPrediction:
    mean = transform.inverse_transform(prediction.mean, **kwargs)

    std = None
    if prediction.std is not None:
        std = transform.inverse_std(prediction.mean, prediction.std, **kwargs)

    lower = None
    if prediction.lower is not None:
        lower = transform.inverse_transform(prediction.lower, **kwargs)

    upper = None
    if prediction.upper is not None:
        upper = transform.inverse_transform(prediction.upper, **kwargs)

    return GPyTorchPrediction(mean=mean, std=std, lower=lower, upper=upper)


def _to_1d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_nonnegative_std(values) -> np.ndarray:
    array = _to_1d_finite(values, "std_transformed")
    if np.any(array < 0):
        raise ValueError("std_transformed must contain only non-negative values")
    return array


def _validate_same_length(first: np.ndarray, second: np.ndarray, first_name: str, second_name: str) -> None:
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same length")
