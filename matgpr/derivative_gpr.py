from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "DerivativeConstrainedGPRPrediction",
    "DerivativeConstrainedGPRResult",
    "DerivativeObservationSet",
    "MonotonicDerivativeConstraint",
    "combine_derivative_observations",
    "fit_derivative_constrained_gpr",
]


@dataclass(frozen=True)
class DerivativeObservationSet:
    """Observed or physics-implied derivatives for derivative-constrained GPR.

    Each row represents one derivative observation:

    ``derivative_values[i] = df(X[i]) / dX[feature_indices[i]]``.

    Derivatives must be expressed with respect to the same feature scale used
    in the GPR feature matrix. If the feature matrix is standardized, provide
    derivatives with respect to the standardized feature values.
    """

    X: np.ndarray
    feature_indices: int | Sequence[int] | np.ndarray
    derivative_values: Sequence[float] | np.ndarray
    noise_std: float | Sequence[float] | np.ndarray = 0.1
    labels: Sequence[str] | np.ndarray | None = None

    def __post_init__(self) -> None:
        x_values = _to_2d_finite(self.X, "X")
        derivative_values = _to_1d_finite(self.derivative_values, "derivative_values")
        _validate_same_length(x_values, derivative_values, "X", "derivative_values")

        feature_indices = _feature_index_array(
            self.feature_indices,
            n_observations=derivative_values.shape[0],
            n_features=x_values.shape[1],
        )
        noise = _noise_std_array(self.noise_std, derivative_values.shape[0])
        labels = _label_array(self.labels, derivative_values.shape[0], default="derivative")

        object.__setattr__(self, "X", x_values)
        object.__setattr__(self, "feature_indices", feature_indices)
        object.__setattr__(self, "derivative_values", derivative_values)
        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "labels", labels)

    @property
    def n_observations(self) -> int:
        """Number of derivative observations."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.X.shape[1])

    @property
    def alpha(self) -> np.ndarray:
        """Derivative observation variances."""
        return self.noise_std**2


@dataclass(frozen=True)
class MonotonicDerivativeConstraint:
    """Create derivative observations for a local monotonic trend.

    For an increasing trend, generated derivative observations have positive
    target value ``minimum_slope``. For a decreasing trend, generated
    derivative observations have target value ``-minimum_slope``.

    These are soft equality observations of the derivative. They encourage the
    posterior to follow the trend near the anchor points, but they are not a
    formal guarantee that the function is globally monotonic everywhere.
    """

    feature: int | str
    direction: str
    minimum_slope: float = 0.0
    noise_std: float | Sequence[float] | np.ndarray = 0.1
    feature_min: float | None = None
    feature_max: float | None = None
    label: str | None = None

    def generate(self, X_reference) -> DerivativeObservationSet:
        """Generate derivative observations from reference feature rows."""
        x_values, columns = _as_numeric_matrix(X_reference, "X_reference")
        feature_index = _resolve_feature_index(self.feature, columns, x_values.shape[1])
        response_sign = _response_sign(self.direction)
        minimum_slope = _nonnegative_scalar(self.minimum_slope, "minimum_slope")
        derivative_value = response_sign * minimum_slope

        keep_mask = _feature_bounds_mask(
            x_values[:, feature_index],
            feature_min=self.feature_min,
            feature_max=self.feature_max,
        )
        if not np.any(keep_mask):
            raise ValueError("No derivative observations remain after applying feature bounds")

        x_virtual = x_values[keep_mask]
        derivative_values = np.full(x_virtual.shape[0], derivative_value, dtype=float)
        feature_indices = np.full(x_virtual.shape[0], feature_index, dtype=int)
        noise = _noise_std_array(self.noise_std, x_values.shape[0])[keep_mask]
        label = self.label or f"derivative_{_normalized_direction(self.direction)}"
        labels = np.full(x_virtual.shape[0], label, dtype=object)

        return DerivativeObservationSet(
            X=x_virtual,
            feature_indices=feature_indices,
            derivative_values=derivative_values,
            noise_std=noise,
            labels=labels,
        )


@dataclass(frozen=True)
class DerivativeConstrainedGPRPrediction:
    """Predictions from a derivative-constrained exact GPR model."""

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


@dataclass(frozen=True)
class DerivativeConstrainedGPRResult:
    """Fitted exact GPR model with derivative observations."""

    X_train: np.ndarray
    y_train: np.ndarray
    derivative_observations: DerivativeObservationSet | None
    length_scale: np.ndarray
    signal_variance: float
    value_noise_std: np.ndarray
    standardize_y: bool
    target_mean: float
    target_scale: float
    lower_cholesky: np.ndarray
    alpha: np.ndarray
    log_marginal_likelihood: float
    optimizer_success: bool | None = None
    optimizer_message: str | None = None

    def predict(
        self,
        X,
        *,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool = False,
    ) -> DerivativeConstrainedGPRPrediction:
        """Predict function values at new feature rows."""
        _validate_confidence_level(confidence_level)
        x_test, _ = _as_numeric_matrix(X, "X")
        if x_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X has {x_test.shape[1]} features, but the model was fitted with "
                f"{self.X_train.shape[1]} features"
            )

        cross_covariance = _cross_covariance_to_training(
            x_test,
            self.X_train,
            self.derivative_observations,
            length_scale=self.length_scale,
            signal_variance=self.signal_variance,
        )
        mean_model = cross_covariance @ self.alpha
        mean = mean_model * self.target_scale + self.target_mean

        std = None
        if return_std or confidence_level is not None:
            solved = _solve_lower_triangular(self.lower_cholesky, cross_covariance.T)
            variance_model = self.signal_variance - np.sum(solved * solved, axis=0)
            if include_observation_noise:
                variance_model = variance_model + float(np.mean((self.value_noise_std / self.target_scale) ** 2))
            variance_model = np.maximum(variance_model, 0.0)
            std = np.sqrt(variance_model) * self.target_scale

        lower = None
        upper = None
        if confidence_level is not None:
            if std is None:
                raise ValueError("confidence_level requires return_std=True")
            z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
            lower = mean - z_value * std
            upper = mean + z_value * std

        return DerivativeConstrainedGPRPrediction(mean=mean, std=std, lower=lower, upper=upper)


def combine_derivative_observations(
    *derivative_observations: DerivativeObservationSet,
) -> DerivativeObservationSet:
    """Combine compatible derivative-observation sets."""
    sets = _flatten_derivative_observation_sets(derivative_observations)
    if not sets:
        raise ValueError("At least one DerivativeObservationSet is required")

    n_features = sets[0].n_features
    for observation_set in sets:
        if observation_set.n_features != n_features:
            raise ValueError("All derivative observation sets must have the same number of features")

    return DerivativeObservationSet(
        X=np.vstack([observation_set.X for observation_set in sets]),
        feature_indices=np.concatenate([observation_set.feature_indices for observation_set in sets]),
        derivative_values=np.concatenate([observation_set.derivative_values for observation_set in sets]),
        noise_std=np.concatenate([observation_set.noise_std for observation_set in sets]),
        labels=np.concatenate([observation_set.labels for observation_set in sets]),
    )


def fit_derivative_constrained_gpr(
    X_train,
    y_train,
    derivative_observations: DerivativeObservationSet | Sequence[DerivativeObservationSet] | None = None,
    *,
    length_scale: float | Sequence[float] | np.ndarray | None = None,
    signal_variance: float | None = 1.0,
    value_noise_std: float | Sequence[float] | np.ndarray = 1e-6,
    standardize_y: bool = True,
    optimize_hyperparameters: bool = False,
    length_scale_bounds: tuple[float, float] = (1e-4, 1e4),
    signal_variance_bounds: tuple[float, float] = (1e-8, 1e8),
    maxiter: int = 100,
    jitter: float = 1e-8,
) -> DerivativeConstrainedGPRResult:
    """Fit exact RBF GPR with function and derivative observations.

    The model uses the joint Gaussian-process covariance over function values
    and partial derivatives. This is useful when physics provides slope
    information, for example positive diffusivity change with temperature,
    decreasing viscosity with temperature, or a known zero-slope saturation
    region.
    """
    x_train, _ = _as_numeric_matrix(X_train, "X_train")
    y_values = _to_1d_finite(y_train, "y_train")
    _validate_same_length(x_train, y_values, "X_train", "y_train")

    derivative_set = _prepare_derivative_observations(derivative_observations, x_train.shape[1])
    value_noise = _noise_std_array(value_noise_std, y_values.shape[0])

    if standardize_y:
        target_mean = float(np.mean(y_values))
        target_scale = float(np.std(y_values, ddof=0))
        if target_scale <= 0:
            raise ValueError("Cannot standardize a constant target")
    else:
        target_mean = 0.0
        target_scale = 1.0

    y_model = (y_values - target_mean) / target_scale
    derivative_values_model = None
    derivative_noise_model = None
    if derivative_set is not None:
        derivative_values_model = derivative_set.derivative_values / target_scale
        derivative_noise_model = derivative_set.noise_std / target_scale
    value_noise_model = value_noise / target_scale

    length_scale_array = _initial_length_scale(length_scale, x_train)
    signal_variance_value = _initial_signal_variance(signal_variance, y_model)

    optimizer_success = None
    optimizer_message = None
    if optimize_hyperparameters:
        length_scale_array, signal_variance_value, optimizer_success, optimizer_message = _optimize_hyperparameters(
            x_train,
            y_model,
            derivative_set,
            derivative_values_model,
            derivative_noise_model,
            value_noise_model,
            initial_length_scale=length_scale_array,
            initial_signal_variance=signal_variance_value,
            length_scale_bounds=length_scale_bounds,
            signal_variance_bounds=signal_variance_bounds,
            maxiter=maxiter,
            jitter=jitter,
        )

    training_target = _combined_training_target(y_model, derivative_values_model)
    training_covariance = _training_covariance(
        x_train,
        derivative_set,
        length_scale=length_scale_array,
        signal_variance=signal_variance_value,
        value_noise_std=value_noise_model,
        derivative_noise_std=derivative_noise_model,
    )
    lower_cholesky = _stable_cholesky(training_covariance, jitter=jitter)
    alpha = _cho_solve(lower_cholesky, training_target)
    log_marginal_likelihood = _log_marginal_likelihood(
        training_target,
        lower_cholesky,
        alpha,
    )

    return DerivativeConstrainedGPRResult(
        X_train=x_train,
        y_train=y_values,
        derivative_observations=derivative_set,
        length_scale=length_scale_array,
        signal_variance=signal_variance_value,
        value_noise_std=value_noise,
        standardize_y=standardize_y,
        target_mean=target_mean,
        target_scale=target_scale,
        lower_cholesky=lower_cholesky,
        alpha=alpha,
        log_marginal_likelihood=log_marginal_likelihood,
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
    )


def _training_covariance(
    x_train: np.ndarray,
    derivative_set: DerivativeObservationSet | None,
    *,
    length_scale: np.ndarray,
    signal_variance: float,
    value_noise_std: np.ndarray,
    derivative_noise_std: np.ndarray | None,
) -> np.ndarray:
    k_ff = _rbf_kernel(
        x_train,
        x_train,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    k_ff = k_ff + np.diag(value_noise_std**2)

    if derivative_set is None:
        return k_ff

    if derivative_noise_std is None:
        raise ValueError("derivative_noise_std is required when derivative observations are provided")

    k_fd = _function_derivative_covariance(
        x_train,
        derivative_set.X,
        derivative_set.feature_indices,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    k_dd = _derivative_derivative_covariance(
        derivative_set.X,
        derivative_set.feature_indices,
        derivative_set.X,
        derivative_set.feature_indices,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    k_dd = k_dd + np.diag(derivative_noise_std**2)

    return np.block(
        [
            [k_ff, k_fd],
            [k_fd.T, k_dd],
        ]
    )


def _cross_covariance_to_training(
    x_test: np.ndarray,
    x_train: np.ndarray,
    derivative_set: DerivativeObservationSet | None,
    *,
    length_scale: np.ndarray,
    signal_variance: float,
) -> np.ndarray:
    k_test_train = _rbf_kernel(
        x_test,
        x_train,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    if derivative_set is None:
        return k_test_train

    k_test_derivative = _function_derivative_covariance(
        x_test,
        derivative_set.X,
        derivative_set.feature_indices,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    return np.hstack([k_test_train, k_test_derivative])


def _rbf_kernel(
    x_left: np.ndarray,
    x_right: np.ndarray,
    *,
    length_scale: np.ndarray,
    signal_variance: float,
) -> np.ndarray:
    scaled_difference = (x_left[:, None, :] - x_right[None, :, :]) / length_scale
    squared_distance = np.sum(scaled_difference * scaled_difference, axis=2)
    return signal_variance * np.exp(-0.5 * squared_distance)


def _function_derivative_covariance(
    x_function: np.ndarray,
    x_derivative: np.ndarray,
    derivative_feature_indices: np.ndarray,
    *,
    length_scale: np.ndarray,
    signal_variance: float,
) -> np.ndarray:
    base = _rbf_kernel(
        x_function,
        x_derivative,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    difference = x_function[:, None, :] - x_derivative[None, :, :]
    row_index = np.arange(x_function.shape[0])[:, None]
    column_index = np.arange(x_derivative.shape[0])[None, :]
    feature_index = derivative_feature_indices[None, :]
    feature_difference = difference[row_index, column_index, feature_index]
    feature_length_scale = length_scale[derivative_feature_indices][None, :]
    return base * feature_difference / (feature_length_scale**2)


def _derivative_derivative_covariance(
    x_left: np.ndarray,
    left_feature_indices: np.ndarray,
    x_right: np.ndarray,
    right_feature_indices: np.ndarray,
    *,
    length_scale: np.ndarray,
    signal_variance: float,
) -> np.ndarray:
    base = _rbf_kernel(
        x_left,
        x_right,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    difference = x_left[:, None, :] - x_right[None, :, :]

    row_index = np.arange(x_left.shape[0])[:, None]
    column_index = np.arange(x_right.shape[0])[None, :]
    left_features = left_feature_indices[:, None]
    right_features = right_feature_indices[None, :]

    left_difference = difference[row_index, column_index, left_features]
    right_difference = difference[row_index, column_index, right_features]
    left_length_scale = length_scale[left_feature_indices][:, None]
    right_length_scale = length_scale[right_feature_indices][None, :]

    same_feature = left_feature_indices[:, None] == right_feature_indices[None, :]
    same_feature_term = np.where(same_feature, 1.0 / (left_length_scale**2), 0.0)
    curvature_term = left_difference * right_difference / (
        left_length_scale**2 * right_length_scale**2
    )
    return base * (same_feature_term - curvature_term)


def _optimize_hyperparameters(
    x_train: np.ndarray,
    y_model: np.ndarray,
    derivative_set: DerivativeObservationSet | None,
    derivative_values_model: np.ndarray | None,
    derivative_noise_model: np.ndarray | None,
    value_noise_model: np.ndarray,
    *,
    initial_length_scale: np.ndarray,
    initial_signal_variance: float,
    length_scale_bounds: tuple[float, float],
    signal_variance_bounds: tuple[float, float],
    maxiter: int,
    jitter: float,
) -> tuple[np.ndarray, float, bool, str]:
    _validate_positive_bounds(length_scale_bounds, "length_scale_bounds")
    _validate_positive_bounds(signal_variance_bounds, "signal_variance_bounds")

    initial_parameters = np.log(np.concatenate([initial_length_scale, [initial_signal_variance]]))
    bounds = [tuple(np.log(length_scale_bounds))] * initial_length_scale.shape[0]
    bounds.append(tuple(np.log(signal_variance_bounds)))
    training_target = _combined_training_target(y_model, derivative_values_model)

    def objective(log_parameters: np.ndarray) -> float:
        length_scale = np.exp(log_parameters[:-1])
        signal_variance = float(np.exp(log_parameters[-1]))
        try:
            covariance = _training_covariance(
                x_train,
                derivative_set,
                length_scale=length_scale,
                signal_variance=signal_variance,
                value_noise_std=value_noise_model,
                derivative_noise_std=derivative_noise_model,
            )
            lower = _stable_cholesky(covariance, jitter=jitter)
            alpha = _cho_solve(lower, training_target)
            return -_log_marginal_likelihood(training_target, lower, alpha)
        except np.linalg.LinAlgError:
            return np.inf

    result = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter)},
    )
    best_parameters = result.x if np.all(np.isfinite(result.x)) else initial_parameters
    return (
        np.exp(best_parameters[:-1]),
        float(np.exp(best_parameters[-1])),
        bool(result.success),
        str(result.message),
    )


def _as_numeric_matrix(values, name: str) -> tuple[np.ndarray, list[str] | None]:
    columns = None
    if isinstance(values, pd.DataFrame):
        columns = [str(column) for column in values.columns]
        array = values.to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    return _to_2d_finite(array, name), columns


def _to_2d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_1d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_same_length(first: np.ndarray, second: np.ndarray, first_name: str, second_name: str) -> None:
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same number of rows")


def _feature_index_array(
    feature_indices: int | Sequence[int] | np.ndarray,
    *,
    n_observations: int,
    n_features: int,
) -> np.ndarray:
    if isinstance(feature_indices, (int, np.integer)) and not isinstance(feature_indices, bool):
        values = np.full(n_observations, int(feature_indices), dtype=int)
    else:
        values = np.asarray(feature_indices, dtype=int).ravel()
        if values.shape[0] != n_observations:
            raise ValueError("feature_indices must be a scalar or have one value per observation")

    if np.any(values < 0) or np.any(values >= n_features):
        raise ValueError("feature_indices contains an out-of-bounds feature index")
    return values


def _resolve_feature_index(feature: int | str, columns: list[str] | None, n_features: int) -> int:
    if isinstance(feature, str):
        if columns is None:
            raise ValueError("String feature names require X_reference to be a dataframe")
        if feature not in columns:
            raise ValueError(f"Feature '{feature}' was not found in dataframe columns")
        return columns.index(feature)

    if isinstance(feature, bool) or not isinstance(feature, int):
        raise TypeError("feature must be an integer column index or dataframe column name")
    if feature < 0 or feature >= n_features:
        raise ValueError(f"feature index {feature} is out of bounds for {n_features} features")
    return int(feature)


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _noise_std_array(values, n_observations: int) -> np.ndarray:
    if np.isscalar(values):
        noise = np.full(n_observations, _nonnegative_scalar(values, "noise_std"), dtype=float)
    else:
        noise = _to_1d_finite(values, "noise_std")
        if noise.shape[0] != n_observations:
            raise ValueError("noise_std must be a scalar or have one value per observation")
        if np.any(noise < 0):
            raise ValueError("noise_std must be non-negative")
    return noise


def _label_array(labels, n_observations: int, *, default: str) -> np.ndarray:
    if labels is None:
        return np.full(n_observations, default, dtype=object)
    label_array = np.asarray(labels, dtype=object).ravel()
    if label_array.shape[0] != n_observations:
        raise ValueError("labels must have one value per observation")
    return label_array


def _normalized_direction(direction: str) -> str:
    normalized = str(direction).lower().replace("-", "_")
    aliases = {
        "increase": "increasing",
        "increasing": "increasing",
        "nondecreasing": "increasing",
        "non_decreasing": "increasing",
        "decrease": "decreasing",
        "decreasing": "decreasing",
        "nonincreasing": "decreasing",
        "non_increasing": "decreasing",
    }
    if normalized not in aliases:
        raise ValueError("direction must be increasing or decreasing")
    return aliases[normalized]


def _response_sign(direction: str) -> float:
    return 1.0 if _normalized_direction(direction) == "increasing" else -1.0


def _feature_bounds_mask(
    feature_values: np.ndarray,
    *,
    feature_min: float | None,
    feature_max: float | None,
) -> np.ndarray:
    if feature_min is not None and feature_max is not None:
        lower = _finite_scalar(feature_min, "feature_min")
        upper = _finite_scalar(feature_max, "feature_max")
        if upper < lower:
            raise ValueError("feature_max must be greater than or equal to feature_min")
    elif feature_min is not None:
        lower = _finite_scalar(feature_min, "feature_min")
        upper = np.inf
    elif feature_max is not None:
        lower = -np.inf
        upper = _finite_scalar(feature_max, "feature_max")
    else:
        lower = -np.inf
        upper = np.inf
    return (feature_values >= lower) & (feature_values <= upper)


def _flatten_derivative_observation_sets(
    derivative_observations,
) -> list[DerivativeObservationSet]:
    if len(derivative_observations) == 1 and isinstance(derivative_observations[0], (list, tuple)):
        derivative_observations = tuple(derivative_observations[0])
    sets = list(derivative_observations)
    if not all(isinstance(observation_set, DerivativeObservationSet) for observation_set in sets):
        raise TypeError("derivative_observations must contain only DerivativeObservationSet objects")
    return sets


def _prepare_derivative_observations(
    derivative_observations: DerivativeObservationSet | Sequence[DerivativeObservationSet] | None,
    n_features: int,
) -> DerivativeObservationSet | None:
    if derivative_observations is None:
        return None
    if isinstance(derivative_observations, DerivativeObservationSet):
        sets = [derivative_observations]
    else:
        sets = list(derivative_observations)
    if not sets:
        return None
    combined = combine_derivative_observations(*sets)
    if combined.n_features != n_features:
        raise ValueError("Derivative observations must have the same number of features as X_train")
    return combined


def _initial_length_scale(
    length_scale: float | Sequence[float] | np.ndarray | None,
    x_train: np.ndarray,
) -> np.ndarray:
    if length_scale is None:
        scale = np.std(x_train, axis=0, ddof=0)
        scale = np.where(scale > 0, scale, 1.0)
        return scale.astype(float)
    if np.isscalar(length_scale):
        value = _positive_scalar(length_scale, "length_scale")
        return np.full(x_train.shape[1], value, dtype=float)
    values = _to_1d_finite(length_scale, "length_scale")
    if values.shape[0] != x_train.shape[1]:
        raise ValueError("length_scale must be a scalar or have one value per feature")
    if np.any(values <= 0):
        raise ValueError("length_scale values must be positive")
    return values


def _initial_signal_variance(signal_variance: float | None, y_model: np.ndarray) -> float:
    if signal_variance is not None:
        return _positive_scalar(signal_variance, "signal_variance")
    variance = float(np.var(y_model, ddof=0))
    return variance if variance > 0 else 1.0


def _combined_training_target(
    y_model: np.ndarray,
    derivative_values_model: np.ndarray | None,
) -> np.ndarray:
    if derivative_values_model is None:
        return y_model
    return np.concatenate([y_model, derivative_values_model])


def _stable_cholesky(covariance: np.ndarray, *, jitter: float) -> np.ndarray:
    jitter_value = _nonnegative_scalar(jitter, "jitter")
    identity = np.eye(covariance.shape[0])
    for scale in (0.0, 1.0, 10.0, 100.0, 1000.0):
        try:
            return np.linalg.cholesky(covariance + identity * jitter_value * scale)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.cholesky(covariance + identity * jitter_value * 10000.0)


def _cho_solve(lower_cholesky: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.solve(lower_cholesky.T, _solve_lower_triangular(lower_cholesky, target))


def _solve_lower_triangular(lower_cholesky: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.solve(lower_cholesky, target)


def _log_marginal_likelihood(
    training_target: np.ndarray,
    lower_cholesky: np.ndarray,
    alpha: np.ndarray,
) -> float:
    n_observations = training_target.shape[0]
    data_fit = -0.5 * float(training_target @ alpha)
    log_determinant = -float(np.sum(np.log(np.diag(lower_cholesky))))
    normalization = -0.5 * n_observations * np.log(2.0 * np.pi)
    return data_fit + log_determinant + normalization


def _validate_confidence_level(confidence_level: float | None) -> None:
    if confidence_level is None:
        return
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")


def _validate_positive_bounds(bounds: tuple[float, float], name: str) -> None:
    lower, upper = bounds
    if lower <= 0 or upper <= 0 or upper <= lower:
        raise ValueError(f"{name} must contain positive increasing bounds")
