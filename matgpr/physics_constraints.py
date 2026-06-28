from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "AugmentedTrainingData",
    "KnownLimitConstraint",
    "MonotonicTrendConstraint",
    "VirtualObservationSet",
    "append_virtual_observations",
    "combine_virtual_observations",
]


@dataclass(frozen=True)
class VirtualObservationSet:
    """Physics-derived pseudo-observations for soft GPR constraints.

    Virtual observations are useful when physics gives trusted anchor behavior
    that is not directly present in a small experimental dataset. Examples
    include zero-response limits, saturation values, or local monotonic trends.

    These observations are soft constraints: they influence the fitted GP, but
    they do not mathematically guarantee global monotonicity or exact boundary
    behavior.
    """

    X: np.ndarray
    y: np.ndarray
    noise_std: float | Sequence[float] | np.ndarray | None = None
    labels: Sequence[str] | np.ndarray | None = None

    def __post_init__(self) -> None:
        x_values = _to_2d_finite(self.X, "X")
        y_values = _to_1d_finite(self.y, "y")
        _validate_same_length(x_values, y_values, "X", "y")

        noise = None
        if self.noise_std is not None:
            noise = _noise_std_array(self.noise_std, y_values.shape[0])

        labels = _label_array(self.labels, y_values.shape[0], default="virtual")

        object.__setattr__(self, "X", x_values)
        object.__setattr__(self, "y", y_values)
        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "labels", labels)

    @property
    def n_observations(self) -> int:
        """Number of virtual observations."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.X.shape[1])

    @property
    def alpha(self) -> np.ndarray | None:
        """Observation variances suitable for scikit-learn GPR ``alpha``."""
        if self.noise_std is None:
            return None
        return self.noise_std**2


@dataclass(frozen=True)
class KnownLimitConstraint:
    """Create virtual observations at a known feature limit.

    This class encodes boundary behavior such as ``fraction = 0`` at zero time,
    a known saturation value at high loading, or a physically required response
    at a limiting temperature/composition. The selected feature is replaced by
    ``limit_value`` for each reference row while all other features are kept
    unchanged.
    """

    feature: int | str
    limit_value: float
    target_value: float | Callable[[np.ndarray], float | Sequence[float] | np.ndarray]
    noise_std: float | Sequence[float] | np.ndarray = 0.05
    label: str = "known_limit"

    def generate(self, X_reference) -> VirtualObservationSet:
        """Generate virtual observations from reference feature rows."""
        x_values, columns = _as_numeric_matrix(X_reference, "X_reference")
        feature_index = _resolve_feature_index(self.feature, columns, x_values.shape[1])
        limit_value = _finite_scalar(self.limit_value, "limit_value")

        x_virtual = x_values.copy()
        x_virtual[:, feature_index] = limit_value
        y_virtual = _evaluate_target_value(self.target_value, x_virtual)
        noise = _noise_std_array(self.noise_std, x_virtual.shape[0])
        labels = np.full(x_virtual.shape[0], self.label, dtype=object)

        return VirtualObservationSet(
            X=x_virtual,
            y=y_virtual,
            noise_std=noise,
            labels=labels,
        )


@dataclass(frozen=True)
class MonotonicTrendConstraint:
    """Create local virtual observations that encode a monotonic trend.

    The constraint shifts one feature by ``step`` and assigns a target that is
    consistent with an increasing or decreasing response. For an increasing
    trend and positive ``step``, the virtual target is
    ``y_reference + minimum_slope * step``. For a decreasing trend, the sign is
    reversed. Negative steps are allowed and move the anchor in the opposite
    feature direction.

    This is a soft data-augmentation strategy for low-data GPR workflows. It is
    not a replacement for derivative-constrained Gaussian-process inference.
    """

    feature: int | str
    direction: str
    step: float | None = None
    minimum_slope: float = 0.0
    feature_min: float | None = None
    feature_max: float | None = None
    noise_std: float | Sequence[float] | np.ndarray = 0.1
    label: str | None = None

    def generate(self, X_reference, y_reference) -> VirtualObservationSet:
        """Generate monotonic virtual observations from reference rows."""
        x_values, columns = _as_numeric_matrix(X_reference, "X_reference")
        y_values = _to_1d_finite(y_reference, "y_reference")
        _validate_same_length(x_values, y_values, "X_reference", "y_reference")

        feature_index = _resolve_feature_index(self.feature, columns, x_values.shape[1])
        response_sign = _response_sign(self.direction)
        step = _resolve_step(self.step, x_values[:, feature_index])
        minimum_slope = _nonnegative_scalar(self.minimum_slope, "minimum_slope")

        x_virtual = x_values.copy()
        x_virtual[:, feature_index] = x_virtual[:, feature_index] + step
        y_virtual = y_values + response_sign * minimum_slope * step
        noise = _noise_std_array(self.noise_std, x_virtual.shape[0])

        keep_mask = _feature_bounds_mask(
            x_virtual[:, feature_index],
            feature_min=self.feature_min,
            feature_max=self.feature_max,
        )
        if not np.any(keep_mask):
            raise ValueError("No monotonic virtual observations remain after applying feature bounds")

        x_virtual = x_virtual[keep_mask]
        y_virtual = y_virtual[keep_mask]
        noise = noise[keep_mask]
        label = self.label or f"monotonic_{_normalized_direction(self.direction)}"
        labels = np.full(x_virtual.shape[0], label, dtype=object)

        return VirtualObservationSet(
            X=x_virtual,
            y=y_virtual,
            noise_std=noise,
            labels=labels,
        )


@dataclass(frozen=True)
class AugmentedTrainingData:
    """Training data after appending virtual physics observations."""

    X: pd.DataFrame | np.ndarray
    y: np.ndarray
    alpha: np.ndarray | None
    labels: np.ndarray


def combine_virtual_observations(*virtual_observations: VirtualObservationSet) -> VirtualObservationSet:
    """Combine compatible virtual-observation sets into one set."""
    sets = _flatten_virtual_observation_sets(virtual_observations)
    if not sets:
        raise ValueError("At least one VirtualObservationSet is required")

    n_features = sets[0].n_features
    for observation_set in sets:
        if observation_set.n_features != n_features:
            raise ValueError("All virtual observation sets must have the same number of features")

    x_combined = np.vstack([observation_set.X for observation_set in sets])
    y_combined = np.concatenate([observation_set.y for observation_set in sets])
    labels = np.concatenate([observation_set.labels for observation_set in sets])

    noise = None
    if any(observation_set.noise_std is not None for observation_set in sets):
        noise = np.concatenate(
            [
                observation_set.noise_std
                if observation_set.noise_std is not None
                else np.zeros(observation_set.n_observations, dtype=float)
                for observation_set in sets
            ]
        )

    return VirtualObservationSet(X=x_combined, y=y_combined, noise_std=noise, labels=labels)


def append_virtual_observations(
    X,
    y,
    *virtual_observations: VirtualObservationSet,
    base_alpha: float | Sequence[float] | np.ndarray | None = 1e-8,
    observed_label: str = "observed",
) -> AugmentedTrainingData:
    """Append virtual physics observations to a training dataset.

    Parameters
    ----------
    X, y
        Real training features and targets.
    virtual_observations
        One or more :class:`VirtualObservationSet` objects.
    base_alpha
        Observation variance for real training rows. If supplied, the returned
        ``alpha`` vector can be passed to scikit-learn's
        ``GaussianProcessRegressor``. Virtual rows use their own
        ``noise_std**2`` values.
    observed_label
        Label assigned to real training rows in the returned metadata.
    """
    x_values, columns = _as_numeric_matrix(X, "X")
    y_values = _to_1d_finite(y, "y")
    _validate_same_length(x_values, y_values, "X", "y")

    sets = _flatten_virtual_observation_sets(virtual_observations)
    for observation_set in sets:
        if observation_set.n_features != x_values.shape[1]:
            raise ValueError("Virtual observations must have the same number of features as X")

    x_parts = [x_values, *[observation_set.X for observation_set in sets]]
    y_parts = [y_values, *[observation_set.y for observation_set in sets]]

    x_augmented = np.vstack(x_parts)
    y_augmented = np.concatenate(y_parts)
    labels = np.concatenate(
        [
            np.full(y_values.shape[0], observed_label, dtype=object),
            *[observation_set.labels for observation_set in sets],
        ]
    )

    alpha = _augmented_alpha(base_alpha, y_values.shape[0], sets)
    if columns is None:
        x_output = x_augmented
    else:
        x_output = pd.DataFrame(x_augmented, columns=columns)

    return AugmentedTrainingData(
        X=x_output,
        y=y_augmented,
        alpha=alpha,
        labels=labels,
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


def _resolve_feature_index(feature: int | str, columns: list[str] | None, n_features: int) -> int:
    if isinstance(feature, str):
        if columns is None:
            raise ValueError("String feature names require X_reference or X to be a dataframe")
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


def _evaluate_target_value(
    target_value: float | Callable[[np.ndarray], float | Sequence[float] | np.ndarray],
    x_virtual: np.ndarray,
) -> np.ndarray:
    if callable(target_value):
        values = target_value(x_virtual)
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return np.full(x_virtual.shape[0], float(array), dtype=float)
        array = array.ravel()
        if array.shape[0] != x_virtual.shape[0]:
            raise ValueError("Callable target_value must return one value per virtual observation")
        if not np.all(np.isfinite(array)):
            raise ValueError("Callable target_value must return only finite values")
        return array

    return np.full(x_virtual.shape[0], _finite_scalar(target_value, "target_value"), dtype=float)


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


def _resolve_step(step: float | None, feature_values: np.ndarray) -> float:
    if step is not None:
        result = _finite_scalar(step, "step")
        if result == 0:
            raise ValueError("step must be non-zero")
        return result

    feature_range = float(np.max(feature_values) - np.min(feature_values))
    if feature_range <= 0:
        raise ValueError("step must be provided when the selected feature has zero range")
    return 0.05 * feature_range


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


def _flatten_virtual_observation_sets(
    virtual_observations,
) -> list[VirtualObservationSet]:
    if len(virtual_observations) == 1 and isinstance(virtual_observations[0], (list, tuple)):
        virtual_observations = tuple(virtual_observations[0])
    sets = list(virtual_observations)
    if not all(isinstance(observation_set, VirtualObservationSet) for observation_set in sets):
        raise TypeError("virtual_observations must contain only VirtualObservationSet objects")
    return sets


def _augmented_alpha(
    base_alpha: float | Sequence[float] | np.ndarray | None,
    n_observed: int,
    virtual_observations: Sequence[VirtualObservationSet],
) -> np.ndarray | None:
    if base_alpha is None and not any(observation_set.noise_std is not None for observation_set in virtual_observations):
        return None

    observed_alpha = (
        np.zeros(n_observed, dtype=float)
        if base_alpha is None
        else _alpha_array(base_alpha, n_observed, "base_alpha")
    )
    virtual_alpha = [
        observation_set.alpha
        if observation_set.alpha is not None
        else np.zeros(observation_set.n_observations, dtype=float)
        for observation_set in virtual_observations
    ]
    return np.concatenate([observed_alpha, *virtual_alpha])


def _alpha_array(values, n_observations: int, name: str) -> np.ndarray:
    if np.isscalar(values):
        alpha = np.full(n_observations, _nonnegative_scalar(values, name), dtype=float)
    else:
        alpha = _to_1d_finite(values, name)
        if alpha.shape[0] != n_observations:
            raise ValueError(f"{name} must be a scalar or have one value per observed row")
        if np.any(alpha < 0):
            raise ValueError(f"{name} must be non-negative")
    return alpha
