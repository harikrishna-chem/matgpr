from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ._validation import (
    feature_bounds_mask,
    finite_scalar,
    label_array,
    noise_std_array,
    nonnegative_scalar,
    normalized_direction,
    response_sign,
    to_1d_finite,
    to_2d_finite,
    validate_same_row_count,
)

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
        x_values = to_2d_finite(self.X, "X")
        y_values = to_1d_finite(self.y, "y")
        validate_same_row_count(x_values, y_values, "X", "y")

        noise = None
        if self.noise_std is not None:
            noise = noise_std_array(self.noise_std, y_values.shape[0])

        labels = label_array(self.labels, y_values.shape[0], default="virtual")

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
        limit_value = finite_scalar(self.limit_value, "limit_value")

        x_virtual = x_values.copy()
        x_virtual[:, feature_index] = limit_value
        y_virtual = _evaluate_target_value(self.target_value, x_virtual)
        noise = noise_std_array(self.noise_std, x_virtual.shape[0])
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
        y_values = to_1d_finite(y_reference, "y_reference")
        validate_same_row_count(x_values, y_values, "X_reference", "y_reference")

        feature_index = _resolve_feature_index(self.feature, columns, x_values.shape[1])
        direction_sign = response_sign(self.direction)
        step = _resolve_step(self.step, x_values[:, feature_index])
        minimum_slope = nonnegative_scalar(self.minimum_slope, "minimum_slope")

        x_virtual = x_values.copy()
        x_virtual[:, feature_index] = x_virtual[:, feature_index] + step
        y_virtual = y_values + direction_sign * minimum_slope * step
        noise = noise_std_array(self.noise_std, x_virtual.shape[0])

        keep_mask = feature_bounds_mask(
            x_virtual[:, feature_index],
            feature_min=self.feature_min,
            feature_max=self.feature_max,
        )
        if not np.any(keep_mask):
            raise ValueError("No monotonic virtual observations remain after applying feature bounds")

        x_virtual = x_virtual[keep_mask]
        y_virtual = y_virtual[keep_mask]
        noise = noise[keep_mask]
        label = self.label or f"monotonic_{normalized_direction(self.direction)}"
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
    y_values = to_1d_finite(y, "y")
    validate_same_row_count(x_values, y_values, "X", "y")

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
    return to_2d_finite(array, name), columns


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

    return np.full(x_virtual.shape[0], finite_scalar(target_value, "target_value"), dtype=float)


def _resolve_step(step: float | None, feature_values: np.ndarray) -> float:
    if step is not None:
        result = finite_scalar(step, "step")
        if result == 0:
            raise ValueError("step must be non-zero")
        return result

    feature_range = float(np.max(feature_values) - np.min(feature_values))
    if feature_range <= 0:
        raise ValueError("step must be provided when the selected feature has zero range")
    return 0.05 * feature_range


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
        alpha = np.full(n_observations, nonnegative_scalar(values, name), dtype=float)
    else:
        alpha = to_1d_finite(values, name)
        if alpha.shape[0] != n_observations:
            raise ValueError(f"{name} must be a scalar or have one value per observed row")
        if np.any(alpha < 0):
            raise ValueError(f"{name} must be non-negative")
    return alpha
