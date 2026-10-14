"""Shared input-validation helpers used across ``matgpr`` modules.

This module is private. It exists so that the same check is defined once
rather than copied into each module that needs it, which previously let the
copies drift apart. Behavior and error messages are kept exactly as they were
in the per-module copies, because those messages are part of what users see.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

__all__ = [
    "cached_float",
    "feature_bounds_mask",
    "finite_scalar",
    "label_array",
    "noise_std_array",
    "nonnegative_scalar",
    "normalize_fidelity_label",
    "normalized_direction",
    "response_sign",
    "to_1d_finite",
    "to_2d_finite",
    "validate_confidence_level",
    "validate_positive_int",
    "validate_same_length",
    "validate_same_row_count",
    "validate_task_covar_rank",
]


def cached_float(value: object) -> float:
    """Return ``value`` as a float, mapping ``None`` to ``NaN``."""
    if value is None:
        return np.nan
    return float(value)


def finite_scalar(value: float, name: str) -> float:
    """Return ``value`` as a float, rejecting ``NaN`` and infinities."""
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def nonnegative_scalar(value: float, name: str) -> float:
    """Return ``value`` as a finite, non-negative float."""
    result = finite_scalar(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def to_1d_finite(values, name: str) -> np.ndarray:
    """Return ``values`` as a non-empty one-dimensional finite float array."""
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def to_2d_finite(values, name: str) -> np.ndarray:
    """Return ``values`` as a non-empty two-dimensional finite float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def label_array(labels, n_observations: int, *, default: str) -> np.ndarray:
    """Return one object-dtype label per observation, defaulting when omitted."""
    if labels is None:
        return np.full(n_observations, default, dtype=object)
    array = np.asarray(labels, dtype=object).ravel()
    if array.shape[0] != n_observations:
        raise ValueError("labels must have one value per observation")
    return array


def noise_std_array(values, n_observations: int) -> np.ndarray:
    """Return per-observation noise standard deviations from a scalar or array."""
    if np.isscalar(values):
        noise = np.full(n_observations, nonnegative_scalar(values, "noise_std"), dtype=float)
    else:
        noise = to_1d_finite(values, "noise_std")
        if noise.shape[0] != n_observations:
            raise ValueError("noise_std must be a scalar or have one value per observation")
        if np.any(noise < 0):
            raise ValueError("noise_std must be non-negative")
    return noise


def feature_bounds_mask(
    feature_values: np.ndarray,
    *,
    feature_min: float | None,
    feature_max: float | None,
) -> np.ndarray:
    """Return a mask for feature values inside the optional bounds."""
    if feature_min is not None and feature_max is not None:
        lower = finite_scalar(feature_min, "feature_min")
        upper = finite_scalar(feature_max, "feature_max")
        if upper < lower:
            raise ValueError("feature_max must be greater than or equal to feature_min")
    elif feature_min is not None:
        lower = finite_scalar(feature_min, "feature_min")
        upper = np.inf
    elif feature_max is not None:
        lower = -np.inf
        upper = finite_scalar(feature_max, "feature_max")
    else:
        lower = -np.inf
        upper = np.inf
    return (feature_values >= lower) & (feature_values <= upper)


def normalized_direction(direction: str) -> str:
    """Return ``"increasing"`` or ``"decreasing"`` from a trend-direction alias."""
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


def response_sign(direction: str) -> float:
    """Return ``+1`` for an increasing trend and ``-1`` for a decreasing one."""
    return 1.0 if normalized_direction(direction) == "increasing" else -1.0


def normalize_fidelity_label(value, name: str) -> str:
    """Return a stripped, non-missing fidelity label."""
    if value is None:
        raise ValueError(f"{name} must not contain missing labels")
    label = str(value).strip()
    if label == "" or label.lower() == "nan":
        raise ValueError(f"{name} must not contain missing labels")
    return label


def validate_confidence_level(confidence_level: float | None) -> float | None:
    """Return a validated central confidence level, passing ``None`` through.

    ``None`` means no interval was requested, so callers that treat the level
    as optional can pass it straight in.
    """
    if confidence_level is None:
        return None
    confidence_level = float(confidence_level)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    return confidence_level


def validate_positive_int(value: int, *, name: str) -> int:
    """Return ``value`` as an integer of at least one."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def validate_task_covar_rank(task_covar_rank: int, num_tasks: int) -> int:
    """Return a task-covariance rank between one and the number of tasks."""
    if not isinstance(task_covar_rank, int):
        raise ValueError("task_covar_rank must be an integer")
    if task_covar_rank < 1:
        raise ValueError("task_covar_rank must be at least 1")
    if task_covar_rank > num_tasks:
        raise ValueError("task_covar_rank cannot be larger than the number of tasks")
    return task_covar_rank


def validate_same_length(
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> None:
    """Raise when two arrays do not have the same leading dimension."""
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same length")


def validate_same_row_count(
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> None:
    """Raise when two arrays do not have the same number of rows.

    Separate from :func:`validate_same_length` only because the modules that
    work with observation tables phrase the error in terms of rows.
    """
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same number of rows")
