from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

__all__ = [
    "calibration_curve",
    "gaussian_nlpd",
    "interval_coverage",
    "prediction_interval_bounds",
    "standardized_residuals",
    "uncertainty_diagnostics",
    "uncertainty_error_correlation",
]


def prediction_interval_bounds(
    y_pred,
    y_std,
    *,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gaussian predictive interval bounds.

    Parameters
    ----------
    y_pred
        Predictive mean values.
    y_std
        Predictive standard deviations in the same units as ``y_pred``.
    confidence_level
        Central interval probability, for example ``0.95``.
    """
    y_pred = _to_1d_array(y_pred, "y_pred")
    y_std = _to_positive_std(y_std)
    _validate_same_length(y_pred, y_std, "y_pred", "y_std")
    confidence_level = _validate_confidence_level(confidence_level)

    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return y_pred - z_value * y_std, y_pred + z_value * y_std


def interval_coverage(
    y_true,
    y_pred,
    y_std,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Calculate observed coverage for a Gaussian prediction interval."""
    y_true, y_pred, y_std = _validate_prediction_arrays(y_true, y_pred, y_std)
    confidence_level = _validate_confidence_level(confidence_level)
    lower, upper = prediction_interval_bounds(
        y_pred,
        y_std,
        confidence_level=confidence_level,
    )
    covered = (y_true >= lower) & (y_true <= upper)
    interval_width = upper - lower
    observed_coverage = float(np.mean(covered))
    return {
        "confidence_level": float(confidence_level),
        "expected_coverage": float(confidence_level),
        "observed_coverage": observed_coverage,
        "coverage_error": observed_coverage - float(confidence_level),
        "mean_interval_width": float(np.mean(interval_width)),
        "median_interval_width": float(np.median(interval_width)),
    }


def calibration_curve(
    y_true,
    y_pred,
    y_std,
    *,
    confidence_levels=None,
) -> pd.DataFrame:
    """Calculate observed coverage across confidence levels.

    The returned dataframe is useful for calibration plots. A well-calibrated
    uncertainty model should have observed coverage close to expected coverage
    across the full range of confidence levels.
    """
    y_true, y_pred, y_std = _validate_prediction_arrays(y_true, y_pred, y_std)
    if confidence_levels is None:
        confidence_levels = np.linspace(0.1, 0.95, 18)
    confidence_levels = _to_confidence_levels(confidence_levels)

    rows = [
        interval_coverage(
            y_true,
            y_pred,
            y_std,
            confidence_level=level,
        )
        for level in confidence_levels
    ]
    return pd.DataFrame(rows)


def gaussian_nlpd(
    y_true,
    y_pred,
    y_std,
    *,
    min_std: float = 1e-12,
    reduction: str = "mean",
):
    """Calculate Gaussian negative log predictive density.

    Lower values indicate sharper and better-calibrated predictive
    distributions. ``reduction="none"`` returns one NLPD value per sample.
    """
    y_true, y_pred, y_std = _validate_prediction_arrays(y_true, y_pred, y_std)
    if min_std <= 0:
        raise ValueError("min_std must be positive")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of: mean, sum, none")

    y_std = np.maximum(y_std, min_std)
    variance = y_std**2
    values = 0.5 * np.log(2.0 * np.pi * variance) + 0.5 * ((y_true - y_pred) ** 2) / variance

    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "sum":
        return float(np.sum(values))
    return values


def standardized_residuals(y_true, y_pred, y_std) -> np.ndarray:
    """Return residuals divided by predictive standard deviation."""
    y_true, y_pred, y_std = _validate_prediction_arrays(y_true, y_pred, y_std)
    return (y_true - y_pred) / y_std


def uncertainty_error_correlation(
    y_true,
    y_pred,
    y_std,
    *,
    method: str = "spearman",
) -> float:
    """Correlate predictive standard deviation with absolute prediction error."""
    y_true, y_pred, y_std = _validate_prediction_arrays(y_true, y_pred, y_std)
    absolute_error = np.abs(y_true - y_pred)
    normalized = str(method).lower()
    if normalized not in {"spearman", "pearson"}:
        raise ValueError("method must be either 'spearman' or 'pearson'")

    if len(y_true) < 2 or np.allclose(absolute_error, absolute_error[0]) or np.allclose(y_std, y_std[0]):
        return np.nan

    if normalized == "spearman":
        return float(spearmanr(y_std, absolute_error).correlation)
    return float(pearsonr(y_std, absolute_error)[0])


def uncertainty_diagnostics(
    y_true,
    y_pred,
    y_std,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Return a compact dictionary of uncertainty diagnostic metrics."""
    y_true, y_pred, y_std = _validate_prediction_arrays(y_true, y_pred, y_std)
    coverage = interval_coverage(
        y_true,
        y_pred,
        y_std,
        confidence_level=confidence_level,
    )
    z_residuals = standardized_residuals(y_true, y_pred, y_std)
    return {
        "NLPD": gaussian_nlpd(y_true, y_pred, y_std),
        "mean_std": float(np.mean(y_std)),
        "median_std": float(np.median(y_std)),
        "mean_absolute_error": float(np.mean(np.abs(y_true - y_pred))),
        "mean_standardized_residual": float(np.mean(z_residuals)),
        "std_standardized_residual": float(np.std(z_residuals, ddof=0)),
        "uncertainty_error_spearman": uncertainty_error_correlation(
            y_true,
            y_pred,
            y_std,
            method="spearman",
        ),
        **coverage,
    }


def _validate_prediction_arrays(y_true, y_pred, y_std) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = _to_1d_array(y_true, "y_true")
    y_pred = _to_1d_array(y_pred, "y_pred")
    y_std = _to_positive_std(y_std)
    _validate_same_length(y_true, y_pred, "y_true", "y_pred")
    _validate_same_length(y_true, y_std, "y_true", "y_std")
    return y_true, y_pred, y_std


def _to_1d_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_positive_std(values) -> np.ndarray:
    array = _to_1d_array(values, "y_std")
    if np.any(array <= 0):
        raise ValueError("y_std must contain only positive values")
    return array


def _validate_same_length(first: np.ndarray, second: np.ndarray, first_name: str, second_name: str) -> None:
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same length")


def _to_confidence_levels(confidence_levels) -> np.ndarray:
    levels = np.asarray(confidence_levels, dtype=float).ravel()
    if levels.size == 0:
        raise ValueError("confidence_levels must contain at least one value")
    for level in levels:
        _validate_confidence_level(level)
    return levels


def _validate_confidence_level(confidence_level: float) -> float:
    confidence_level = float(confidence_level)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    return confidence_level
