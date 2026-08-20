from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "json_safe_metrics",
    "json_safe_regression_metrics",
    "json_safe_train_test_regression_metrics",
    "regression_metrics",
    "train_test_regression_metrics",
]


def regression_metrics(
    y_true,
    y_pred,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Calculate common regression metrics for model analysis.

    The returned dictionary contains R2, RMSE, MAE, and Pearson r. Use
    ``prefix`` to distinguish train/test metrics when storing experiment
    results, for example ``prefix="test"`` gives ``test_R2``.
    """
    y_true = _as_1d_float_array(y_true, "y_true")
    y_pred = _as_1d_float_array(y_pred, "y_pred")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = _r2_score(y_true, y_pred)
    r = _pearson_r(y_true, y_pred)

    metric_prefix = f"{prefix}_" if prefix else ""
    return {
        f"{metric_prefix}R2": r2,
        f"{metric_prefix}RMSE": rmse,
        f"{metric_prefix}MAE": mae,
        f"{metric_prefix}r": r,
    }


def train_test_regression_metrics(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
) -> dict[str, float]:
    """Calculate regression metrics for both train and test predictions."""
    metrics = {}
    metrics.update(
        regression_metrics(
            y_train_true,
            y_train_pred,
            prefix="train",
        )
    )
    metrics.update(
        regression_metrics(
            y_test_true,
            y_test_pred,
            prefix="test",
        )
    )
    return metrics


def json_safe_metrics(
    metrics: Mapping[str, Any],
    *,
    include_warnings: bool = True,
    warning_key: str = "metric_warnings",
) -> dict[str, Any]:
    """Return a strict-JSON-safe copy of a metric dictionary.

    Scientific metrics such as R2, Pearson r, or calibration correlations can
    be undefined for legitimate edge cases, including one-sample splits or
    constant arrays. The numerical metric helpers keep those values as
    ``NaN``. This helper converts non-finite scalar values to ``None`` so the
    result can be serialized with ``json.dumps(..., allow_nan=False)``.
    """
    if include_warnings and warning_key in metrics:
        raise ValueError(f"warning_key {warning_key!r} conflicts with a metric key")

    safe: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in metrics.items():
        safe_value, value_warnings = _to_json_safe_value(value, path=str(key))
        safe[str(key)] = safe_value
        warnings.extend(value_warnings)

    if include_warnings and warnings:
        safe[warning_key] = warnings
    return safe


def json_safe_regression_metrics(
    y_true,
    y_pred,
    *,
    prefix: str = "",
    include_warnings: bool = True,
    warning_key: str = "metric_warnings",
) -> dict[str, Any]:
    """Calculate regression metrics and return a strict-JSON-safe dictionary."""
    return json_safe_metrics(
        regression_metrics(y_true, y_pred, prefix=prefix),
        include_warnings=include_warnings,
        warning_key=warning_key,
    )


def json_safe_train_test_regression_metrics(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
    *,
    include_warnings: bool = True,
    warning_key: str = "metric_warnings",
) -> dict[str, Any]:
    """Calculate train/test regression metrics with non-finite values as ``None``."""
    return json_safe_metrics(
        train_test_regression_metrics(
            y_train_true,
            y_train_pred,
            y_test_true,
            y_test_pred,
        ),
        include_warnings=include_warnings,
        warning_key=warning_key,
    )


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return np.nan
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    return correlation if math.isfinite(correlation) else np.nan


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.ptp(y_true) == 0:
        return np.nan
    try:
        value = float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan
    return value if math.isfinite(value) else np.nan


def _as_1d_float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_json_safe_value(value: Any, *, path: str) -> tuple[Any, list[str]]:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool) or value is None:
        return value, []

    if isinstance(value, float):
        if not math.isfinite(value):
            return None, [f"{path} is non-finite and was converted to None"]
        return value, []

    if isinstance(value, int | str):
        return value, []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, Mapping):
        safe_mapping: dict[str, Any] = {}
        warnings: list[str] = []
        for key, nested_value in value.items():
            nested_path = f"{path}.{key}"
            safe_value, nested_warnings = _to_json_safe_value(
                nested_value,
                path=nested_path,
            )
            safe_mapping[str(key)] = safe_value
            warnings.extend(nested_warnings)
        return safe_mapping, warnings

    if isinstance(value, list | tuple):
        safe_list = []
        warnings = []
        for index, nested_value in enumerate(value):
            safe_value, nested_warnings = _to_json_safe_value(
                nested_value,
                path=f"{path}[{index}]",
            )
            safe_list.append(safe_value)
            warnings.extend(nested_warnings)
        return safe_list, warnings

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value), [f"{path} is not JSON serializable and was converted to string"]
    return value, []
