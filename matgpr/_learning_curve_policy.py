"""Pure input and metric policy for learning-curve workflows."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

__all__: list[str] = []


def _resolve_learning_curve_train_sizes(
    train_sizes: Sequence[float | int] | None,
    *,
    start: float | int,
    stop: float | int,
    step: float | int,
    unit: str,
) -> tuple[dict[str, float | int | str], ...]:
    unit = _normalize_train_size_unit(unit)
    values = (
        tuple(train_sizes)
        if train_sizes is not None
        else _inclusive_train_size_range(start=start, stop=stop, step=step)
    )
    if not values:
        raise ValueError("train_sizes must contain at least one value")

    resolved = []
    for value in values:
        requested_value = float(value)
        if unit == "percent":
            if not 0 < requested_value <= 100:
                raise ValueError("Percent train sizes must be in the interval (0, 100]")
            model_train_size: float | int = requested_value / 100.0
        elif unit == "fraction":
            if not 0 < requested_value <= 1:
                raise ValueError("Fraction train sizes must be in the interval (0, 1]")
            model_train_size = requested_value
        else:
            requested_float = float(value)
            if not requested_float.is_integer():
                raise ValueError("Count train sizes must be integers")
            count = int(requested_float)
            if count <= 0:
                raise ValueError("Count train sizes must be positive")
            model_train_size = count
            requested_value = count
        resolved.append(
            {
                "requested_train_size": requested_value,
                "requested_train_size_unit": unit,
                "model_train_size": model_train_size,
            }
        )
    return tuple(resolved)


def _inclusive_train_size_range(
    *,
    start: float | int,
    stop: float | int,
    step: float | int,
) -> tuple[float, ...]:
    start = float(start)
    stop = float(stop)
    step = float(step)
    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if not math.isfinite(value):
            raise ValueError(f"train_size_{name} must be finite")
    if step <= 0:
        raise ValueError("train_size_step must be positive")
    if stop < start:
        raise ValueError("train_size_stop must be greater than or equal to train_size_start")

    values = []
    current = start
    tolerance = abs(step) * 1e-9
    while current <= stop + tolerance:
        values.append(round(current, 12))
        current += step
    return tuple(values)


def _normalize_train_size_unit(unit: str) -> str:
    normalized = str(unit).strip().lower()
    aliases = {
        "percent": "percent",
        "percentage": "percent",
        "%": "percent",
        "fraction": "fraction",
        "proportion": "fraction",
        "count": "count",
        "counts": "count",
        "n": "count",
    }
    if normalized not in aliases:
        raise ValueError("train_size_unit must be one of: percent, fraction, count")
    return aliases[normalized]


def _normalize_metric_names(metrics: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(metrics, str):
        metrics = (metrics,)
    names = tuple(dict.fromkeys(_normalize_metric_name(metric) for metric in metrics))
    if not names:
        raise ValueError("metrics must contain at least one metric name")
    return names


def _normalize_metric_name(metric: str) -> str:
    normalized = str(metric).strip().lower()
    aliases = {
        "rmse": "RMSE",
        "root_mean_squared_error": "RMSE",
        "r2": "R2",
        "r^2": "R2",
        "mae": "MAE",
        "mean_absolute_error": "MAE",
        "r": "r",
        "pearson": "r",
        "pearson_r": "r",
    }
    if normalized not in aliases:
        raise ValueError("metrics can include only: RMSE, R2, MAE, r")
    return aliases[normalized]


def _normalize_metric_splits(splits: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(splits, str):
        if splits.strip().lower().replace("_", "-") in {"train-test", "both"}:
            return ("train", "test")
        splits = (splits,)
    normalized = tuple(dict.fromkeys(str(split).strip().lower() for split in splits))
    if not normalized:
        raise ValueError("metric_splits must contain at least one split")
    unknown = [split for split in normalized if split not in {"train", "test"}]
    if unknown:
        raise ValueError("metric_splits can include only 'train' and 'test'")
    return normalized


def _metric_column_name(split: str, metric: str) -> str:
    split = _normalize_metric_splits(split)[0]
    metric = _normalize_metric_name(metric)
    return f"{split}_{metric}"


def _metric_columns(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] | str,
    splits: Sequence[str] | str,
) -> tuple[str, ...]:
    columns = tuple(
        _metric_column_name(split, metric)
        for split in _normalize_metric_splits(splits)
        for metric in _normalize_metric_names(metrics)
    )
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Metric columns are not present in the result: {missing}")
    return columns
