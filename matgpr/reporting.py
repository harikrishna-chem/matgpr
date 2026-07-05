from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "decompose_multifidelity_prediction",
    "summarize_multifidelity_components",
    "summarize_missingness",
    "summarize_numeric_columns",
]


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and fractions by column."""
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_fraction": df.isna().mean(),
            "dtype": df.dtypes.astype(str),
        }
    )
    return report.sort_values("missing_fraction", ascending=False)


def summarize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.describe().T


def decompose_multifidelity_prediction(
    prediction,
    *,
    y_true=None,
    sample_labels: Sequence[object] | None = None,
    model_name: str = "multi_fidelity_gpr",
    split: str = "prediction",
) -> pd.DataFrame:
    """Return a per-sample component table for a multi-fidelity prediction.

    Parameters
    ----------
    prediction
        Object with a ``mean`` attribute and optional multi-fidelity component
        attributes such as ``low_fidelity_mean``, ``correction_mean``, ``rho``,
        ``intercept``, ``std``, ``low_fidelity_std``, and ``correction_std``.
        :class:`matgpr.MultiFidelityGPRPrediction` follows this convention.
    y_true
        Optional measured high-fidelity values. When supplied, signed and
        absolute prediction errors are added.
    sample_labels
        Optional sample identifiers. If omitted, integer positions are used.
    model_name, split
        Labels stored in the returned dataframe for reporting and grouping.

    Returns
    -------
    pandas.DataFrame
        Per-sample rows with the high-fidelity prediction and any available
        low-fidelity, correction, and uncertainty components.
    """
    if not hasattr(prediction, "mean"):
        raise ValueError("prediction must provide a mean attribute")
    y_pred = _to_1d_numeric(getattr(prediction, "mean"), "prediction.mean")
    n_samples = y_pred.shape[0]
    labels = _resolve_sample_labels(sample_labels, n_samples)
    frame = pd.DataFrame(
        {
            "model": str(model_name),
            "split": str(split),
            "sample_position": np.arange(n_samples, dtype=int),
            "sample_label": labels,
            "y_pred": y_pred,
        }
    )

    y_true_array = _optional_1d_numeric(y_true, "y_true")
    if y_true_array is not None:
        _validate_same_length(y_true_array, n_samples, "y_true")
        frame["y_true"] = y_true_array

    for attribute, column in (
        ("std", "y_std"),
        ("lower", "y_lower"),
        ("upper", "y_upper"),
        ("low_fidelity_mean", "low_fidelity_pred"),
        ("low_fidelity_std", "low_fidelity_std"),
        ("correction_mean", "correction_pred"),
        ("correction_std", "correction_std"),
    ):
        values = _optional_prediction_attribute(prediction, attribute)
        if values is not None:
            array = _to_1d_numeric(values, f"prediction.{attribute}")
            _validate_same_length(array, n_samples, f"prediction.{attribute}")
            frame[column] = array

    rho = _optional_scalar_attribute(prediction, "rho")
    intercept = _optional_scalar_attribute(prediction, "intercept")
    if rho is not None:
        frame["rho"] = rho
    if intercept is not None:
        frame["intercept"] = intercept

    return _with_multifidelity_derived_columns(frame)


def summarize_multifidelity_components(
    components,
    *,
    group_by: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Summarize multi-fidelity prediction components for reports.

    ``components`` can be either a per-sample dataframe, such as the
    ``predictions`` table returned by ``multifidelity_learning_curve`` with
    ``store_predictions=True``, or a prediction object accepted by
    :func:`decompose_multifidelity_prediction`.

    The summary reports low-fidelity and correction contribution statistics,
    uncertainty component statistics, reconstruction residuals, and prediction
    error metrics when ``y_true`` is available.
    """
    frame = _component_frame_from_input(components)
    frame = _with_multifidelity_derived_columns(frame)
    group_columns = _resolve_group_columns(frame, group_by)
    rows = []

    for key_values, group in _iter_groups(frame, group_columns):
        row = dict(key_values)
        row["n_samples"] = int(group.shape[0])
        for column in (
            "y_true",
            "y_pred",
            "low_fidelity_input",
            "low_fidelity_pred",
            "scaled_low_fidelity_pred",
            "correction_pred",
            "reconstructed_y_pred",
            "y_std",
            "low_fidelity_std",
            "scaled_low_fidelity_std",
            "correction_std",
            "low_fidelity_variance_fraction",
            "correction_variance_fraction",
        ):
            _add_numeric_column_summary(row, group, column)

        if "correction_pred" in group:
            row["mean_abs_correction_pred"] = _finite_abs_mean(group["correction_pred"])
        if "scaled_low_fidelity_pred" in group:
            row["mean_abs_scaled_low_fidelity_pred"] = _finite_abs_mean(
                group["scaled_low_fidelity_pred"]
            )
        if "component_residual" in group:
            row["max_abs_component_residual"] = _finite_abs_max(group["component_residual"])
        if "signed_error" in group:
            signed_error = _finite_array(group["signed_error"])
            if signed_error.size:
                row["bias"] = float(np.mean(signed_error))
                row["RMSE"] = float(np.sqrt(np.mean(signed_error**2)))
                row["MAE"] = float(np.mean(np.abs(signed_error)))
        rows.append(row)

    return pd.DataFrame(rows)


def _component_frame_from_input(components) -> pd.DataFrame:
    if isinstance(components, pd.DataFrame):
        return components.copy()
    return decompose_multifidelity_prediction(components)


def _with_multifidelity_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _copy_alias(frame, "mean", "y_pred")
    _copy_alias(frame, "std", "y_std")
    _copy_alias(frame, "lower", "y_lower")
    _copy_alias(frame, "upper", "y_upper")
    _copy_alias(frame, "low_fidelity_mean", "low_fidelity_pred")
    _copy_alias(frame, "correction_mean", "correction_pred")

    if {"y_pred", "y_true"}.issubset(frame.columns):
        frame["signed_error"] = _numeric_series(frame["y_pred"]) - _numeric_series(frame["y_true"])
        frame["absolute_error"] = frame["signed_error"].abs()

    if {"low_fidelity_pred", "rho"}.issubset(frame.columns):
        frame["scaled_low_fidelity_pred"] = _numeric_series(frame["rho"]) * _numeric_series(
            frame["low_fidelity_pred"]
        )
    if {"low_fidelity_std", "rho"}.issubset(frame.columns):
        frame["scaled_low_fidelity_std"] = _numeric_series(frame["rho"]).abs() * _numeric_series(
            frame["low_fidelity_std"]
        )

    if {"scaled_low_fidelity_pred", "intercept", "correction_pred"}.issubset(frame.columns):
        frame["reconstructed_y_pred"] = (
            _numeric_series(frame["scaled_low_fidelity_pred"])
            + _numeric_series(frame["intercept"])
            + _numeric_series(frame["correction_pred"])
        )
    if {"y_pred", "reconstructed_y_pred"}.issubset(frame.columns):
        frame["component_residual"] = _numeric_series(frame["y_pred"]) - _numeric_series(
            frame["reconstructed_y_pred"]
        )

    if "scaled_low_fidelity_std" in frame:
        frame["low_fidelity_variance_contribution"] = _numeric_series(
            frame["scaled_low_fidelity_std"]
        ) ** 2
    if "correction_std" in frame:
        frame["correction_variance_contribution"] = _numeric_series(frame["correction_std"]) ** 2
    if "y_std" in frame:
        total_variance = _numeric_series(frame["y_std"]) ** 2
        frame["total_variance"] = total_variance
        _add_variance_fraction(
            frame,
            "low_fidelity_variance_contribution",
            "low_fidelity_variance_fraction",
            total_variance,
        )
        _add_variance_fraction(
            frame,
            "correction_variance_contribution",
            "correction_variance_fraction",
            total_variance,
        )
    return frame


def _copy_alias(frame: pd.DataFrame, source: str, destination: str) -> None:
    if destination not in frame and source in frame:
        frame[destination] = frame[source]


def _add_variance_fraction(
    frame: pd.DataFrame,
    contribution_column: str,
    fraction_column: str,
    total_variance: pd.Series,
) -> None:
    if contribution_column not in frame:
        return
    contribution = _numeric_series(frame[contribution_column])
    frame[fraction_column] = np.where(total_variance > 0.0, contribution / total_variance, np.nan)


def _resolve_group_columns(frame: pd.DataFrame, group_by: Sequence[str] | str | None) -> tuple[str, ...]:
    if group_by is None:
        return tuple(column for column in ("model", "split") if column in frame.columns)
    elif isinstance(group_by, str):
        group_by = (group_by,)
    group_columns = tuple(str(column) for column in group_by)
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing group_by columns: {missing}")
    return group_columns


def _iter_groups(frame: pd.DataFrame, group_columns: tuple[str, ...]):
    if not group_columns:
        yield {}, frame
        return
    grouping = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in frame.groupby(grouping, dropna=False):
        if len(group_columns) == 1:
            keys = (keys,)
        yield dict(zip(group_columns, keys)), group


def _add_numeric_column_summary(row: dict[str, object], group: pd.DataFrame, column: str) -> None:
    if column not in group:
        return
    values = _finite_array(group[column])
    if values.size == 0:
        return
    row[f"{column}_mean"] = float(np.mean(values))
    row[f"{column}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    row[f"{column}_min"] = float(np.min(values))
    row[f"{column}_max"] = float(np.max(values))


def _finite_abs_mean(values) -> float:
    array = _finite_array(values)
    if array.size == 0:
        return np.nan
    return float(np.mean(np.abs(array)))


def _finite_abs_max(values) -> float:
    array = _finite_array(values)
    if array.size == 0:
        return np.nan
    return float(np.max(np.abs(array)))


def _finite_array(values) -> np.ndarray:
    array = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float).ravel()
    return array[np.isfinite(array)]


def _numeric_series(values) -> pd.Series:
    return pd.Series(pd.to_numeric(values, errors="coerce"), index=getattr(values, "index", None))


def _optional_prediction_attribute(prediction, attribute: str):
    if not hasattr(prediction, attribute):
        return None
    value = getattr(prediction, attribute)
    return None if value is None else value


def _optional_scalar_attribute(prediction, attribute: str) -> float | None:
    value = _optional_prediction_attribute(prediction, attribute)
    if value is None:
        return None
    array = np.asarray(value, dtype=float).ravel()
    if array.size != 1:
        raise ValueError(f"prediction.{attribute} must be a scalar")
    if not np.isfinite(array[0]):
        raise ValueError(f"prediction.{attribute} must be finite")
    return float(array[0])


def _optional_1d_numeric(values, name: str) -> np.ndarray | None:
    if values is None:
        return None
    return _to_1d_numeric(values, name)


def _to_1d_numeric(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _resolve_sample_labels(sample_labels: Sequence[object] | None, n_samples: int) -> np.ndarray:
    if sample_labels is None:
        return np.arange(n_samples, dtype=int)
    labels = np.asarray(sample_labels, dtype=object).ravel()
    _validate_same_length(labels, n_samples, "sample_labels")
    return labels


def _validate_same_length(values: Sequence[object], n_samples: int, name: str) -> None:
    if len(values) != n_samples:
        raise ValueError(f"{name} must contain one value per prediction")
