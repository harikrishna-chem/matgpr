from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

__all__ = [
    "drop_columns_by_missing_fraction",
    "drop_duplicate_rows",
    "filter_iqr_outliers",
    "impute_missing_values",
    "normalize_column_names",
    "replace_missing_placeholders",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent lowercase snake_case column names.

    Use this as an early cleaning step so later functions can refer to stable
    feature names. Non-alphanumeric runs are converted to underscores, leading
    and trailing underscores are removed, and duplicate generated names receive
    a numeric suffix.
    """
    out = df.copy()
    seen: dict[str, int] = {}
    columns: list[str] = []

    for column in out.columns:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(column).strip().lower())
        name = re.sub(r"_+", "_", name).strip("_") or "column"

        count = seen.get(name, 0)
        seen[name] = count + 1
        columns.append(name if count == 0 else f"{name}_{count + 1}")

    out.columns = columns
    return out


def replace_missing_placeholders(
    df: pd.DataFrame,
    placeholders: Iterable[str] = ("", "nan", "none", "null", "na", "n/a", "-"),
) -> pd.DataFrame:
    """Replace common string placeholders for missing values with ``np.nan``.

    Only object/string-like columns are inspected. Numeric columns are left
    unchanged so valid numeric zeros or negative values cannot be altered by
    accident.
    """
    out = df.copy()
    normalized = {str(value).strip().lower() for value in placeholders}

    for column in out.select_dtypes(include=["object", "string"]).columns:
        text = out[column].astype("string").str.strip().str.lower()
        mask = text.isin(normalized).fillna(False)
        out.loc[mask, column] = np.nan

    return out


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: Iterable[str] | None = None,
    *,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Remove duplicate rows.

    Parameters
    ----------
    df
        Input dataframe.
    subset
        Optional columns used to identify duplicates. When omitted, every
        column is considered.
    reset_index
        If ``True``, return a compact ``RangeIndex``. Set ``False`` to preserve
        material IDs or other meaningful row indices.
    """
    subset_columns = None if subset is None else list(subset)
    out = df.drop_duplicates(subset=subset_columns)
    return out.reset_index(drop=True) if reset_index else out.copy()


def drop_columns_by_missing_fraction(
    df: pd.DataFrame,
    max_missing_fraction: float = 0.5,
    *,
    return_dropped: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Drop columns whose missing fraction is above ``max_missing_fraction``.

    ``max_missing_fraction=0.5`` keeps columns with up to 50 percent missing
    values and removes columns with more missing data than that.
    """
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")

    keep = df.isna().mean() <= max_missing_fraction
    cleaned = df.loc[:, keep].copy()
    if return_dropped:
        return cleaned, df.columns[~keep].tolist()
    return cleaned


def impute_missing_values(
    df: pd.DataFrame,
    *,
    strategy: str = "median",
    columns: Iterable[str] | None = None,
    fill_value: object | None = None,
    return_imputer: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, SimpleImputer]:
    """Impute missing values in selected dataframe columns.

    Parameters
    ----------
    df
        Input dataframe.
    strategy
        Any ``sklearn.impute.SimpleImputer`` strategy: ``"mean"``,
        ``"median"``, ``"most_frequent"``, or ``"constant"``.
    columns
        Columns to impute. If omitted, numeric columns are selected.
    fill_value
        Replacement value used only when ``strategy="constant"``.
    return_imputer
        If ``True``, also return the fitted ``SimpleImputer`` so the same
        transformation can be applied to validation or test data.
    """
    out = df.copy()

    if columns is None:
        columns = out.select_dtypes(include="number").columns

    selected_columns = list(columns)
    if not selected_columns:
        raise ValueError("No columns selected for imputation")

    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    out[selected_columns] = imputer.fit_transform(out[selected_columns])
    if return_imputer:
        return out, imputer
    return out


def filter_iqr_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    factor: float = 1.5,
    drop_if: Literal["any", "all"] = "any",
    reset_index: bool = True,
) -> pd.DataFrame:
    """Remove rows with IQR outliers in selected numeric columns.

    Missing values are preserved by the filter. Use imputation before or after
    this step depending on whether missing values should influence the computed
    quartiles.

    ``drop_if="any"`` keeps the historical behavior: a row is removed if it is
    an outlier in at least one selected column. ``drop_if="all"`` removes a row
    only when it is an outlier in every selected column.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    if drop_if not in {"any", "all"}:
        raise ValueError("drop_if must be either 'any' or 'all'")

    selected_columns = list(columns)
    if not selected_columns:
        raise ValueError("At least one column is required")

    outlier_masks: list[pd.Series] = []

    for column in selected_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(q1) or pd.isna(q3) or not np.isfinite(iqr) or iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        outlier_masks.append(~(df[column].between(lower, upper) | df[column].isna()))

    if not outlier_masks:
        return df.reset_index(drop=True) if reset_index else df.copy()

    outlier_table = pd.concat(outlier_masks, axis=1)
    if drop_if == "any":
        keep = ~outlier_table.any(axis=1)
    else:
        keep = ~outlier_table.all(axis=1)

    filtered = df.loc[keep]
    return filtered.reset_index(drop=True) if reset_index else filtered.copy()
