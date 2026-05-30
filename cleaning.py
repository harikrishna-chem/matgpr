from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


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
        out[column] = out[column].map(
            lambda value: np.nan
            if str(value).strip().lower() in normalized
            else value
        )

    return out


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Remove duplicate rows and reset the index.

    Parameters
    ----------
    df
        Input dataframe.
    subset
        Optional columns used to identify duplicates. When omitted, every
        column is considered.
    """
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def drop_columns_by_missing_fraction(
    df: pd.DataFrame,
    max_missing_fraction: float = 0.5,
) -> pd.DataFrame:
    """Drop columns whose missing fraction is above ``max_missing_fraction``.

    ``max_missing_fraction=0.5`` keeps columns with up to 50 percent missing
    values and removes columns with more missing data than that.
    """
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")

    keep = df.isna().mean() <= max_missing_fraction
    return df.loc[:, keep].copy()


def impute_missing_values(
    df: pd.DataFrame,
    *,
    strategy: str = "median",
    columns: Iterable[str] | None = None,
    fill_value: object | None = None,
) -> pd.DataFrame:
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
    """
    out = df.copy()

    if columns is None:
        columns = out.select_dtypes(include="number").columns

    selected_columns = list(columns)
    if not selected_columns:
        raise ValueError("No columns selected for imputation")

    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    out[selected_columns] = imputer.fit_transform(out[selected_columns])
    return out


def filter_iqr_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Remove rows with IQR outliers in selected numeric columns.

    Missing values are preserved by the filter. Use imputation before or after
    this step depending on whether missing values should influence the computed
    quartiles.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")

    mask = pd.Series(True, index=df.index)

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= df[column].between(lower, upper) | df[column].isna()

    return df.loc[mask].reset_index(drop=True)
