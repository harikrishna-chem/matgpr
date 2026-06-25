from __future__ import annotations

import pandas as pd


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
