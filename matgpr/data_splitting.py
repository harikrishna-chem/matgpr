from __future__ import annotations

from typing import TypeVar

import pandas as pd
from sklearn.model_selection import train_test_split

from ._validation import validate_dataframe

__all__ = [
    "separate_features_target",
    "split_train_test",
]

_FeatureData = TypeVar("_FeatureData")
_TargetData = TypeVar("_TargetData")


def separate_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature columns ``X`` and target vector ``y``.

    This is the first modeling split: it separates the column to predict from
    the columns used as model inputs.
    """
    df = validate_dataframe(df, "df")
    if target_column not in df.columns:
        raise KeyError(f"target_column '{target_column}' not found in dataframe")

    return df.drop(columns=[target_column]), df[target_column]


def split_train_test(
    X: _FeatureData,
    y: _TargetData,
    *,
    test_size: int | float = 0.2,
    random_state: int | None = 42,
    shuffle: bool = True,
    stratify: object | None = None,
) -> list[_FeatureData | _TargetData]:
    """Split features and targets into train/test sets.

    Parameters mirror ``sklearn.model_selection.train_test_split`` so the
    returned values are ``X_train, X_test, y_train, y_test``.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )
