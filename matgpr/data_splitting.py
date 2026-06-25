from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def separate_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature columns ``X`` and target vector ``y``.

    This is the first modeling split: it separates the column to predict from
    the columns used as model inputs.
    """
    if target_column not in df.columns:
        raise KeyError(f"target_column '{target_column}' not found in dataframe")

    return df.drop(columns=[target_column]), df[target_column]


def split_train_test(
    X,
    y,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    stratify=None,
):
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
