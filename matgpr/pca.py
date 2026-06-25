from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca(
    X,
    n_components: int | float | None = None,
    *,
    scale: bool = False,
) -> tuple[pd.DataFrame, PCA, StandardScaler | None]:
    """Fit PCA and return component scores, the fitted PCA, and optional scaler.

    DataFrames are reduced to numeric columns before PCA. Set ``scale=True`` to
    fit a ``StandardScaler`` before PCA; the fitted scaler is returned so new
    data can be transformed consistently.
    """
    X_values = _numeric_values(X, context="PCA")
    scaler = StandardScaler() if scale else None

    if scaler is not None:
        X_values = scaler.fit_transform(X_values)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_values)
    return _scores_dataframe(scores), pca, scaler


def summarize_pca(pca: PCA) -> pd.DataFrame:
    """Return explained and cumulative variance for a fitted PCA model."""
    explained = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(explained))],
            "explained_variance_ratio": explained,
            "cumulative_variance_ratio": np.cumsum(explained),
        }
    )


def transform_pca(
    X,
    pca: PCA,
    *,
    scaler: StandardScaler | None = None,
    imputer=None,
) -> pd.DataFrame:
    """Transform new data using an already-fitted PCA model.

    Pass the same scaler and imputer used for training data so new data follows
    the identical preprocessing path.
    """
    original_index = X.index if isinstance(X, pd.DataFrame) else None
    X_values = _numeric_values(X, context="PCA transform")

    if imputer is not None:
        X_values = imputer.transform(X_values)
    if scaler is not None:
        X_values = scaler.transform(X_values)

    scores = pca.transform(X_values)
    return _scores_dataframe(scores, index=original_index)


def _numeric_values(X, *, context: str) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include=np.number)
        if X.empty:
            raise ValueError(f"{context} requires at least one numeric column")
        return X.to_numpy()
    return np.asarray(X)


def _scores_dataframe(scores, index=None) -> pd.DataFrame:
    columns = [f"PC{i + 1}" for i in range(scores.shape[1])]
    return pd.DataFrame(scores, columns=columns, index=index)
