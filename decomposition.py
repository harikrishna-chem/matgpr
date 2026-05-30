from __future__ import annotations

import matplotlib.pyplot as plt
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


def plot_pca_scree(
    pca: PCA,
    *,
    figsize: tuple[float, float] = (6, 4),
    title: str = "PCA Scree Plot",
):
    """Plot explained variance and cumulative variance from a fitted PCA."""
    explained = pca.explained_variance_ratio_
    components = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(components, explained, alpha=0.7, label="Explained variance")
    ax.plot(components, cumulative, marker="o", label="Cumulative variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title(title)
    ax.set_xticks(components)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_pca_scores(
    train_scores: pd.DataFrame,
    *,
    test_scores: pd.DataFrame | None = None,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    figsize: tuple[float, float] = (6, 5),
    title: str = "PCA Scores",
):
    """Plot PCA scores for training data and optionally overlay test data."""
    _require_columns(train_scores, [pc_x, pc_y], "train_scores")
    if test_scores is not None:
        _require_columns(test_scores, [pc_x, pc_y], "test_scores")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        train_scores[pc_x],
        train_scores[pc_y],
        s=55,
        alpha=0.75,
        label="Train",
        edgecolor="k",
        linewidth=0.4,
    )

    if test_scores is not None:
        ax.scatter(
            test_scores[pc_x],
            test_scores[pc_y],
            s=75,
            alpha=0.95,
            label="Test",
            marker="^",
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


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


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")
