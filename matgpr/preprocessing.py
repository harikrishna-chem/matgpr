from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

__all__ = [
    "build_preprocessor",
    "build_scaler",
    "identify_feature_types",
]


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns in a dataframe.

    Returns
    -------
    numeric_features, categorical_features
        Lists that can be passed directly into ``build_preprocessor``.
    """
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_scaler(name: str = "standard"):
    """Create a scikit-learn feature scaler by name.

    Parameters
    ----------
    name
        ``"standard"`` for zero-mean/unit-variance scaling, ``"minmax"`` for
        range scaling, ``"robust"`` for median/IQR scaling, or ``"none"`` /
        ``"passthrough"`` to leave features unchanged in a pipeline.
    """
    normalized = name.lower()

    if normalized == "standard":
        return StandardScaler()
    if normalized == "minmax":
        return MinMaxScaler()
    if normalized == "robust":
        return RobustScaler()
    if normalized in {"none", "passthrough"}:
        return "passthrough"

    raise ValueError("name must be one of: standard, minmax, robust, none")


def build_preprocessor(
    *,
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
    scaler: str = "standard",
    numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
) -> ColumnTransformer:
    """Build a reusable scikit-learn preprocessing transformer.

    Fit this transformer only on training data, then reuse it for validation,
    test, or prediction data. Numeric columns are imputed and optionally scaled;
    categorical columns are imputed and one-hot encoded.
    """
    transformers = []
    numeric_features = list(numeric_features or [])
    categorical_features = list(categorical_features or [])

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=numeric_imputation)),
                ("scaler", build_scaler(scaler)),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_imputation)),
                ("onehot", _build_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("Provide at least one numeric or categorical feature")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
