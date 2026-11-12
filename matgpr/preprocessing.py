from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from ._validation import validate_dataframe

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
    X = validate_dataframe(X, "X")
    if any(not isinstance(column, str) for column in X.columns):
        raise TypeError("X column names must be strings")
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_scaler(
    name: str = "standard",
) -> StandardScaler | MinMaxScaler | RobustScaler | Literal["passthrough"]:
    """Create a scikit-learn feature scaler by name.

    Parameters
    ----------
    name
        ``"standard"`` for zero-mean/unit-variance scaling, ``"minmax"`` for
        range scaling, ``"robust"`` for median/IQR scaling, or ``"none"`` /
        ``"passthrough"`` to leave features unchanged in a pipeline.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    normalized = name.lower()

    if normalized == "standard":
        return StandardScaler()
    if normalized == "minmax":
        return MinMaxScaler()
    if normalized == "robust":
        return RobustScaler()
    if normalized in {"none", "passthrough"}:
        return "passthrough"

    raise ValueError("name must be one of: standard, minmax, robust, none, passthrough")


def build_preprocessor(
    *,
    numeric_features: str | Iterable[str] | None = None,
    categorical_features: str | Iterable[str] | None = None,
    scaler: str = "standard",
    numeric_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
) -> ColumnTransformer:
    """Build a reusable scikit-learn preprocessing transformer.

    Fit this transformer only on training data, then reuse it for validation,
    test, or prediction data. Numeric columns are imputed and optionally scaled;
    categorical columns are imputed and one-hot encoded. Feature names must be
    unique strings, and numeric and categorical groups must not overlap.
    """
    transformers = []
    numeric_features = _normalize_feature_names(numeric_features, "numeric_features")
    categorical_features = _normalize_feature_names(
        categorical_features,
        "categorical_features",
    )
    categorical_set = set(categorical_features)
    overlap = [name for name in numeric_features if name in categorical_set]
    if overlap:
        raise ValueError(
            "numeric_features and categorical_features must not overlap: " + ", ".join(overlap)
        )

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
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("Provide at least one numeric or categorical feature")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _normalize_feature_names(
    features: str | Iterable[str] | None,
    argument_name: str,
) -> list[str]:
    """Return one validated list of unique column names."""
    if features is None:
        return []
    if isinstance(features, str):
        names = [features]
    else:
        try:
            names = list(features)
        except TypeError as exc:
            raise TypeError(
                f"{argument_name} must be a column name or iterable of column names"
            ) from exc

    if any(not isinstance(name, str) for name in names):
        raise TypeError(f"{argument_name} must contain only string column names")
    if len(names) != len(set(names)):
        raise ValueError(f"{argument_name} must contain unique names")
    return names
