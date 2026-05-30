from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .scaling import make_scaler
except ImportError:
    from scaling import make_scaler


def infer_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns from a dataframe.

    Returns
    -------
    numeric_features, categorical_features
        Two lists that can be passed directly into ``build_preprocessor``.
    """
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


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
                ("scaler", make_scaler(scaler)),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_imputation)),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("Provide at least one numeric or categorical feature")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def save_artifact(artifact, path: str | Path) -> None:
    """Save a fitted preprocessor, model, or full pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path):
    """Load a joblib artifact saved by ``save_artifact``."""
    return joblib.load(path)


def append_experiment_result(
    metrics: dict,
    *,
    metadata: dict | None = None,
    path: str | Path = "results.csv",
) -> None:
    """Append one experiment result row to a CSV file.

    ``metadata`` is written before metric columns and is intended for values
    such as model name, kernel, train size, random state, or feature set.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row_data = {}
    if metadata is not None:
        row_data.update(metadata)
    row_data.update(metrics)

    row = pd.DataFrame([row_data])
    file_exists = path.exists()
    row.to_csv(
        path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
