from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

__all__ = [
    "load_artifact",
    "log_experiment_result",
    "save_artifact",
]


def save_artifact(artifact, path: str | Path) -> None:
    """Save a fitted preprocessor, model, or full pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path):
    """Load a joblib artifact saved by ``save_artifact``."""
    return joblib.load(path)


def log_experiment_result(
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
