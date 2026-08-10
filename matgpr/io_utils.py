from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    metrics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    path: str | Path = "results.csv",
) -> None:
    """Append one experiment result row to a schema-safe CSV file.

    ``metadata`` is written before metric columns and is intended for values
    such as model name, kernel, train size, random state, or feature set.

    If the CSV already exists, the file is read, aligned to the union of old
    and new columns, and rewritten. Existing column order is preserved and new
    columns are appended at the end. This keeps experiment logs readable when
    later runs add or omit metrics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row_data: dict[str, Any] = {}
    if metadata is not None:
        _update_row_data(row_data, metadata, source_name="metadata")
    _update_row_data(row_data, metrics, source_name="metrics")

    row = pd.DataFrame([row_data])
    if not path.exists() or path.stat().st_size == 0:
        row.to_csv(path, index=False)
        return

    try:
        existing = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        row.to_csv(path, index=False)
        return

    columns = _merged_columns(existing.columns, row.columns)
    combined = pd.concat(
        [
            existing.reindex(columns=columns),
            row.reindex(columns=columns),
        ],
        ignore_index=True,
    )
    combined.to_csv(path, index=False)


def _update_row_data(
    row_data: dict[str, Any],
    values: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    for key, value in values.items():
        column = str(key)
        if column in row_data:
            raise ValueError(
                f"Duplicate experiment result column {column!r} found while reading {source_name}"
            )
        row_data[column] = value


def _merged_columns(existing_columns, new_columns) -> list[str]:
    columns = [str(column) for column in existing_columns]
    seen = set(columns)
    for column in new_columns:
        normalized = str(column)
        if normalized not in seen:
            columns.append(normalized)
            seen.add(normalized)
    return columns
