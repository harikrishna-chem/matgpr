from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "append_closed_loop_records",
    "load_closed_loop_log",
    "log_bo_recommendations",
    "log_observations",
    "log_selected_experiments",
    "summarize_closed_loop_log",
]

_RESERVED_LOG_COLUMNS = (
    "matgpr_campaign_id",
    "matgpr_iteration",
    "matgpr_record_type",
    "matgpr_timestamp_utc",
)
_SUMMARY_GROUP_COLUMNS = (
    "matgpr_campaign_id",
    "matgpr_iteration",
    "matgpr_record_type",
)


def append_closed_loop_records(
    records: pd.DataFrame,
    *,
    path: str | Path,
    campaign_id: str,
    iteration: int,
    record_type: str,
    metadata: Mapping[str, Any] | None = None,
    timestamp: datetime | str | None = None,
) -> pd.DataFrame:
    """Append annotated rows to a closed-loop experiment CSV log.

    Parameters
    ----------
    records
        Dataframe of recommendations, selected experiments, or measured
        observations to record.
    path
        CSV path for the campaign log. Existing logs are read and rewritten so
        new record types can add new columns while preserving previous rows.
    campaign_id
        Stable label for the optimization campaign.
    iteration
        Non-negative closed-loop iteration number.
    record_type
        Short label such as `"recommendation"`, `"selection"`, or
        `"observation"`.
    metadata
        Optional scalar metadata written to `matgpr_metadata_*` columns.
    timestamp
        Optional timestamp. Datetime values are converted to UTC. If omitted,
        the current UTC time is used.

    Returns
    -------
    pandas.DataFrame
        The annotated rows that were appended to the log.
    """
    raw_records = _validate_records(records)
    campaign_id = _validate_campaign_id(campaign_id)
    iteration = _validate_iteration(iteration)
    record_type = _validate_record_type(record_type)
    timestamp_text = _format_timestamp(timestamp)
    metadata_columns = _metadata_to_columns(metadata)

    collisions = [
        column
        for column in (*_RESERVED_LOG_COLUMNS, *metadata_columns)
        if column in raw_records.columns
    ]
    if collisions:
        raise ValueError(
            "records contain columns reserved for closed-loop logging: "
            f"{collisions}"
        )

    prefix = pd.DataFrame(
        {
            "matgpr_campaign_id": campaign_id,
            "matgpr_iteration": iteration,
            "matgpr_record_type": record_type,
            "matgpr_timestamp_utc": timestamp_text,
            **metadata_columns,
        },
        index=raw_records.index,
    )
    annotated = pd.concat(
        [prefix.reset_index(drop=True), raw_records.reset_index(drop=True)],
        axis=1,
    )

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, annotated], ignore_index=True, sort=False)
    else:
        combined = annotated.reset_index(drop=True)
    combined.to_csv(log_path, index=False)

    return annotated.reset_index(drop=True)


def log_bo_recommendations(
    recommendations: pd.DataFrame,
    *,
    path: str | Path,
    campaign_id: str,
    iteration: int,
    model_name: str | None = None,
    acquisition_function: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp: datetime | str | None = None,
) -> pd.DataFrame:
    """Log Bayesian-optimization recommendations for a campaign iteration."""
    merged_metadata = _merge_metadata(
        metadata,
        {
            "model_name": model_name,
            "acquisition_function": acquisition_function,
        },
    )
    return append_closed_loop_records(
        recommendations,
        path=path,
        campaign_id=campaign_id,
        iteration=iteration,
        record_type="recommendation",
        metadata=merged_metadata,
        timestamp=timestamp,
    )


def log_selected_experiments(
    selected_experiments: pd.DataFrame,
    *,
    path: str | Path,
    campaign_id: str,
    iteration: int,
    selection_policy: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp: datetime | str | None = None,
) -> pd.DataFrame:
    """Log the recommendations actually selected for experimental execution."""
    merged_metadata = _merge_metadata(
        metadata,
        {"selection_policy": selection_policy},
    )
    return append_closed_loop_records(
        selected_experiments,
        path=path,
        campaign_id=campaign_id,
        iteration=iteration,
        record_type="selection",
        metadata=merged_metadata,
        timestamp=timestamp,
    )


def log_observations(
    observations: pd.DataFrame,
    *,
    path: str | Path,
    campaign_id: str,
    iteration: int,
    target_column: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp: datetime | str | None = None,
) -> pd.DataFrame:
    """Log measured outcomes returned from selected experiments."""
    merged_metadata = _merge_metadata(
        metadata,
        {"target_column": target_column},
    )
    return append_closed_loop_records(
        observations,
        path=path,
        campaign_id=campaign_id,
        iteration=iteration,
        record_type="observation",
        metadata=merged_metadata,
        timestamp=timestamp,
    )


def load_closed_loop_log(path: str | Path) -> pd.DataFrame:
    """Load a closed-loop experiment CSV log."""
    return pd.read_csv(Path(path))


def summarize_closed_loop_log(
    log_data: pd.DataFrame | str | Path,
    *,
    campaign_id: str | None = None,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Summarize closed-loop records by campaign, iteration, and record type.

    The summary always reports record counts. When `target_column` is provided,
    the summary also reports target count, mean, minimum, and maximum for each
    group. This is useful for quickly auditing whether a BO campaign has
    moved from recommendation to selection to measured outcome.
    """
    log = _coerce_log_data(log_data)
    _validate_log_schema(log)

    if campaign_id is not None:
        campaign_id = _validate_campaign_id(campaign_id)
        log = log.loc[log["matgpr_campaign_id"] == campaign_id].copy()

    summary_columns = [*_SUMMARY_GROUP_COLUMNS, "matgpr_record_count"]
    if log.empty:
        if target_column is not None:
            summary_columns.extend(
                [
                    "matgpr_target_count",
                    "matgpr_target_mean",
                    "matgpr_target_min",
                    "matgpr_target_max",
                ]
            )
        return pd.DataFrame(columns=summary_columns)

    summary = (
        log.groupby(list(_SUMMARY_GROUP_COLUMNS), dropna=False)
        .size()
        .reset_index(name="matgpr_record_count")
    )

    if target_column is not None:
        target_column = str(target_column).strip()
        if not target_column:
            raise ValueError("target_column must be non-empty when provided")
        if target_column not in log.columns:
            raise ValueError(f"target_column {target_column!r} is missing from the log")

        target_log = log.assign(
            _matgpr_target=pd.to_numeric(log[target_column], errors="coerce")
        )
        target_summary = (
            target_log.groupby(list(_SUMMARY_GROUP_COLUMNS), dropna=False)["_matgpr_target"]
            .agg(
                matgpr_target_count=lambda values: int(values.notna().sum()),
                matgpr_target_mean="mean",
                matgpr_target_min="min",
                matgpr_target_max="max",
            )
            .reset_index()
        )
        summary = summary.merge(
            target_summary,
            on=list(_SUMMARY_GROUP_COLUMNS),
            how="left",
        )

    return summary.sort_values(list(_SUMMARY_GROUP_COLUMNS)).reset_index(drop=True)


def _validate_records(records: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    if records.empty:
        raise ValueError("records must contain at least one row")
    return records.copy()


def _validate_campaign_id(campaign_id: str) -> str:
    campaign_id = str(campaign_id).strip()
    if not campaign_id:
        raise ValueError("campaign_id must be non-empty")
    return campaign_id


def _validate_iteration(iteration: int) -> int:
    if isinstance(iteration, bool) or not isinstance(iteration, Integral):
        raise TypeError("iteration must be an integer")
    iteration = int(iteration)
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return iteration


def _validate_record_type(record_type: str) -> str:
    record_type = str(record_type).strip().lower().replace("-", "_")
    if not record_type:
        raise ValueError("record_type must be non-empty")
    return record_type


def _format_timestamp(timestamp: datetime | str | None) -> str:
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        return timestamp.isoformat()
    timestamp_text = str(timestamp).strip()
    if not timestamp_text:
        raise ValueError("timestamp must be non-empty when provided")
    return timestamp_text


def _metadata_to_columns(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping of scalar values")

    columns: dict[str, Any] = {}
    for key, value in metadata.items():
        slug = _slugify(str(key))
        if not slug:
            raise ValueError("metadata keys must be non-empty after normalization")
        column = f"matgpr_metadata_{slug}"
        if column in columns:
            raise ValueError(f"metadata keys produce duplicate column {column!r}")
        if not pd.api.types.is_scalar(value):
            raise TypeError(
                "metadata values must be scalar so they can be written to CSV; "
                f"got non-scalar value for {key!r}"
            )
        columns[column] = value
    return columns


def _merge_metadata(
    metadata: Mapping[str, Any] | None,
    extra: Mapping[str, Any | None],
) -> Mapping[str, Any] | None:
    if metadata is None:
        combined: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        combined = dict(metadata)
    else:
        raise TypeError("metadata must be a mapping of scalar values")

    existing_slugs = {_slugify(str(key)) for key in combined}
    for key, value in extra.items():
        if value is None:
            continue
        slug = _slugify(str(key))
        if slug in existing_slugs:
            raise ValueError(f"metadata already contains a value for {key!r}")
        combined[key] = value
        existing_slugs.add(slug)

    return combined or None


def _coerce_log_data(log_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(log_data, pd.DataFrame):
        return log_data.copy()
    if isinstance(log_data, (str, Path)):
        return load_closed_loop_log(log_data)
    raise TypeError("log_data must be a dataframe or CSV path")


def _validate_log_schema(log: pd.DataFrame) -> None:
    missing = [column for column in _SUMMARY_GROUP_COLUMNS if column not in log.columns]
    if missing:
        raise ValueError(f"log is missing required closed-loop columns: {missing}")


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())).strip(
        "_"
    )
