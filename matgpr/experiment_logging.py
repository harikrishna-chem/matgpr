from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "BOCampaignState",
    "append_closed_loop_records",
    "infer_next_bo_iteration",
    "load_closed_loop_log",
    "log_bo_recommendations",
    "log_observations",
    "log_selected_experiments",
    "resume_bo_campaign",
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


@dataclass(frozen=True)
class BOCampaignState:
    """Restart state for a finite-pool closed-loop BO campaign.

    Parameters
    ----------
    campaign_id
        Stable campaign identifier.
    key_columns
        Candidate key columns used to match recommendations, selections,
        observations, and candidate-pool rows.
    current_iteration
        Largest iteration number currently present in the campaign log, or
        `None` when the campaign has not logged any rows.
    last_recommendation_iteration
        Largest recommendation iteration currently present in the campaign log,
        or `None` when no BO ask has been logged yet.
    next_iteration
        Recommended iteration number for the next BO ask. This is inferred
        from recommendation rows, so observation rows can be logged as the
        iteration that will be used for the next model update.
    log
        Campaign-specific closed-loop log rows.
    recommendations
        Logged recommendation rows.
    selections
        Logged selection rows.
    observations
        Logged observation rows.
    pending_experiments
        Selected candidates that do not yet have matching observation rows.
    completed_experiments
        Observation rows, de-duplicated by key with the latest row kept.
    unavailable_candidates
        Candidate-pool rows or log rows that should not be recommended again.
        This includes completed observations and pending selections.
    available_candidates
        Candidate-pool rows that remain available after removing unavailable
        keys. Empty when `candidate_pool` was not provided.
    candidate_pool_size
        Number of rows in the original candidate pool, or `None` when no pool
        was provided.
    """

    campaign_id: str
    key_columns: tuple[str, ...]
    current_iteration: int | None
    last_recommendation_iteration: int | None
    next_iteration: int
    log: pd.DataFrame
    recommendations: pd.DataFrame
    selections: pd.DataFrame
    observations: pd.DataFrame
    pending_experiments: pd.DataFrame
    completed_experiments: pd.DataFrame
    unavailable_candidates: pd.DataFrame
    available_candidates: pd.DataFrame
    candidate_pool_size: int | None = None

    def duplicate_policy(
        self,
        *,
        key_columns: tuple[str, ...] | None = None,
        feature_columns: tuple[str, ...] | None = None,
        feature_tolerance: float | None = None,
        metric: str = "euclidean",
        feature_scales: Any | None = None,
    ) -> Any:
        """Build a duplicate policy for the next BO recommendation step."""
        from .bayesian_optimization import CandidateDuplicatePolicy

        resolved_keys = self.key_columns if key_columns is None else _validate_key_columns(key_columns)
        return CandidateDuplicatePolicy(
            existing_candidates=self.unavailable_candidates,
            key_columns=resolved_keys,
            feature_columns=feature_columns,
            feature_tolerance=feature_tolerance,
            metric=metric,
            feature_scales=feature_scales,
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


def infer_next_bo_iteration(
    log_data: pd.DataFrame | str | Path,
    *,
    campaign_id: str | None = None,
    recommendation_record_type: str = "recommendation",
) -> int:
    """Infer the next BO ask iteration from a closed-loop campaign log.

    The next iteration is based on logged recommendation rows. If a campaign
    has no recommendation rows yet, the next ask iteration is `0`.
    """
    log = _coerce_campaign_log_data(log_data)
    if log.empty:
        return 0
    _validate_log_schema(log)

    if campaign_id is not None:
        campaign_id = _validate_campaign_id(campaign_id)
        log = log.loc[log["matgpr_campaign_id"] == campaign_id].copy()
    if log.empty:
        return 0

    recommendation_record_type = _validate_record_type(recommendation_record_type)
    recommendations = log.loc[
        log["matgpr_record_type"] == recommendation_record_type
    ].copy()
    if recommendations.empty:
        return 0
    iterations = _numeric_iteration_series(recommendations, label="recommendation log")
    return int(iterations.max()) + 1


def resume_bo_campaign(
    log_data: pd.DataFrame | str | Path,
    *,
    campaign_id: str,
    key_columns: tuple[str, ...] = ("candidate_id",),
    candidate_pool: pd.DataFrame | None = None,
    recommendation_record_type: str = "recommendation",
    selection_record_type: str = "selection",
    observation_record_type: str = "observation",
) -> BOCampaignState:
    """Build restart state for an ask-tell Bayesian-optimization campaign.

    Use this helper at the beginning of a new session to recover the next
    iteration number, pending selections, completed observations, unavailable
    candidate keys, and the remaining candidate pool.
    """
    campaign_id = _validate_campaign_id(campaign_id)
    key_columns = _validate_key_columns(key_columns)
    recommendation_record_type = _validate_record_type(recommendation_record_type)
    selection_record_type = _validate_record_type(selection_record_type)
    observation_record_type = _validate_record_type(observation_record_type)

    log = _coerce_campaign_log_data(log_data)
    if not log.empty:
        _validate_log_schema(log)
        log = log.loc[log["matgpr_campaign_id"] == campaign_id].reset_index(drop=True)
    else:
        log = _empty_closed_loop_log()

    candidate_pool_data = _validate_candidate_pool(candidate_pool, key_columns)

    recommendations = _records_by_type(log, recommendation_record_type)
    selections = _records_by_type(log, selection_record_type)
    observations = _records_by_type(log, observation_record_type)
    completed = _latest_records_by_key(observations, key_columns, label="observation records")
    pending = _pending_selection_records(selections, completed, key_columns)

    unavailable_records = pd.concat(
        [completed, pending],
        ignore_index=True,
        sort=False,
    )
    if unavailable_records.empty:
        unavailable_records = pd.DataFrame(columns=list(key_columns))
    unavailable_keys = _row_key_set(unavailable_records, key_columns)

    if candidate_pool_data is None:
        available_candidates = pd.DataFrame()
        unavailable_candidates = unavailable_records.reset_index(drop=True)
        candidate_pool_size = None
    else:
        candidate_keys = _row_keys(candidate_pool_data, key_columns, label="candidate_pool")
        available_mask = [key not in unavailable_keys for key in candidate_keys]
        unavailable_mask = [not keep for keep in available_mask]
        available_candidates = candidate_pool_data.loc[available_mask].reset_index(drop=True)
        unavailable_candidates = candidate_pool_data.loc[unavailable_mask].reset_index(drop=True)
        candidate_pool_size = int(candidate_pool_data.shape[0])

    current_iteration = _latest_iteration(log)
    last_recommendation_iteration = _latest_iteration(recommendations)
    next_iteration = infer_next_bo_iteration(
        log,
        campaign_id=campaign_id,
        recommendation_record_type=recommendation_record_type,
    )

    return BOCampaignState(
        campaign_id=campaign_id,
        key_columns=key_columns,
        current_iteration=current_iteration,
        last_recommendation_iteration=last_recommendation_iteration,
        next_iteration=next_iteration,
        log=log.reset_index(drop=True),
        recommendations=recommendations.reset_index(drop=True),
        selections=selections.reset_index(drop=True),
        observations=observations.reset_index(drop=True),
        pending_experiments=pending.reset_index(drop=True),
        completed_experiments=completed.reset_index(drop=True),
        unavailable_candidates=unavailable_candidates,
        available_candidates=available_candidates,
        candidate_pool_size=candidate_pool_size,
    )


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


def _coerce_campaign_log_data(log_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(log_data, pd.DataFrame):
        return log_data.copy()
    if isinstance(log_data, (str, Path)):
        path = Path(log_data)
        if not path.exists() or path.stat().st_size == 0:
            return _empty_closed_loop_log()
        return load_closed_loop_log(path)
    raise TypeError("log_data must be a dataframe or CSV path")


def _empty_closed_loop_log() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_RESERVED_LOG_COLUMNS])


def _validate_log_schema(log: pd.DataFrame) -> None:
    missing = [column for column in _SUMMARY_GROUP_COLUMNS if column not in log.columns]
    if missing:
        raise ValueError(f"log is missing required closed-loop columns: {missing}")


def _validate_key_columns(key_columns: tuple[str, ...] | str) -> tuple[str, ...]:
    if isinstance(key_columns, str):
        key_columns = (key_columns,)
    try:
        columns = tuple(str(column).strip() for column in key_columns)
    except TypeError as exc:
        raise TypeError("key_columns must be an iterable of column names") from exc
    if not columns or any(not column for column in columns):
        raise ValueError("key_columns must contain at least one non-empty column name")
    if len(set(columns)) != len(columns):
        raise ValueError("key_columns must be unique")
    return columns


def _validate_candidate_pool(
    candidate_pool: pd.DataFrame | None,
    key_columns: tuple[str, ...],
) -> pd.DataFrame | None:
    if candidate_pool is None:
        return None
    if not isinstance(candidate_pool, pd.DataFrame):
        raise TypeError("candidate_pool must be a pandas DataFrame")
    missing = [column for column in key_columns if column not in candidate_pool.columns]
    if missing:
        raise ValueError(f"candidate_pool is missing key columns: {missing}")
    return candidate_pool.copy()


def _records_by_type(log: pd.DataFrame, record_type: str) -> pd.DataFrame:
    if log.empty:
        return log.copy()
    return log.loc[log["matgpr_record_type"] == record_type].copy()


def _latest_records_by_key(
    records: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    label: str,
) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    _row_keys(records, key_columns, label=label)
    return records.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)


def _pending_selection_records(
    selections: pd.DataFrame,
    completed: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    if selections.empty:
        return selections.copy()
    latest_selections = _latest_records_by_key(
        selections,
        key_columns,
        label="selection records",
    )
    completed_keys = _row_key_set(completed, key_columns)
    selection_keys = _row_keys(
        latest_selections,
        key_columns,
        label="selection records",
    )
    pending_mask = [key not in completed_keys for key in selection_keys]
    return latest_selections.loc[pending_mask].reset_index(drop=True)


def _row_key_set(records: pd.DataFrame, key_columns: tuple[str, ...]) -> set[tuple[Any, ...]]:
    if records.empty:
        return set()
    return set(_row_keys(records, key_columns, label="records"))


def _row_keys(
    records: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    label: str,
) -> list[tuple[Any, ...]]:
    missing = [column for column in key_columns if column not in records.columns]
    if missing:
        raise ValueError(f"{label} are missing key columns: {missing}")

    missing_sentinel = ("__matgpr_missing_key__",)
    return [
        tuple(missing_sentinel if pd.isna(value) else value for value in row)
        for row in records.loc[:, list(key_columns)].itertuples(index=False, name=None)
    ]


def _latest_iteration(log: pd.DataFrame) -> int | None:
    if log.empty:
        return None
    iterations = _numeric_iteration_series(log, label="campaign log")
    return int(iterations.max())


def _numeric_iteration_series(log: pd.DataFrame, *, label: str) -> pd.Series:
    if "matgpr_iteration" not in log.columns:
        raise ValueError(f"{label} is missing matgpr_iteration")
    iterations = pd.to_numeric(log["matgpr_iteration"], errors="coerce")
    if iterations.isna().any():
        raise ValueError(f"{label} contains missing or non-numeric iterations")
    if (iterations < 0).any():
        raise ValueError(f"{label} contains negative iterations")
    return iterations.astype(int)


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())).strip(
        "_"
    )
