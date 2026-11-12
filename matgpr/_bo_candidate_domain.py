from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

__all__: list[str] = []


class _TrustRegionLike(Protocol):
    centers: Any
    radius: float
    metric: str
    feature_columns: tuple[str, ...] | None
    feature_scales: Any | None


class _DuplicatePolicyLike(Protocol):
    existing_candidates: Any
    key_columns: tuple[str, ...] | None
    feature_columns: tuple[str, ...] | None
    feature_tolerance: float | None
    metric: str
    feature_scales: Any | None
    existing_id_column: str | None


def _validate_candidate_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in {"filter", "annotate"}:
        raise ValueError("candidate policy must be 'filter' or 'annotate'")
    return normalized


def _validate_output_name(value: str, *, name: str) -> str:
    output_name = str(value).strip()
    if not output_name:
        raise ValueError(f"{name} must be a non-empty column name")
    return output_name


def _validate_distinct_names(*names: str) -> None:
    normalized = tuple(str(name) for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("output column names must be distinct")


def _normalize_distance_metric(metric: str) -> str:
    normalized = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "euclidean": "euclidean",
        "l2": "euclidean",
        "manhattan": "manhattan",
        "cityblock": "manhattan",
        "l1": "manhattan",
        "chebyshev": "chebyshev",
        "linf": "chebyshev",
        "l_inf": "chebyshev",
        "max": "chebyshev",
    }
    if normalized not in aliases:
        raise ValueError("metric must be 'euclidean', 'manhattan', or 'chebyshev'")
    return aliases[normalized]


def _trust_region_annotations(
    candidate_features: pd.DataFrame | np.ndarray,
    trust_region: _TrustRegionLike,
    *,
    expected_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_matrix = _policy_feature_matrix(
        candidate_features,
        feature_columns=trust_region.feature_columns,
        expected_rows=expected_rows,
        allow_empty_rows=True,
        name="candidate_features",
    )
    center_matrix = _policy_feature_matrix(
        trust_region.centers,
        feature_columns=trust_region.feature_columns,
        expected_features=candidate_matrix.shape[1],
        allow_empty_rows=False,
        name="trust_region.centers",
    )
    scales = _resolve_policy_feature_scales(
        trust_region.feature_scales,
        candidate_matrix,
        center_matrix,
        name="trust_region.feature_scales",
    )
    distances = _minimum_scaled_distances(
        candidate_matrix,
        center_matrix,
        metric=trust_region.metric,
        scales=scales,
    )
    return distances, distances <= trust_region.radius


def _duplicate_policy_annotations(
    candidates: pd.DataFrame,
    candidate_features: pd.DataFrame | np.ndarray,
    duplicate_policy: _DuplicatePolicyLike,
    *,
    expected_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    is_duplicate = np.zeros(candidates.shape[0], dtype=bool)
    reason_parts: list[list[str]] = [[] for _ in range(candidates.shape[0])]
    distances = None
    matched_ids = np.full(candidates.shape[0], None, dtype=object)
    existing = duplicate_policy.existing_candidates
    existing_ids = _existing_candidate_ids(existing, duplicate_policy.existing_id_column)

    if duplicate_policy.key_columns is not None:
        if not isinstance(duplicate_policy.existing_candidates, pd.DataFrame):
            raise TypeError(
                "existing_candidates must be a pandas DataFrame when key_columns are used"
            )
        candidate_keys = _candidate_policy_row_keys(
            candidates,
            duplicate_policy.key_columns,
            label="candidate key columns",
        )
        existing_key_rows = _candidate_policy_row_keys(
            duplicate_policy.existing_candidates,
            duplicate_policy.key_columns,
            label="existing key columns",
        )
        existing_key_to_id = {
            key: existing_ids[index] for index, key in enumerate(existing_key_rows)
        }
        existing_keys = set(existing_key_to_id)
        key_duplicates = np.asarray([key in existing_keys for key in candidate_keys], dtype=bool)
        is_duplicate |= key_duplicates
        for row_index, duplicate in enumerate(key_duplicates):
            if duplicate:
                reason_parts[row_index].append("key")
                matched_ids[row_index] = existing_key_to_id[candidate_keys[row_index]]

    if duplicate_policy.feature_tolerance is not None:
        candidate_matrix = _policy_feature_matrix(
            candidate_features,
            feature_columns=duplicate_policy.feature_columns,
            expected_rows=expected_rows,
            allow_empty_rows=True,
            name="candidate_features",
        )
        existing_matrix = _policy_feature_matrix(
            duplicate_policy.existing_candidates,
            feature_columns=duplicate_policy.feature_columns,
            expected_features=candidate_matrix.shape[1],
            allow_empty_rows=True,
            name="duplicate_policy.existing_candidates",
        )
        scales = _resolve_policy_feature_scales(
            duplicate_policy.feature_scales,
            candidate_matrix,
            existing_matrix,
            name="duplicate_policy.feature_scales",
        )
        distances, nearest_positions = _minimum_scaled_distance_matches(
            candidate_matrix,
            existing_matrix,
            metric=duplicate_policy.metric,
            scales=scales,
        )
        feature_duplicates = distances <= duplicate_policy.feature_tolerance
        is_duplicate |= feature_duplicates
        for row_index, duplicate in enumerate(feature_duplicates):
            if duplicate:
                reason_parts[row_index].append("feature")
                if matched_ids[row_index] is None and nearest_positions[row_index] >= 0:
                    matched_ids[row_index] = existing_ids[nearest_positions[row_index]]

    reasons = np.asarray(["; ".join(parts) for parts in reason_parts], dtype=object)
    return is_duplicate, distances, reasons, matched_ids


def _policy_feature_matrix(
    values: pd.DataFrame | np.ndarray,
    *,
    feature_columns: tuple[str, ...] | None,
    expected_features: int | None = None,
    expected_rows: int | None = None,
    allow_empty_rows: bool,
    name: str,
) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        columns = _resolve_policy_feature_columns(values, feature_columns, name=name)
        matrix = values.loc[:, list(columns)].to_numpy(dtype=float)
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if (
            feature_columns is not None
            and matrix.ndim == 2
            and len(feature_columns) != matrix.shape[1]
        ):
            raise ValueError(
                f"{name} feature_columns length must match feature count "
                f"({len(feature_columns)} != {matrix.shape[1]})"
            )

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D numeric array or dataframe")
    if matrix.shape[0] == 0 and not allow_empty_rows:
        raise ValueError(f"{name} must contain at least one row")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            f"{name} feature count must match candidate features "
            f"({matrix.shape[1]} != {expected_features})"
        )
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{name} rows must match candidate rows ({matrix.shape[0]} != {expected_rows})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def _resolve_policy_feature_columns(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...] | None,
    *,
    name: str,
) -> tuple[Any, ...]:
    if feature_columns is not None:
        column_lookup = {str(column): column for column in data.columns}
        columns = tuple(
            column if column in data.columns else column_lookup.get(str(column))
            for column in feature_columns
        )
        missing = [
            requested for requested, resolved in zip(feature_columns, columns) if resolved is None
        ]
        if missing:
            raise ValueError(f"{name} feature columns are missing: {missing}")
    else:
        columns = tuple(
            column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])
        )
        if not columns:
            raise ValueError(
                f"No numeric columns could be inferred from {name}; provide feature_columns"
            )

    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"{name} feature columns must be numeric: {non_numeric}")
    return columns


def _resolve_policy_feature_scales(
    feature_scales: Any | None,
    candidate_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    n_features = candidate_matrix.shape[1]
    if feature_scales is None:
        scales = np.ones(n_features, dtype=float)
    elif isinstance(feature_scales, str):
        normalized = feature_scales.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in {"std", "standard_deviation", "standardize"}:
            raise ValueError(f"{name} must be None, 'std', or one positive scale per feature")
        if reference_matrix.shape[0] == 0:
            combined = candidate_matrix
        elif candidate_matrix.shape[0] == 0:
            combined = reference_matrix
        else:
            combined = np.vstack([candidate_matrix, reference_matrix])
        scales = combined.std(axis=0)
        scales[~np.isfinite(scales) | (scales <= 0.0)] = 1.0
    else:
        scales = np.asarray(feature_scales, dtype=float).reshape(-1)
        if scales.shape[0] != n_features:
            raise ValueError(
                f"{name} length must match feature count ({scales.shape[0]} != {n_features})"
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError(f"{name} must contain finite positive values")
    return scales.astype(float)


def _minimum_scaled_distances(
    candidate_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    metric: str,
    scales: np.ndarray,
) -> np.ndarray:
    if candidate_matrix.shape[0] == 0:
        return np.asarray([], dtype=float)
    if reference_matrix.shape[0] == 0:
        return np.full(candidate_matrix.shape[0], np.inf, dtype=float)

    scaled_diff = np.abs(
        (candidate_matrix[:, np.newaxis, :] - reference_matrix[np.newaxis, :, :])
        / scales.reshape(1, 1, -1)
    )
    if metric == "euclidean":
        distances = np.sqrt(np.sum(scaled_diff**2, axis=2))
    elif metric == "manhattan":
        distances = np.sum(scaled_diff, axis=2)
    elif metric == "chebyshev":
        distances = np.max(scaled_diff, axis=2)
    else:
        raise ValueError("metric must be 'euclidean', 'manhattan', or 'chebyshev'")
    return distances.min(axis=1)


def _minimum_scaled_distance_matches(
    candidate_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    metric: str,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_matrix.shape[0] == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=int)
    if reference_matrix.shape[0] == 0:
        return (
            np.full(candidate_matrix.shape[0], np.inf, dtype=float),
            np.full(candidate_matrix.shape[0], -1, dtype=int),
        )
    scaled_diff = np.abs(
        (candidate_matrix[:, np.newaxis, :] - reference_matrix[np.newaxis, :, :])
        / scales.reshape(1, 1, -1)
    )
    if metric == "euclidean":
        pairwise = np.sqrt(np.sum(scaled_diff**2, axis=2))
    elif metric == "manhattan":
        pairwise = np.sum(scaled_diff, axis=2)
    elif metric == "chebyshev":
        pairwise = np.max(scaled_diff, axis=2)
    else:
        raise ValueError("metric must be 'euclidean', 'manhattan', or 'chebyshev'")
    nearest = np.argmin(pairwise, axis=1)
    return pairwise[np.arange(pairwise.shape[0]), nearest], nearest.astype(int)


def _existing_candidate_ids(
    existing_candidates: Any,
    id_column: str | None,
) -> np.ndarray:
    if isinstance(existing_candidates, pd.DataFrame):
        if id_column is not None:
            if id_column not in existing_candidates.columns:
                raise ValueError(f"existing candidate ID column is missing: {id_column!r}")
            values = existing_candidates[id_column]
            if values.isna().any():
                raise ValueError("existing candidate IDs must not contain missing values")
            identifiers = np.asarray([str(value).strip() for value in values], dtype=object)
            if any(not value for value in identifiers):
                raise ValueError("existing candidate IDs must not contain empty values")
            if len(set(identifiers.tolist())) != len(identifiers):
                raise ValueError("existing candidate IDs must be unique")
            return identifiers
        return np.asarray([str(value) for value in existing_candidates.index], dtype=object)
    if id_column is not None:
        raise TypeError("existing_id_column requires existing_candidates to be a dataframe")
    values = np.asarray(existing_candidates)
    row_count = 1 if values.ndim == 1 else values.shape[0]
    return np.asarray([str(index) for index in range(row_count)], dtype=object)


def _candidate_policy_row_keys(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> list[tuple[Any, ...]]:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} are missing: {missing}")

    missing_sentinel = ("__matgpr_missing_key__",)
    return [
        tuple(missing_sentinel if pd.isna(value) else value for value in row)
        for row in data.loc[:, list(columns)].itertuples(index=False, name=None)
    ]
