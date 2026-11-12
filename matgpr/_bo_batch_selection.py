from __future__ import annotations

import numpy as np
import pandas as pd

__all__: list[str] = []

_OUTPUT_COLUMNS = (
    "matgpr_candidate_index",
    "matgpr_rank",
    "matgpr_feasible",
    "matgpr_constraint_violations",
    "matgpr_batch_selected",
    "matgpr_batch_order",
    "matgpr_batch_score",
    "matgpr_diversity_distance",
    "matgpr_predicted_mean",
    "matgpr_predicted_std",
    "matgpr_preference_score",
    "matgpr_objective_component",
    "matgpr_feasibility_probability",
    "matgpr_acquisition_base",
    "matgpr_acquisition",
    "matgpr_acquisition_rank",
    "matgpr_eligible_rank",
    "matgpr_eligible_for_recommendation",
    "matgpr_selected_for_batch",
)


def _as_finite_numeric_vector(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if numeric.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return numeric


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    value_range = float(np.max(values) - np.min(values))
    if value_range == 0.0:
        return np.ones_like(values, dtype=float)
    return (values - np.min(values)) / value_range


def _top_score_positions(effective_scores: np.ndarray, top_k: int) -> np.ndarray:
    positions = np.arange(effective_scores.shape[0])
    order = np.lexsort((positions, -effective_scores))
    return order[: min(top_k, effective_scores.shape[0])]


def _rank_descending(values: pd.Series) -> np.ndarray:
    scores = _as_finite_numeric_vector(values, name=str(values.name or "scores"))
    positions = np.arange(scores.shape[0])
    order = np.lexsort((positions, -scores))
    ranks = np.empty(scores.shape[0], dtype=int)
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    return ranks


def _greedy_diverse_positions(
    candidates: pd.DataFrame,
    *,
    top_k: int,
    score_column: str,
    feature_columns: list[str] | tuple[str, ...] | None,
    normalized_scores: np.ndarray,
    diversity_weight: float,
    min_distance: float | None,
    standardize_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_matrix, resolved_features = _candidate_diversity_matrix(
        candidates,
        score_column=score_column,
        feature_columns=feature_columns,
        standardize_features=standardize_features,
    )
    if feature_matrix.shape[0] != normalized_scores.shape[0]:
        raise ValueError("feature matrix rows must match candidates")
    if len(resolved_features) == 0:
        raise ValueError("at least one diversity feature is required")

    selected: list[int] = []
    selection_scores: list[float] = []
    diversity_distances: list[float] = []
    remaining = np.ones(feature_matrix.shape[0], dtype=bool)
    min_distances = np.full(feature_matrix.shape[0], np.inf, dtype=float)

    while len(selected) < top_k and np.any(remaining):
        eligible = remaining.copy()
        if selected and min_distance is not None:
            eligible &= min_distances >= min_distance
        if not np.any(eligible):
            break

        if selected:
            finite_distances = np.where(np.isfinite(min_distances), min_distances, 0.0)
            distance_scale = float(np.max(finite_distances[eligible]))
            if distance_scale > 0.0:
                normalized_distances = finite_distances / distance_scale
            else:
                normalized_distances = np.zeros_like(finite_distances)
            candidate_scores = normalized_scores + diversity_weight * normalized_distances
        else:
            candidate_scores = normalized_scores.copy()

        candidate_scores[~eligible] = -np.inf
        chosen = _argmax_stable(candidate_scores)
        if chosen is None:
            break

        selected.append(chosen)
        selection_scores.append(float(candidate_scores[chosen]))
        if len(selected) == 1:
            diversity_distances.append(np.nan)
        else:
            diversity_distances.append(float(min_distances[chosen]))

        remaining[chosen] = False
        distances = np.linalg.norm(feature_matrix - feature_matrix[chosen], axis=1)
        min_distances = np.minimum(min_distances, distances)

    return (
        np.asarray(selected, dtype=int),
        np.asarray(selection_scores, dtype=float),
        np.asarray(diversity_distances, dtype=float),
    )


def _candidate_diversity_matrix(
    candidates: pd.DataFrame,
    *,
    score_column: str,
    feature_columns: list[str] | tuple[str, ...] | None,
    standardize_features: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    resolved_features = _resolve_diversity_feature_columns(
        candidates,
        score_column=score_column,
        feature_columns=feature_columns,
    )
    feature_frame = candidates.loc[:, list(resolved_features)]
    non_numeric = [
        str(column)
        for column in resolved_features
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if non_numeric:
        raise ValueError(f"diversity feature columns must be numeric: {non_numeric}")
    matrix = feature_frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("diversity feature columns contain NaN or infinite values")
    nonconstant = np.ptp(matrix, axis=0) > 0.0
    if not np.any(nonconstant):
        raise ValueError("diversity requires at least one nonconstant numeric feature")
    if not np.all(nonconstant):
        matrix = matrix[:, nonconstant]
        resolved_features = tuple(
            column for column, keep in zip(resolved_features, nonconstant, strict=True) if keep
        )
    if standardize_features and matrix.shape[0] > 0:
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales == 0.0] = 1.0
        matrix = (matrix - means) / scales
    return matrix, resolved_features


def _resolve_diversity_feature_columns(
    candidates: pd.DataFrame,
    *,
    score_column: str,
    feature_columns: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if feature_columns is not None:
        resolved = tuple(str(column) for column in feature_columns)
        missing = [column for column in resolved if column not in candidates.columns]
        if missing:
            raise ValueError(f"diversity feature columns are missing: {missing}")
        if len(resolved) == 0:
            raise ValueError("feature_columns must contain at least one column")
        return resolved

    excluded = set(_OUTPUT_COLUMNS)
    excluded.add(score_column)
    inferred = tuple(
        str(column)
        for column in candidates.columns
        if str(column) not in excluded and pd.api.types.is_numeric_dtype(candidates[column])
    )
    if not inferred:
        raise ValueError(
            "No numeric diversity feature columns could be inferred. Provide "
            "feature_columns explicitly or include numeric candidate descriptors."
        )
    return inferred


def _argmax_stable(values: np.ndarray) -> int | None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    positions = np.arange(values.shape[0])
    order = np.lexsort((positions, -values))
    return int(order[0])
