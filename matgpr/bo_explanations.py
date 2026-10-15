"""Plot-ready explanation tables for Bayesian-optimization recommendations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .bayesian_optimization_specs import BOOutputConstraintSpec

__all__ = ["BOExplanationTables", "build_bo_explanation_tables"]

_REQUIRED_RANKING_COLUMNS = (
    "matgpr_rank",
    "matgpr_candidate_index",
    "matgpr_predicted_mean",
    "matgpr_predicted_std",
    "matgpr_preference_score",
    "matgpr_objective_component",
    "matgpr_feasibility_probability",
    "matgpr_acquisition",
)
_AUTO_IDENTIFIER_COLUMNS = (
    "candidate_id",
    "row_id",
    "material_id",
    "input_id",
    "sample_id",
)


@dataclass(frozen=True)
class BOExplanationTables:
    """App-neutral tables explaining BO recommendation ranking."""

    recommendation_table: pd.DataFrame
    acquisition_table: pd.DataFrame
    constraint_table: pd.DataFrame
    feature_space_table: pd.DataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def recommendation_frame(self) -> pd.DataFrame:
        """Return recommendation summary rows."""

        return self.recommendation_table.copy()

    def acquisition_frame(self) -> pd.DataFrame:
        """Return acquisition decomposition rows."""

        return self.acquisition_table.copy()

    def constraint_frame(self) -> pd.DataFrame:
        """Return one feasibility row per candidate and output constraint."""

        return self.constraint_table.copy()

    def feature_space_frame(self) -> pd.DataFrame:
        """Return training/reference and candidate coordinates for plotting."""

        return self.feature_space_table.copy()


def build_bo_explanation_tables(
    ranked_candidates: pd.DataFrame,
    *,
    top_k: int,
    output_constraints: Sequence[BOOutputConstraintSpec | Mapping[str, Any]] = (),
    reference_features: pd.DataFrame | None = None,
    feature_columns: Sequence[str] | None = None,
    identifier_columns: Sequence[str] | None = None,
) -> BOExplanationTables:
    """Build plot-ready tables that explain finite-pool BO recommendations.

    The returned tables contain no plotting-library objects. One- and
    two-feature spaces retain their original axes. Higher-dimensional feature
    spaces are standardized and projected onto two principal components fitted
    jointly to reference and candidate points.
    """

    ranked = _validate_ranked_candidates(ranked_candidates)
    recommendation_count = _positive_int(top_k, "top_k")
    recommendation_count = min(recommendation_count, len(ranked))
    constraints = tuple(_coerce_output_constraint(item) for item in output_constraints)
    identifiers = _resolve_identifier_columns(ranked, identifier_columns)
    labels = _candidate_labels(ranked, identifiers)
    selected = np.arange(len(ranked)) < recommendation_count

    recommendation_table = _recommendation_table(
        ranked,
        labels,
        selected=selected,
        constrained=bool(constraints),
    )
    acquisition_table = _acquisition_table(ranked, labels, selected=selected)
    constraint_table = _constraint_table(
        ranked,
        labels,
        constraints,
        selected=selected,
    )
    feature_space_table, feature_metadata = _feature_space_table(
        ranked,
        labels,
        selected=selected,
        reference_features=reference_features,
        feature_columns=feature_columns,
    )
    return BOExplanationTables(
        recommendation_table=recommendation_table,
        acquisition_table=acquisition_table,
        constraint_table=constraint_table,
        feature_space_table=feature_space_table,
        metadata={
            "candidate_count": len(ranked),
            "recommendation_count": recommendation_count,
            "constraint_count": len(constraints),
            "identifier_columns": list(identifiers),
            **feature_metadata,
        },
    )


def _recommendation_table(
    ranked: pd.DataFrame,
    labels: pd.Series,
    *,
    selected: np.ndarray,
    constrained: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "rank": ranked["matgpr_rank"].to_numpy(dtype=int),
            "candidate_or_input_id": labels,
            "predicted_mean": ranked["matgpr_predicted_mean"].to_numpy(dtype=float),
            "predicted_std": ranked["matgpr_predicted_std"].to_numpy(dtype=float),
            "objective_preference_score": ranked["matgpr_preference_score"].to_numpy(dtype=float),
            "objective_component": ranked["matgpr_objective_component"].to_numpy(dtype=float),
            "feasibility_probability": ranked["matgpr_feasibility_probability"].to_numpy(
                dtype=float
            ),
            "acquisition_score": ranked["matgpr_acquisition"].to_numpy(dtype=float),
            "feasibility_status": _feasibility_status(ranked, constrained),
            "constraint_violations": _optional_text_column(
                ranked,
                "matgpr_constraint_violations",
            ),
        }
    )
    return frame.loc[selected].reset_index(drop=True)


def _acquisition_table(
    ranked: pd.DataFrame,
    labels: pd.Series,
    *,
    selected: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": ranked["matgpr_rank"].to_numpy(dtype=int),
            "candidate_or_input_id": labels,
            "predicted_mean": ranked["matgpr_predicted_mean"].to_numpy(dtype=float),
            "predicted_std": ranked["matgpr_predicted_std"].to_numpy(dtype=float),
            "objective_preference_score": ranked["matgpr_preference_score"].to_numpy(dtype=float),
            "objective_component": ranked["matgpr_objective_component"].to_numpy(dtype=float),
            "feasibility_probability": ranked["matgpr_feasibility_probability"].to_numpy(
                dtype=float
            ),
            "acquisition_score": ranked["matgpr_acquisition"].to_numpy(dtype=float),
            "is_recommended": selected.astype(bool),
        }
    )


def _constraint_table(
    ranked: pd.DataFrame,
    labels: pd.Series,
    constraints: Sequence[BOOutputConstraintSpec],
    *,
    selected: np.ndarray,
) -> pd.DataFrame:
    columns = (
        "rank",
        "candidate_or_input_id",
        "constraint_name",
        "predicted_value",
        "predicted_std",
        "lower_bound",
        "upper_bound",
        "probability_threshold",
        "feasibility_probability",
        "is_feasible",
        "is_recommended",
    )
    rows: list[dict[str, Any]] = []
    target_occurrences: dict[str, int] = {}
    for constraint in constraints:
        occurrence = target_occurrences.get(constraint.target_name, 0) + 1
        target_occurrences[constraint.target_name] = occurrence
        probability_column = _constraint_probability_column(
            ranked,
            constraint,
            occurrence=occurrence,
        )
        probabilities = ranked[probability_column].to_numpy(dtype=float)
        threshold = (
            0.5
            if constraint.probability_threshold is None
            else float(constraint.probability_threshold)
        )
        for row_index in range(len(ranked)):
            rows.append(
                {
                    "rank": int(ranked["matgpr_rank"].iloc[row_index]),
                    "candidate_or_input_id": labels.iloc[row_index],
                    "constraint_name": constraint.target_name,
                    "predicted_value": float(ranked["matgpr_predicted_mean"].iloc[row_index]),
                    "predicted_std": float(ranked["matgpr_predicted_std"].iloc[row_index]),
                    "lower_bound": constraint.lower_bound,
                    "upper_bound": constraint.upper_bound,
                    "probability_threshold": threshold,
                    "feasibility_probability": float(probabilities[row_index]),
                    "is_feasible": bool(probabilities[row_index] >= threshold),
                    "is_recommended": bool(selected[row_index]),
                }
            )
    return pd.DataFrame(rows, columns=list(columns))


def _feature_space_table(
    ranked: pd.DataFrame,
    labels: pd.Series,
    *,
    selected: np.ndarray,
    reference_features: pd.DataFrame | None,
    feature_columns: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = _resolve_feature_columns(ranked, reference_features, feature_columns)
    columns = (
        "x",
        "y",
        "point_role",
        "rank",
        "candidate_or_input_id",
        "acquisition_score",
        "embedding_method",
        "feature_x",
        "feature_y",
    )
    if not features:
        return pd.DataFrame(columns=list(columns)), {
            "embedding_method": "unavailable",
            "feature_columns": [],
            "embedding_explained_variance_ratio": [],
            "reference_point_count": 0,
        }

    candidate_matrix = ranked.loc[:, list(features)].to_numpy(dtype=float)
    if not np.all(np.isfinite(candidate_matrix)):
        raise ValueError("ranked candidate features contain NaN or infinite values")
    reference_matrix = _reference_matrix(reference_features, features)
    combined = (
        candidate_matrix
        if reference_matrix is None
        else np.vstack((reference_matrix, candidate_matrix))
    )
    coordinates, method, variance_ratio = _embedding_coordinates(combined)
    reference_count = 0 if reference_matrix is None else reference_matrix.shape[0]
    reference_coordinates = coordinates[:reference_count]
    candidate_coordinates = coordinates[reference_count:]

    rows: list[dict[str, Any]] = []
    for row_index, coordinate in enumerate(reference_coordinates):
        rows.append(
            {
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "point_role": "reference",
                "rank": None,
                "candidate_or_input_id": f"reference_{row_index + 1}",
                "acquisition_score": None,
                "embedding_method": method,
                "feature_x": features[0] if len(features) <= 2 else "PC1",
                "feature_y": features[1] if len(features) == 2 else _feature_y(method),
            }
        )
    for row_index, coordinate in enumerate(candidate_coordinates):
        rows.append(
            {
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "point_role": "recommended" if selected[row_index] else "candidate",
                "rank": int(ranked["matgpr_rank"].iloc[row_index]),
                "candidate_or_input_id": labels.iloc[row_index],
                "acquisition_score": float(ranked["matgpr_acquisition"].iloc[row_index]),
                "embedding_method": method,
                "feature_x": features[0] if len(features) <= 2 else "PC1",
                "feature_y": features[1] if len(features) == 2 else _feature_y(method),
            }
        )
    table = pd.DataFrame(rows, columns=list(columns))
    if not table.empty:
        table["rank"] = table["rank"].astype("Int64")
    return table, {
        "embedding_method": method,
        "feature_columns": list(features),
        "embedding_explained_variance_ratio": variance_ratio,
        "reference_point_count": reference_count,
    }


def _embedding_coordinates(
    matrix: np.ndarray,
) -> tuple[np.ndarray, str, list[float]]:
    if matrix.shape[1] == 1:
        return (
            np.column_stack((matrix[:, 0], np.zeros(matrix.shape[0]))),
            "single_feature",
            [1.0, 0.0],
        )
    if matrix.shape[1] == 2:
        return matrix.astype(float), "selected_features", []

    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    standardized = (matrix - means) / scales
    _, singular_values, right_vectors = np.linalg.svd(
        standardized,
        full_matrices=False,
    )
    component_count = min(2, right_vectors.shape[0])
    components = right_vectors[:component_count].copy()
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            component *= -1.0
    coordinates = standardized @ components.T
    if component_count == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(matrix.shape[0])))
    total_variance = float(np.sum(singular_values**2))
    variance_ratio = (
        [0.0] * component_count
        if total_variance <= 0.0
        else [float(value**2 / total_variance) for value in singular_values[:component_count]]
    )
    if component_count == 1:
        variance_ratio.append(0.0)
    return coordinates.astype(float), "pca", variance_ratio


def _validate_ranked_candidates(value: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("ranked_candidates must be a pandas DataFrame")
    if value.empty:
        raise ValueError("ranked_candidates must contain at least one row")
    missing = [column for column in _REQUIRED_RANKING_COLUMNS if column not in value]
    if missing:
        raise ValueError(f"ranked_candidates missing required columns: {missing}")
    ranked = value.reset_index(drop=True).copy()
    ranks = pd.to_numeric(ranked["matgpr_rank"], errors="coerce")
    if (
        ranks.isna().any()
        or not np.isfinite(ranks.to_numpy(dtype=float)).all()
        or not np.equal(ranks, np.floor(ranks)).all()
        or (ranks < 1).any()
    ):
        raise ValueError("matgpr_rank must contain positive finite integers")
    if ranks.duplicated().any():
        raise ValueError("matgpr_rank values must be unique")
    for column in _REQUIRED_RANKING_COLUMNS[1:]:
        numeric = pd.to_numeric(ranked[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"ranked_candidates column {column!r} must be finite numeric")
    if (pd.to_numeric(ranked["matgpr_predicted_std"]) < 0.0).any():
        raise ValueError("matgpr_predicted_std must be non-negative")
    feasibility = pd.to_numeric(ranked["matgpr_feasibility_probability"])
    if ((feasibility < 0.0) | (feasibility > 1.0)).any():
        raise ValueError("matgpr_feasibility_probability must be in [0, 1]")
    ranked["matgpr_rank"] = ranks.astype(int)
    return ranked.sort_values("matgpr_rank").reset_index(drop=True)


def _resolve_identifier_columns(
    ranked: pd.DataFrame,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is None:
        return tuple(column for column in _AUTO_IDENTIFIER_COLUMNS if column in ranked)
    columns = tuple(str(column).strip() for column in requested if str(column).strip())
    missing = [column for column in columns if column not in ranked]
    if missing:
        raise ValueError(f"identifier_columns missing from ranked_candidates: {missing}")
    return columns


def _candidate_labels(ranked: pd.DataFrame, identifiers: Sequence[str]) -> pd.Series:
    if identifiers:
        return ranked.loc[:, list(identifiers)].astype(str).agg(" | ".join, axis=1)
    return pd.Series(
        [f"candidate_{int(index)}" for index in ranked["matgpr_candidate_index"]],
        dtype=object,
    )


def _resolve_feature_columns(
    ranked: pd.DataFrame,
    reference_features: pd.DataFrame | None,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is None:
        columns = tuple(
            column
            for column in ranked.columns
            if not str(column).startswith("matgpr_")
            and pd.api.types.is_numeric_dtype(ranked[column])
        )
    else:
        columns = tuple(str(column).strip() for column in requested if str(column).strip())
    missing = [column for column in columns if column not in ranked]
    if missing:
        raise ValueError(f"feature_columns missing from ranked_candidates: {missing}")
    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(ranked[column])
    ]
    if non_numeric:
        raise ValueError(f"feature_columns must be numeric: {non_numeric}")
    if reference_features is not None:
        if not isinstance(reference_features, pd.DataFrame):
            raise TypeError("reference_features must be a pandas DataFrame")
        reference_missing = [column for column in columns if column not in reference_features]
        if reference_missing:
            raise ValueError(f"reference_features missing feature columns: {reference_missing}")
    return columns


def _reference_matrix(
    reference_features: pd.DataFrame | None,
    feature_columns: Sequence[str],
) -> np.ndarray | None:
    if reference_features is None or reference_features.empty:
        return None
    matrix = reference_features.loc[:, list(feature_columns)].to_numpy(dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("reference_features contains NaN or infinite values")
    return matrix


def _feasibility_status(ranked: pd.DataFrame, constrained: bool) -> list[str]:
    if not constrained:
        return ["not_constrained"] * len(ranked)
    if "matgpr_feasible" not in ranked:
        return ["unknown"] * len(ranked)
    return ["feasible" if bool(value) else "infeasible" for value in ranked["matgpr_feasible"]]


def _optional_text_column(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return [""] * len(frame)
    return ["" if pd.isna(value) else str(value) for value in frame[column]]


def _constraint_probability_column(
    ranked: pd.DataFrame,
    constraint: BOOutputConstraintSpec,
    *,
    occurrence: int,
) -> str:
    base = "matgpr_constraint_probability_" + _slug(constraint.target_name)
    column = base if occurrence == 1 else f"{base}_{occurrence}"
    if column not in ranked:
        raise ValueError(
            "ranked_candidates missing feasibility probabilities for constraint "
            f"{constraint.target_name!r}"
        )
    return column


def _coerce_output_constraint(
    value: BOOutputConstraintSpec | Mapping[str, Any],
) -> BOOutputConstraintSpec:
    if isinstance(value, BOOutputConstraintSpec):
        return value
    if isinstance(value, Mapping):
        return BOOutputConstraintSpec(**dict(value))
    raise TypeError("output_constraints must contain BOOutputConstraintSpec or mappings")


def _positive_int(value: Any, name: str) -> int:
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer


def _slug(value: str) -> str:
    normalized = [character.lower() if character.isalnum() else "_" for character in value]
    return "_".join(part for part in "".join(normalized).split("_") if part) or "target"


def _feature_y(method: str) -> str | None:
    if method == "pca":
        return "PC2"
    return None
