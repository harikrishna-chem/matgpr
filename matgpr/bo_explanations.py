"""Plot-ready explanation tables for Bayesian-optimization recommendations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .bayesian_optimization_specs import BOOutputConstraintSpec

__all__ = [
    "BOExplanationTables",
    "BOWhatIfSlice",
    "build_bo_explanation_tables",
    "build_bo_what_if_slice",
]

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


@dataclass(frozen=True)
class BOWhatIfSlice:
    """Evaluated one- or two-feature BO slice with uncertainty metadata."""

    rows: pd.DataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        """Return a copy of the plot-ready slice rows."""

        return self.rows.copy()


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


def build_bo_what_if_slice(
    ranked_candidates: pd.DataFrame,
    *,
    x_feature: str,
    y_feature: str | None = None,
    fixed_feature_values: Mapping[str, Any] | None = None,
    top_k: int = 5,
    uncertainty_multiplier: float = 1.96,
    identifier_columns: Sequence[str] | None = None,
    match_relative_tolerance: float = 1e-9,
    match_absolute_tolerance: float = 1e-12,
) -> BOWhatIfSlice:
    """Build a strict one- or two-feature slice from evaluated BO candidates.

    This function does not interpolate, refit, or call a surrogate model. It
    filters already evaluated candidate predictions to the requested fixed
    context. A slice must contain at most one candidate per axis coordinate;
    callers must condition additional varying features when coordinates are
    ambiguous.
    """

    ranked = _validate_ranked_candidates(ranked_candidates)
    identifiers = _resolve_identifier_columns(ranked, identifier_columns)
    x_axis = _resolve_slice_axis(
        ranked,
        x_feature,
        "x_feature",
        identifier_columns=identifiers,
    )
    y_axis = (
        None
        if y_feature is None
        else _resolve_slice_axis(
            ranked,
            y_feature,
            "y_feature",
            identifier_columns=identifiers,
        )
    )
    if y_axis == x_axis:
        raise ValueError("y_feature must differ from x_feature")

    recommendation_count = _positive_int(top_k, "top_k")
    multiplier = _positive_float(uncertainty_multiplier, "uncertainty_multiplier")
    relative_tolerance = _nonnegative_float(
        match_relative_tolerance,
        "match_relative_tolerance",
    )
    absolute_tolerance = _nonnegative_float(
        match_absolute_tolerance,
        "match_absolute_tolerance",
    )
    fixed_values = dict(fixed_feature_values or {})
    _validate_fixed_slice_features(
        ranked,
        fixed_values,
        axis_features=(x_axis,) if y_axis is None else (x_axis, y_axis),
        identifier_columns=identifiers,
    )
    sliced = _filter_slice_context(
        ranked,
        fixed_values,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if sliced.empty:
        raise ValueError("fixed_feature_values do not match any ranked candidates")

    axis_features = (x_axis,) if y_axis is None else (x_axis, y_axis)
    for axis in axis_features:
        if sliced[axis].nunique(dropna=False) < 2:
            raise ValueError(
                f"what-if slice axis {axis!r} must contain at least two "
                "distinct evaluated values after conditioning"
            )
    duplicate_coordinates = sliced.duplicated(
        subset=list(axis_features),
        keep=False,
    )
    if duplicate_coordinates.any():
        varying = _varying_context_columns(
            sliced.loc[duplicate_coordinates],
            excluded_columns=axis_features,
        )
        detail = f"; varying context columns={list(varying)}" if varying else ""
        raise ValueError(
            "what-if slice has multiple candidates at the same axis coordinate; "
            f"provide additional fixed_feature_values{detail}"
        )

    labels = _candidate_labels(sliced, identifiers)
    predicted_mean = sliced["matgpr_predicted_mean"].to_numpy(dtype=float)
    predicted_std = sliced["matgpr_predicted_std"].to_numpy(dtype=float)
    ranks = sliced["matgpr_rank"].to_numpy(dtype=int)
    recommended = ranks <= recommendation_count
    constrained = any(
        str(column).startswith("matgpr_constraint_probability_") for column in sliced.columns
    )
    rows = pd.DataFrame(
        {
            "candidate_or_input_id": labels,
            "rank": ranks,
            "x": sliced[x_axis].to_numpy(dtype=float),
            "y": (
                np.full(len(sliced), np.nan)
                if y_axis is None
                else sliced[y_axis].to_numpy(dtype=float)
            ),
            "x_feature": [x_axis] * len(sliced),
            "y_feature": [y_axis] * len(sliced),
            "predicted_mean": predicted_mean,
            "predicted_std": predicted_std,
            "predicted_lower": predicted_mean - multiplier * predicted_std,
            "predicted_upper": predicted_mean + multiplier * predicted_std,
            "objective_preference_score": sliced["matgpr_preference_score"].to_numpy(dtype=float),
            "objective_component": sliced["matgpr_objective_component"].to_numpy(dtype=float),
            "feasibility_probability": sliced["matgpr_feasibility_probability"].to_numpy(
                dtype=float
            ),
            "acquisition_score": sliced["matgpr_acquisition"].to_numpy(dtype=float),
            "feasibility_status": _feasibility_status(sliced, constrained),
            "constraint_violations": _optional_text_column(
                sliced,
                "matgpr_constraint_violations",
            ),
            "is_recommended": recommended.astype(bool),
            "point_role": np.where(recommended, "recommended", "candidate"),
        }
    )
    sort_columns = ["x", "rank"] if y_axis is None else ["y", "x", "rank"]
    rows = rows.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if y_axis is None:
        rows["y"] = None

    constant_context = _constant_context_values(
        sliced,
        excluded_columns=(*axis_features, *identifiers),
    )
    return BOWhatIfSlice(
        rows=rows,
        metadata={
            "slice_dimension": len(axis_features),
            "x_feature": x_axis,
            "y_feature": y_axis,
            "fixed_feature_values": {
                str(name): _python_scalar(value) for name, value in fixed_values.items()
            },
            "constant_context_values": constant_context,
            "source_candidate_count": len(ranked),
            "matched_candidate_count": len(sliced),
            "recommendation_rank_cutoff": recommendation_count,
            "uncertainty_multiplier": multiplier,
            "interpolation_method": "none",
            "identifier_columns": list(identifiers),
            "match_relative_tolerance": relative_tolerance,
            "match_absolute_tolerance": absolute_tolerance,
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


def _resolve_slice_axis(
    ranked: pd.DataFrame,
    value: str,
    field_name: str,
    *,
    identifier_columns: Sequence[str],
) -> str:
    feature = str(value).strip()
    if not feature:
        raise ValueError(f"{field_name} must not be empty")
    if feature not in ranked:
        raise ValueError(f"{field_name} {feature!r} is missing from ranked_candidates")
    if feature.startswith("matgpr_"):
        raise ValueError(f"{field_name} must reference a candidate input feature")
    if feature in identifier_columns:
        raise ValueError(f"{field_name} must not reference an identifier column")
    if not pd.api.types.is_numeric_dtype(ranked[feature]):
        raise ValueError(f"{field_name} {feature!r} must be numeric")
    values = pd.to_numeric(ranked[feature], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} {feature!r} must contain finite values")
    return feature


def _validate_fixed_slice_features(
    ranked: pd.DataFrame,
    fixed_values: Mapping[str, Any],
    *,
    axis_features: Sequence[str],
    identifier_columns: Sequence[str],
) -> None:
    missing = [str(name) for name in fixed_values if str(name) not in ranked]
    if missing:
        raise ValueError(f"fixed_feature_values missing from ranked_candidates: {missing}")
    invalid = [
        str(name)
        for name in fixed_values
        if str(name).startswith("matgpr_")
        or str(name) in axis_features
        or str(name) in identifier_columns
    ]
    if invalid:
        raise ValueError(
            "fixed_feature_values must contain non-axis candidate input features; "
            f"invalid={invalid}"
        )


def _filter_slice_context(
    ranked: pd.DataFrame,
    fixed_values: Mapping[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> pd.DataFrame:
    selected = np.ones(len(ranked), dtype=bool)
    for raw_name, expected in fixed_values.items():
        name = str(raw_name)
        series = ranked[name]
        if expected is None:
            matches = series.isna().to_numpy(dtype=bool)
        elif pd.api.types.is_numeric_dtype(series) and not isinstance(expected, bool):
            try:
                expected_numeric = float(expected)
            except (TypeError, ValueError) as error:
                raise ValueError(f"fixed_feature_values[{name!r}] must be numeric") from error
            if not np.isfinite(expected_numeric):
                raise ValueError(f"fixed_feature_values[{name!r}] must be finite")
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            matches = np.isfinite(numeric) & np.isclose(
                numeric,
                expected_numeric,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        else:
            matches = series.eq(expected).fillna(False).to_numpy(dtype=bool)
        selected &= matches
    return ranked.loc[selected].reset_index(drop=True)


def _varying_context_columns(
    frame: pd.DataFrame,
    *,
    excluded_columns: Sequence[str],
) -> tuple[str, ...]:
    excluded = set(excluded_columns)
    return tuple(
        str(column)
        for column in frame.columns
        if column not in excluded
        and not str(column).startswith("matgpr_")
        and column not in _AUTO_IDENTIFIER_COLUMNS
        and frame[column].nunique(dropna=False) > 1
    )


def _constant_context_values(
    frame: pd.DataFrame,
    *,
    excluded_columns: Sequence[str],
) -> dict[str, Any]:
    excluded = set(excluded_columns)
    context: dict[str, Any] = {}
    for column in frame.columns:
        if column in excluded or str(column).startswith("matgpr_"):
            continue
        if frame[column].nunique(dropna=False) != 1:
            continue
        value = frame[column].iloc[0]
        context[str(column)] = None if pd.isna(value) else _python_scalar(value)
    return context


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _positive_int(value: Any, name: str) -> int:
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number


def _slug(value: str) -> str:
    normalized = [character.lower() if character.isalnum() else "_" for character in value]
    return "_".join(part for part in "".join(normalized).split("_") if part) or "target"


def _feature_y(method: str) -> str | None:
    if method == "pca":
        return "PC2"
    return None
