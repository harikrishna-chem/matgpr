from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ObjectiveSpec",
    "pareto_front_mask",
    "rank_multi_objective_candidates",
    "scalarize_objectives",
    "select_pareto_front",
]


@dataclass(frozen=True)
class ObjectiveSpec:
    """Definition of one objective in a finite-pool multi-objective workflow.

    Parameters
    ----------
    name
        Short objective label used in utility column names.
    column
        Numeric dataframe column containing objective values.
    goal
        Either `"maximize"` or `"minimize"`.
    weight
        Non-negative scalarization weight. At least one objective supplied to a
        scalarization workflow must have a positive weight.
    """

    name: str
    column: str
    goal: str = "maximize"
    weight: float = 1.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        column = str(self.column).strip()
        goal = str(self.goal).strip().lower().replace("-", "_")
        weight = float(self.weight)

        if not name:
            raise ValueError("ObjectiveSpec.name must be non-empty")
        if not column:
            raise ValueError("ObjectiveSpec.column must be non-empty")
        if goal not in {"maximize", "minimize"}:
            raise ValueError("ObjectiveSpec.goal must be 'maximize' or 'minimize'")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("ObjectiveSpec.weight must be finite and non-negative")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "weight", weight)


def pareto_front_mask(
    data: pd.DataFrame,
    objectives: ObjectiveSpec | Sequence[ObjectiveSpec],
) -> pd.Series:
    """Return a boolean mask for nondominated candidate rows.

    Objective directions are taken from each `ObjectiveSpec`, so minimize-type
    objectives such as cost, toxicity, synthesis difficulty, or degradation rate
    are handled directly.
    """
    objective_tuple = _as_objective_tuple(objectives)
    values = _objective_values(data, objective_tuple)
    oriented = _to_maximization_space(values, objective_tuple)
    mask = _pareto_mask(oriented)
    return pd.Series(mask, index=data.index, name="matgpr_pareto_front")


def select_pareto_front(
    data: pd.DataFrame,
    objectives: ObjectiveSpec | Sequence[ObjectiveSpec],
    *,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Return nondominated rows from a finite candidate table."""
    mask = pareto_front_mask(data, objectives)
    front = data.loc[mask].copy()
    if reset_index:
        return front.reset_index(drop=True)
    return front


def scalarize_objectives(
    data: pd.DataFrame,
    objectives: ObjectiveSpec | Sequence[ObjectiveSpec],
    *,
    normalize: bool = True,
) -> pd.Series:
    """Compute a weighted scalar utility score for multiple objectives.

    Each objective is first converted so larger utility is better. With
    `normalize=True`, every objective is min-max scaled to `[0, 1]` before the
    weighted average is computed. This is the recommended default when columns
    have different physical units.
    """
    objective_tuple = _as_objective_tuple(objectives)
    utilities = _objective_utilities(data, objective_tuple, normalize=normalize)
    weights = _objective_weights(objective_tuple)
    score = utilities @ weights
    return pd.Series(score, index=data.index, name="matgpr_multi_objective_score")


def rank_multi_objective_candidates(
    data: pd.DataFrame,
    objectives: ObjectiveSpec | Sequence[ObjectiveSpec],
    *,
    top_k: int | None = None,
    normalize: bool = True,
    score_column: str = "matgpr_multi_objective_score",
    rank_column: str = "matgpr_multi_objective_rank",
    pareto_column: str = "matgpr_pareto_front",
    include_objective_utilities: bool = True,
    utility_prefix: str = "matgpr_objective_",
) -> pd.DataFrame:
    """Rank finite candidates by weighted multi-objective utility.

    The result keeps original candidate columns, adds the scalarized score,
    marks Pareto-front candidates, and sorts from most to least desirable.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided")
    objective_tuple = _as_objective_tuple(objectives)
    score_column = str(score_column).strip()
    rank_column = str(rank_column).strip()
    pareto_column = str(pareto_column).strip()
    utility_prefix = str(utility_prefix).strip()
    _validate_output_column(score_column, objective_tuple, name="score_column")
    _validate_output_column(rank_column, objective_tuple, name="rank_column")
    _validate_output_column(pareto_column, objective_tuple, name="pareto_column")
    _validate_distinct_output_columns(score_column, rank_column, pareto_column)
    if not utility_prefix:
        raise ValueError("utility_prefix must be non-empty")

    result = data.copy()
    utilities = _objective_utilities(result, objective_tuple, normalize=normalize)
    weights = _objective_weights(objective_tuple)
    result[score_column] = utilities @ weights
    result[pareto_column] = pareto_front_mask(result, objective_tuple).to_numpy(dtype=bool)

    if include_objective_utilities:
        for index, objective in enumerate(objective_tuple):
            column = f"{utility_prefix}{_slugify(objective.name)}_utility"
            _validate_output_column(column, objective_tuple, name="utility_column")
            _validate_distinct_output_columns(
                column,
                score_column,
                rank_column,
                pareto_column,
            )
            result[column] = utilities[:, index]

    result["_matgpr_original_order"] = np.arange(result.shape[0], dtype=int)
    result = result.sort_values(
        [score_column, pareto_column, "_matgpr_original_order"],
        ascending=[False, False, True],
    ).drop(columns="_matgpr_original_order")
    result.insert(0, rank_column, np.arange(1, result.shape[0] + 1))
    result = result.reset_index(drop=True)

    if top_k is not None:
        return result.head(top_k).reset_index(drop=True)
    return result


def _as_objective_tuple(
    objectives: ObjectiveSpec | Sequence[ObjectiveSpec],
) -> tuple[ObjectiveSpec, ...]:
    if isinstance(objectives, ObjectiveSpec):
        objective_tuple = (objectives,)
    else:
        try:
            objective_tuple = tuple(objectives)
        except TypeError as exc:
            raise TypeError(
                "objectives must be an ObjectiveSpec or an iterable of ObjectiveSpec objects"
            ) from exc

    if not objective_tuple:
        raise ValueError("objectives must contain at least one ObjectiveSpec")
    invalid = [
        type(objective).__name__
        for objective in objective_tuple
        if not isinstance(objective, ObjectiveSpec)
    ]
    if invalid:
        raise TypeError(f"objectives must contain only ObjectiveSpec objects; got {invalid}")
    columns = [objective.column for objective in objective_tuple]
    if len(set(columns)) != len(columns):
        raise ValueError("objectives must not contain duplicate columns")
    names = [objective.name for objective in objective_tuple]
    if len(set(names)) != len(names):
        raise ValueError("objectives must not contain duplicate names")
    slugs = [_slugify(name) for name in names]
    if len(set(slugs)) != len(slugs):
        raise ValueError("objectives must not produce duplicate utility column names")
    return objective_tuple


def _objective_values(
    data: pd.DataFrame,
    objectives: tuple[ObjectiveSpec, ...],
) -> np.ndarray:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = [objective.column for objective in objectives if objective.column not in data.columns]
    if missing:
        raise ValueError(f"Objective columns are missing: {missing}")

    values = data.loc[:, [objective.column for objective in objectives]]
    non_numeric = [
        objective.column
        for objective in objectives
        if not pd.api.types.is_numeric_dtype(values[objective.column])
    ]
    if non_numeric:
        raise ValueError(f"Objective columns must be numeric: {non_numeric}")
    matrix = values.to_numpy(dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Objective columns contain NaN or infinite values")
    return matrix


def _to_maximization_space(
    values: np.ndarray,
    objectives: tuple[ObjectiveSpec, ...],
) -> np.ndarray:
    oriented = values.copy()
    for index, objective in enumerate(objectives):
        if objective.goal == "minimize":
            oriented[:, index] = -oriented[:, index]
    return oriented


def _objective_utilities(
    data: pd.DataFrame,
    objectives: tuple[ObjectiveSpec, ...],
    *,
    normalize: bool,
) -> np.ndarray:
    values = _objective_values(data, objectives)
    oriented = _to_maximization_space(values, objectives)
    if not normalize or oriented.shape[0] == 0:
        return oriented

    utilities = np.empty_like(oriented, dtype=float)
    for index in range(oriented.shape[1]):
        column = oriented[:, index]
        value_range = float(np.max(column) - np.min(column))
        if value_range == 0.0:
            utilities[:, index] = 1.0
        else:
            utilities[:, index] = (column - np.min(column)) / value_range
    return utilities


def _objective_weights(objectives: tuple[ObjectiveSpec, ...]) -> np.ndarray:
    weights = np.asarray([objective.weight for objective in objectives], dtype=float)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("At least one objective weight must be positive")
    return weights / total_weight


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    n_rows = values.shape[0]
    mask = np.ones(n_rows, dtype=bool)
    for index in range(n_rows):
        if not mask[index]:
            continue
        dominated = np.all(values >= values[index], axis=1) & np.any(
            values > values[index],
            axis=1,
        )
        if np.any(dominated):
            mask[index] = False
    return mask


def _validate_output_column(
    column: str,
    objectives: tuple[ObjectiveSpec, ...],
    *,
    name: str,
) -> None:
    column = str(column).strip()
    if not column:
        raise ValueError(f"{name} must be non-empty")
    objective_columns = {objective.column for objective in objectives}
    if column in objective_columns:
        raise ValueError(f"{name} must not overwrite objective column {column!r}")


def _validate_distinct_output_columns(*columns: str) -> None:
    if len(set(columns)) != len(columns):
        raise ValueError("Output columns must be distinct")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "objective"
