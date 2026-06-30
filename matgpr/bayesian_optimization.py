from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .multi_objective import ObjectiveSpec, pareto_front_mask
from .optional_dependencies import require_optional_dependency

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
    "matgpr_acquisition",
)

__all__ = [
    "BayesianOptimizationResult",
    "BoTorchSurrogate",
    "CandidateConstraint",
    "CandidateDuplicatePolicy",
    "CandidateTrustRegion",
    "MultiObjectiveBayesianOptimizationResult",
    "MultiObjectiveBoTorchSurrogate",
    "apply_candidate_constraints",
    "apply_candidate_duplicate_policy",
    "apply_candidate_trust_region",
    "fit_botorch_surrogate",
    "fit_multi_objective_botorch_surrogate",
    "observation_noise_variance",
    "rank_discrete_candidates",
    "rank_multi_objective_discrete_candidates",
    "select_diverse_batch",
    "select_sequential_multi_objective_batch",
    "suggest_multi_objective_next_experiments",
    "suggest_next_experiments",
]


@dataclass(frozen=True)
class BoTorchSurrogate:
    """Fitted BoTorch surrogate model and metadata for candidate ranking.

    Parameters
    ----------
    model
        Fitted BoTorch model. The first implementation uses
        `botorch.models.SingleTaskGP`.
    train_X
        Training features as a torch tensor.
    train_y
        Observed targets in the original target direction.
    objective_y
        Training targets in BoTorch maximization space. For minimization
        problems this is `-train_y`.
    maximize
        Whether larger original target values are preferred.
    best_observed_objective
        Best observed target in BoTorch maximization space.
    feature_names
        Feature names when `X_train` was provided as a pandas dataframe.
    noise_variance
        Known observation-noise variance passed to BoTorch, if provided.
    """

    model: Any
    train_X: Any
    train_y: Any
    objective_y: Any
    maximize: bool
    best_observed_objective: float
    feature_names: tuple[str, ...] | None = None
    noise_variance: Any | None = None


@dataclass(frozen=True)
class BayesianOptimizationResult:
    """Result from finite-pool Bayesian optimization.

    Parameters
    ----------
    recommendations
        Top-ranked candidates to evaluate next.
    ranked_candidates
        Full candidate table sorted by acquisition value.
    surrogate
        Fitted BoTorch surrogate and associated metadata.
    acquisition_function
        Acquisition function used for ranking.
    maximize
        Whether the original target was maximized.
    top_k
        Number of recommendations requested.
    """

    recommendations: pd.DataFrame
    ranked_candidates: pd.DataFrame
    surrogate: BoTorchSurrogate
    acquisition_function: str
    maximize: bool
    top_k: int


@dataclass(frozen=True)
class MultiObjectiveBoTorchSurrogate:
    """Fitted independent BoTorch surrogates for multi-objective BO.

    Each objective is modeled by an independent `SingleTaskGP` and combined
    with BoTorch's `ModelListGP`. Minimize-type objectives are negated
    internally so all acquisition functions operate in maximization space.

    Parameters
    ----------
    model
        Fitted BoTorch `ModelListGP`.
    train_X
        Training features as a torch tensor.
    train_y
        Observed targets in original objective directions and units.
    objective_y
        Targets transformed to BoTorch maximization space.
    objective_names
        Stable objective names used in output column labels.
    objective_directions
        One direction per objective: `"maximize"` or `"minimize"`.
    objective_signs
        Numeric signs used to convert original targets to maximization space.
    reference_point
        Hypervolume reference point in BoTorch maximization space.
    reference_point_original
        Same reference point in original objective directions and units.
    feature_names
        Feature names when `X_train` was provided as a pandas dataframe.
    noise_variance
        Known observation-noise variance matrix passed to BoTorch, if provided.
    """

    model: Any
    train_X: Any
    train_y: Any
    objective_y: Any
    objective_names: tuple[str, ...]
    objective_directions: tuple[str, ...]
    objective_signs: Any
    reference_point: Any
    reference_point_original: Any
    feature_names: tuple[str, ...] | None = None
    noise_variance: Any | None = None


@dataclass(frozen=True)
class MultiObjectiveBayesianOptimizationResult:
    """Result from finite-pool multi-objective Bayesian optimization.

    Parameters
    ----------
    recommendations
        Top-ranked candidates to evaluate next.
    ranked_candidates
        Full candidate table sorted by hypervolume-improvement acquisition.
    surrogate
        Fitted multi-objective BoTorch surrogate and associated metadata.
    acquisition_function
        Multi-objective acquisition function used for ranking.
    objective_names
        Objective names used for prediction columns.
    objective_directions
        Optimization direction for each objective.
    top_k
        Number of recommendations requested.
    """

    recommendations: pd.DataFrame
    ranked_candidates: pd.DataFrame
    surrogate: MultiObjectiveBoTorchSurrogate
    acquisition_function: str
    objective_names: tuple[str, ...]
    objective_directions: tuple[str, ...]
    top_k: int


@dataclass(frozen=True)
class CandidateTrustRegion:
    """Descriptor-space trust region for finite-pool BO candidates.

    Parameters
    ----------
    centers
        One or more center points in the same feature space as the candidate
        features. This is commonly a dataframe or array of measured features,
        the current best material, or a manually chosen local search center.
    radius
        Maximum allowed distance from the nearest center.
    metric
        Distance metric: `"euclidean"`, `"manhattan"`, or `"chebyshev"`.
    feature_columns
        Optional feature columns to use when `centers` or candidate features
        are dataframes.
    feature_scales
        Optional feature scales used before distance calculation. Use `None`
        for raw distances, `"std"` for standard-deviation scaling over centers
        and candidates, or one positive scale per feature.
    """

    centers: Any
    radius: float
    metric: str = "euclidean"
    feature_columns: tuple[str, ...] | None = None
    feature_scales: Any | None = None

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("CandidateTrustRegion.radius must be finite and non-negative")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "metric", _normalize_distance_metric(self.metric))
        if self.feature_columns is not None:
            columns = tuple(str(column).strip() for column in self.feature_columns)
            if not columns or any(not column for column in columns):
                raise ValueError("feature_columns must contain non-empty column names")
            if len(set(columns)) != len(columns):
                raise ValueError("feature_columns must be unique")
            object.__setattr__(self, "feature_columns", columns)


@dataclass(frozen=True)
class CandidateDuplicatePolicy:
    """Duplicate-avoidance policy for finite-pool BO candidates.

    Parameters
    ----------
    existing_candidates
        Existing measured, already-selected, or otherwise unavailable
        candidates. Use metadata rows for key matching, feature rows for
        distance matching, or both.
    key_columns
        Optional metadata columns used for exact duplicate matching.
    feature_columns
        Optional numeric feature columns used when dataframe features are
        compared by distance.
    feature_tolerance
        Optional maximum descriptor-space distance for feature duplicates.
    metric
        Distance metric: `"euclidean"`, `"manhattan"`, or `"chebyshev"`.
    feature_scales
        Optional feature scales used before distance calculation. Use `None`
        for raw distances, `"std"` for standard-deviation scaling over
        candidates and existing rows, or one positive scale per feature.
    """

    existing_candidates: Any
    key_columns: tuple[str, ...] | None = None
    feature_columns: tuple[str, ...] | None = None
    feature_tolerance: float | None = None
    metric: str = "euclidean"
    feature_scales: Any | None = None

    def __post_init__(self) -> None:
        if self.key_columns is None and self.feature_tolerance is None:
            raise ValueError(
                "CandidateDuplicatePolicy requires key_columns or feature_tolerance"
            )
        if self.key_columns is not None:
            columns = tuple(str(column).strip() for column in self.key_columns)
            if not columns or any(not column for column in columns):
                raise ValueError("key_columns must contain non-empty column names")
            if len(set(columns)) != len(columns):
                raise ValueError("key_columns must be unique")
            object.__setattr__(self, "key_columns", columns)
        if self.feature_columns is not None:
            columns = tuple(str(column).strip() for column in self.feature_columns)
            if not columns or any(not column for column in columns):
                raise ValueError("feature_columns must contain non-empty column names")
            if len(set(columns)) != len(columns):
                raise ValueError("feature_columns must be unique")
            object.__setattr__(self, "feature_columns", columns)
        if self.feature_tolerance is not None:
            tolerance = float(self.feature_tolerance)
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError(
                    "CandidateDuplicatePolicy.feature_tolerance must be finite "
                    "and non-negative"
                )
            object.__setattr__(self, "feature_tolerance", tolerance)
        object.__setattr__(self, "metric", _normalize_distance_metric(self.metric))


@dataclass(frozen=True)
class CandidateConstraint:
    """Finite-pool feasibility constraint for candidate materials.

    Parameters
    ----------
    name
        Short label used in violation reports.
    column
        Candidate dataframe column to evaluate. The column can come from
        `candidate_data` or from `X_candidates` when it is a dataframe and no
        separate `candidate_data` is supplied.
    lower_bound
        Optional numeric lower bound.
    upper_bound
        Optional numeric upper bound.
    allowed_values
        Optional categorical set of allowed values.
    inclusive
        If `True`, numeric bounds are inclusive. If `False`, they are strict.
    allow_missing
        If `True`, missing values satisfy the constraint. The default treats
        missing values as infeasible.
    """

    name: str
    column: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    allowed_values: tuple[Any, ...] | None = None
    inclusive: bool = True
    allow_missing: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        column = str(self.column).strip()
        if not name:
            raise ValueError("CandidateConstraint.name must be non-empty")
        if not column:
            raise ValueError("CandidateConstraint.column must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "column", column)
        if (
            self.lower_bound is None
            and self.upper_bound is None
            and self.allowed_values is None
        ):
            raise ValueError(
                "CandidateConstraint requires at least one of lower_bound, "
                "upper_bound, or allowed_values"
            )
        if self.allowed_values is not None and len(self.allowed_values) == 0:
            raise ValueError("allowed_values must not be empty when provided")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must be <= upper_bound")

    def evaluate(self, candidates: pd.DataFrame) -> pd.Series:
        """Return a boolean feasibility mask for this constraint."""
        if self.column not in candidates.columns:
            raise ValueError(f"Constraint column {self.column!r} is missing")

        values = candidates[self.column]
        feasible = pd.Series(True, index=candidates.index, dtype=bool)

        if self.lower_bound is not None or self.upper_bound is not None:
            numeric = pd.to_numeric(values, errors="coerce")
            if self.lower_bound is not None:
                if self.inclusive:
                    feasible &= numeric >= self.lower_bound
                else:
                    feasible &= numeric > self.lower_bound
            if self.upper_bound is not None:
                if self.inclusive:
                    feasible &= numeric <= self.upper_bound
                else:
                    feasible &= numeric < self.upper_bound

        if self.allowed_values is not None:
            feasible &= values.isin(self.allowed_values)

        if self.allow_missing:
            feasible |= values.isna()
        else:
            feasible &= ~values.isna()

        return feasible.fillna(False).astype(bool)


def apply_candidate_constraints(
    candidates: pd.DataFrame,
    constraints: CandidateConstraint | list[CandidateConstraint] | tuple[CandidateConstraint, ...],
    *,
    require_all: bool = True,
    feasible_column: str = "matgpr_feasible",
    violations_column: str = "matgpr_constraint_violations",
) -> pd.DataFrame:
    """Annotate a candidate table with finite-pool feasibility constraints.

    Parameters
    ----------
    candidates
        Candidate table to annotate.
    constraints
        One or more `CandidateConstraint` objects.
    require_all
        If `True`, candidates must satisfy every constraint. If `False`,
        satisfying at least one constraint is enough.
    feasible_column
        Name of the boolean feasibility column to add.
    violations_column
        Name of the semicolon-separated violation-label column to add.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    constraint_list = _as_constraint_tuple(constraints)
    if not constraint_list:
        raise ValueError("constraints must contain at least one CandidateConstraint")

    evaluated = [(constraint, constraint.evaluate(candidates)) for constraint in constraint_list]
    mask_frame = pd.DataFrame(
        {constraint.name: mask.to_numpy(dtype=bool) for constraint, mask in evaluated},
        index=candidates.index,
    )
    if require_all:
        feasible = mask_frame.all(axis=1)
    else:
        feasible = mask_frame.any(axis=1)

    result = candidates.copy()
    result[feasible_column] = feasible.to_numpy(dtype=bool)
    result[violations_column] = [
        "; ".join(mask_frame.columns[~row.to_numpy(dtype=bool)])
        for _, row in mask_frame.iterrows()
    ]
    return result


def apply_candidate_trust_region(
    candidates: pd.DataFrame,
    trust_region: CandidateTrustRegion,
    *,
    candidate_features: pd.DataFrame | np.ndarray | None = None,
    policy: str = "filter",
    in_region_column: str = "matgpr_in_trust_region",
    distance_column: str = "matgpr_trust_region_distance",
) -> pd.DataFrame:
    """Annotate or filter candidates by a descriptor-space trust region.

    Use this helper to keep a finite BO campaign near known feasible chemistry,
    processing conditions, or local optima. Distances are measured from each
    candidate to the nearest trust-region center.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    if not isinstance(trust_region, CandidateTrustRegion):
        raise TypeError("trust_region must be a CandidateTrustRegion")
    policy = _validate_candidate_policy(policy)
    in_region_column = _validate_output_name(in_region_column, name="in_region_column")
    distance_column = _validate_output_name(distance_column, name="distance_column")
    if in_region_column == distance_column:
        raise ValueError("in_region_column and distance_column must be distinct")

    candidate_data = candidates if candidate_features is None else candidate_features
    distances, in_region = _trust_region_annotations(
        candidate_data,
        trust_region,
        expected_rows=candidates.shape[0],
    )
    result = candidates.copy()
    result[distance_column] = distances
    result[in_region_column] = in_region
    if policy == "filter":
        result = result.loc[in_region].copy()
    return result.reset_index(drop=True)


def apply_candidate_duplicate_policy(
    candidates: pd.DataFrame,
    duplicate_policy: CandidateDuplicatePolicy,
    *,
    candidate_features: pd.DataFrame | np.ndarray | None = None,
    policy: str = "filter",
    duplicate_column: str = "matgpr_is_duplicate",
    distance_column: str = "matgpr_duplicate_distance",
    reason_column: str = "matgpr_duplicate_reason",
) -> pd.DataFrame:
    """Annotate or filter duplicate candidates in a finite BO pool.

    Duplicates can be detected by exact metadata keys, by descriptor-space
    tolerance, or by both. This helps closed-loop BO avoid recommending
    materials that were already measured, selected, or queued.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    if not isinstance(duplicate_policy, CandidateDuplicatePolicy):
        raise TypeError("duplicate_policy must be a CandidateDuplicatePolicy")
    policy = _validate_candidate_policy(policy)
    duplicate_column = _validate_output_name(duplicate_column, name="duplicate_column")
    distance_column = _validate_output_name(distance_column, name="distance_column")
    reason_column = _validate_output_name(reason_column, name="reason_column")
    _validate_distinct_names(duplicate_column, distance_column, reason_column)

    candidate_data = candidates if candidate_features is None else candidate_features
    is_duplicate, distances, reasons = _duplicate_policy_annotations(
        candidates,
        candidate_data,
        duplicate_policy,
        expected_rows=candidates.shape[0],
    )
    result = candidates.copy()
    result[duplicate_column] = is_duplicate
    result[reason_column] = reasons
    if distances is not None:
        result[distance_column] = distances
    if policy == "filter":
        result = result.loc[~is_duplicate].copy()
    return result.reset_index(drop=True)


def select_diverse_batch(
    candidates: pd.DataFrame,
    *,
    top_k: int,
    score_column: str = "matgpr_acquisition",
    feature_columns: list[str] | tuple[str, ...] | None = None,
    diversity_weight: float = 0.25,
    min_distance: float | None = None,
    higher_is_better: bool = True,
    standardize_features: bool = True,
    return_all: bool = False,
) -> pd.DataFrame:
    """Select a diversity-aware batch from a ranked candidate table.

    This helper is useful when several high-acquisition candidates are nearly
    identical in descriptor space, but the next experimental batch should cover
    a broader region of the candidate pool.

    Parameters
    ----------
    candidates
        Candidate table containing an acquisition or utility score.
    top_k
        Maximum number of candidates to select.
    score_column
        Column used as the primary utility score.
    feature_columns
        Numeric columns used to compute diversity distances. If omitted,
        numeric candidate columns other than reserved `matgpr_*` output columns
        and `score_column` are used.
    diversity_weight
        Weight applied to the normalized distance from already selected
        candidates. Set to `0.0` for ordinary score-only top-k selection.
    min_distance
        Optional minimum distance from the already selected batch. If this
        cannot be satisfied, fewer than `top_k` candidates are returned.
    higher_is_better
        If `True`, larger scores are preferred. If `False`, lower scores are
        preferred.
    standardize_features
        If `True`, standardize diversity features before computing distances.
    return_all
        If `True`, return the full candidate table with batch annotations.
        Otherwise return only selected candidates in batch order.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if diversity_weight < 0.0:
        raise ValueError("diversity_weight must be non-negative")
    if min_distance is not None and min_distance < 0.0:
        raise ValueError("min_distance must be non-negative when provided")
    if score_column not in candidates.columns:
        raise ValueError(f"score_column {score_column!r} is missing")

    result = candidates.copy()
    result["matgpr_batch_selected"] = False
    result["matgpr_batch_order"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["matgpr_batch_score"] = np.nan
    result["matgpr_diversity_distance"] = np.nan

    if result.empty:
        return result.reset_index(drop=True)

    score_values = _as_finite_numeric_vector(result[score_column], name=score_column)
    effective_scores = score_values if higher_is_better else -score_values
    normalized_scores = _normalize_vector(effective_scores)

    if diversity_weight == 0.0 and min_distance is None:
        selected_positions = _top_score_positions(effective_scores, top_k)
        selection_scores = normalized_scores[selected_positions]
        diversity_distances = np.full(len(selected_positions), np.nan)
    else:
        selected_positions, selection_scores, diversity_distances = _greedy_diverse_positions(
            result,
            top_k=top_k,
            score_column=score_column,
            feature_columns=feature_columns,
            normalized_scores=normalized_scores,
            diversity_weight=diversity_weight,
            min_distance=min_distance,
            standardize_features=standardize_features,
        )

    for order, (position, selection_score, distance) in enumerate(
        zip(selected_positions, selection_scores, diversity_distances),
        start=1,
    ):
        label = result.index[position]
        result.loc[label, "matgpr_batch_selected"] = True
        result.loc[label, "matgpr_batch_order"] = order
        result.loc[label, "matgpr_batch_score"] = selection_score
        result.loc[label, "matgpr_diversity_distance"] = distance

    if return_all:
        return result.sort_values(
            ["matgpr_batch_selected", "matgpr_batch_order"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    return (
        result[result["matgpr_batch_selected"]]
        .sort_values("matgpr_batch_order")
        .reset_index(drop=True)
    )


def observation_noise_variance(
    data: pd.DataFrame,
    *,
    variance_column: str | None = None,
    std_column: str | None = None,
    sem_column: str | None = None,
    replicate_group_column: str | None = None,
    target_column: str | None = None,
    min_variance: float = 1e-12,
    ddof: int = 1,
) -> pd.Series:
    """Build known observation-noise variances for BoTorch GPR.

    This helper converts common experimental uncertainty formats into the
    per-training-row variance vector expected by `fit_botorch_surrogate` and
    `suggest_next_experiments`.

    Provide exactly one source of known noise:

    - `variance_column`: target variance in squared target units.
    - `std_column`: target standard deviation in target units.
    - `sem_column`: standard error of the mean in target units.
    - `replicate_group_column` with `target_column`: estimate replicate-group
      target variance from repeated measurements.

    Replicate groups with too few rows to estimate variance receive the pooled
    replicate variance when available, otherwise `min_variance`.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not np.isfinite(min_variance) or min_variance < 0.0:
        raise ValueError("min_variance must be a finite non-negative value")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    replicate_requested = replicate_group_column is not None or target_column is not None
    source_count = sum(
        value is not None
        for value in (
            variance_column,
            std_column,
            sem_column,
        )
    ) + int(replicate_requested)
    if source_count != 1:
        raise ValueError(
            "Provide exactly one observation-noise source: variance_column, "
            "std_column, sem_column, or replicate_group_column with target_column"
        )
    if replicate_requested and (replicate_group_column is None or target_column is None):
        raise ValueError(
            "replicate_group_column and target_column must be provided together"
        )

    if variance_column is not None:
        variance = _nonnegative_numeric_column(
            data,
            variance_column,
            label="variance_column",
        )
    elif std_column is not None:
        std = _nonnegative_numeric_column(data, std_column, label="std_column")
        variance = std**2
    elif sem_column is not None:
        sem = _nonnegative_numeric_column(data, sem_column, label="sem_column")
        variance = sem**2
    else:
        variance = _replicate_group_variance(
            data,
            group_column=str(replicate_group_column),
            target_column=str(target_column),
            min_variance=min_variance,
            ddof=ddof,
        )

    variance = variance.astype(float).clip(lower=min_variance)
    variance.name = "matgpr_noise_variance"
    return variance


def fit_botorch_surrogate(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    *,
    maximize: bool = True,
    noise_variance: float | np.ndarray | pd.Series | None = None,
    normalize_features: bool = True,
    standardize_target: bool = True,
    fit_model: bool = True,
    device: str | None = None,
) -> BoTorchSurrogate:
    """Fit a BoTorch `SingleTaskGP` surrogate for Bayesian optimization.

    This helper keeps BoTorch optional. Install the Bayesian-optimization extra
    before using it:

    ```bash
    python -m pip install "matgpr[bo]"
    ```

    Parameters
    ----------
    X_train
        Numeric training features with shape `(n_samples, n_features)`.
    y_train
        Numeric target values with shape `(n_samples,)`.
    maximize
        If `True`, larger target values are preferred. If `False`, the target
        is negated internally so BoTorch can solve the equivalent maximization
        problem.
    noise_variance
        Optional known observation-noise variance. Provide either a scalar or
        one variance per training sample. These are variances, not standard
        deviations.
    normalize_features
        If `True`, use BoTorch's `Normalize` input transform.
    standardize_target
        If `True`, use BoTorch's `Standardize` outcome transform.
    fit_model
        If `True`, optimize GP hyperparameters with `fit_gpytorch_mll`.
        Disable only for tests or advanced workflows that fit the model later.
    device
        Optional torch device string such as `"cpu"` or `"cuda"`.
    """
    require_optional_dependency("botorch")
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize

    train_array, feature_names, _ = _as_numeric_matrix(X_train, name="X_train")
    target_array = _as_target_vector(y_train, expected_length=train_array.shape[0], name="y_train")
    objective_array = target_array if maximize else -target_array

    torch_device = torch.device(device) if device is not None else None
    train_X = _to_tensor(torch, train_array, device=torch_device)
    train_Y = _to_tensor(torch, objective_array.reshape(-1, 1), device=torch_device)
    train_y_original = _to_tensor(torch, target_array.reshape(-1, 1), device=torch_device)

    model_kwargs: dict[str, Any] = {}
    noise_tensor = None
    if noise_variance is not None:
        noise_array = _as_noise_variance(
            noise_variance,
            expected_length=train_array.shape[0],
        )
        noise_tensor = _to_tensor(
            torch,
            noise_array.reshape(-1, 1),
            device=torch_device,
        )
        model_kwargs["train_Yvar"] = noise_tensor
    if normalize_features:
        model_kwargs["input_transform"] = Normalize(d=train_X.shape[-1])
    if standardize_target:
        model_kwargs["outcome_transform"] = Standardize(m=1)

    model = SingleTaskGP(train_X=train_X, train_Y=train_Y, **model_kwargs)
    if torch_device is not None:
        model = model.to(torch_device)

    if fit_model:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    return BoTorchSurrogate(
        model=model,
        train_X=train_X,
        train_y=train_y_original,
        objective_y=train_Y,
        maximize=maximize,
        best_observed_objective=float(train_Y.max().detach().cpu().item()),
        feature_names=feature_names,
        noise_variance=noise_tensor,
    )


def fit_multi_objective_botorch_surrogate(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
    *,
    objective_directions: str | Sequence[str] = "maximize",
    objective_names: Sequence[str] | None = None,
    reference_point: Sequence[float] | np.ndarray | pd.Series | None = None,
    noise_variance: float | np.ndarray | pd.DataFrame | None = None,
    normalize_features: bool = True,
    standardize_targets: bool = True,
    fit_model: bool = True,
    device: str | None = None,
) -> MultiObjectiveBoTorchSurrogate:
    """Fit independent BoTorch GP surrogates for multi-objective BO.

    Parameters
    ----------
    X_train
        Numeric training features with shape `(n_samples, n_features)`.
    y_train
        Numeric target matrix with shape `(n_samples, n_objectives)`. At least
        two objectives are required.
    objective_directions
        Either one direction applied to every objective or one direction per
        objective. Use `"maximize"` for objectives where larger values are
        better and `"minimize"` for objectives where smaller values are better.
    objective_names
        Optional objective names used in output columns. If omitted, dataframe
        column names are used when available.
    reference_point
        Optional hypervolume reference point in original objective units and
        directions. If omitted, a conservative point worse than the observed
        training targets is estimated.
    noise_variance
        Optional known observation-noise variance. Provide a scalar, a vector
        with one variance per training row, or a matrix with one column per
        objective.
    normalize_features
        If `True`, use BoTorch's `Normalize` input transform for each GP.
    standardize_targets
        If `True`, use BoTorch's `Standardize` outcome transform for each GP.
    fit_model
        If `True`, optimize GP hyperparameters with `fit_gpytorch_mll`.
    device
        Optional torch device string such as `"cpu"` or `"cuda"`.
    """
    require_optional_dependency("botorch")
    import torch
    from gpytorch.mlls import SumMarginalLogLikelihood

    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize

    train_array, feature_names, _ = _as_numeric_matrix(X_train, name="X_train")
    target_array, inferred_names = _as_target_matrix(
        y_train,
        expected_length=train_array.shape[0],
        name="y_train",
    )
    objective_names = _resolve_objective_names(
        objective_names,
        inferred_names=inferred_names,
        n_objectives=target_array.shape[1],
    )
    directions, signs = _resolve_objective_directions(
        objective_directions,
        n_objectives=target_array.shape[1],
    )
    objective_array = target_array * signs.reshape(1, -1)
    ref_array, ref_original = _resolve_reference_point(
        reference_point,
        objective_array=objective_array,
        signs=signs,
    )

    torch_device = torch.device(device) if device is not None else None
    train_X = _to_tensor(torch, train_array, device=torch_device)
    train_y_original = _to_tensor(torch, target_array, device=torch_device)
    train_Y = _to_tensor(torch, objective_array, device=torch_device)
    reference_tensor = _to_tensor(torch, ref_array, device=torch_device)
    reference_original_tensor = _to_tensor(torch, ref_original, device=torch_device)
    signs_tensor = _to_tensor(torch, signs, device=torch_device)

    noise_tensor = None
    noise_array = None
    if noise_variance is not None:
        noise_array = _as_noise_variance_matrix(
            noise_variance,
            expected_length=train_array.shape[0],
            n_objectives=target_array.shape[1],
        )
        noise_tensor = _to_tensor(torch, noise_array, device=torch_device)

    models = []
    for objective_index in range(target_array.shape[1]):
        model_kwargs: dict[str, Any] = {}
        if noise_array is not None:
            model_kwargs["train_Yvar"] = _to_tensor(
                torch,
                noise_array[:, [objective_index]],
                device=torch_device,
            )
        if normalize_features:
            model_kwargs["input_transform"] = Normalize(d=train_X.shape[-1])
        if standardize_targets:
            model_kwargs["outcome_transform"] = Standardize(m=1)
        models.append(
            SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y[:, [objective_index]],
                **model_kwargs,
            )
        )

    model = ModelListGP(*models)
    if torch_device is not None:
        model = model.to(torch_device)

    if fit_model:
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    return MultiObjectiveBoTorchSurrogate(
        model=model,
        train_X=train_X,
        train_y=train_y_original,
        objective_y=train_Y,
        objective_names=objective_names,
        objective_directions=directions,
        objective_signs=signs_tensor,
        reference_point=reference_tensor,
        reference_point_original=reference_original_tensor,
        feature_names=feature_names,
        noise_variance=noise_tensor,
    )


def rank_discrete_candidates(
    surrogate: BoTorchSurrogate,
    X_candidates: pd.DataFrame | np.ndarray,
    *,
    acquisition_function: str = "log_expected_improvement",
    top_k: int | None = None,
    beta: float = 0.2,
    candidate_data: pd.DataFrame | None = None,
    constraints: CandidateConstraint | list[CandidateConstraint] | tuple[CandidateConstraint, ...] | None = None,
    constraint_policy: str = "filter",
    trust_region: CandidateTrustRegion | None = None,
    trust_region_policy: str = "filter",
    duplicate_policy: CandidateDuplicatePolicy | None = None,
    duplicate_policy_action: str = "filter",
) -> pd.DataFrame:
    """Rank a finite pool of candidate materials using a BoTorch acquisition.

    This is the first Bayesian-optimization workflow in `matgpr`: users provide
    already-featurized candidate materials, and the function ranks them for the
    next experimental or simulation batch.

    Parameters
    ----------
    surrogate
        Fitted surrogate returned by `fit_botorch_surrogate`.
    X_candidates
        Numeric candidate features with shape `(n_candidates, n_features)`.
    acquisition_function
        Acquisition function name. Supported values are
        `"log_expected_improvement"`, `"log_noisy_expected_improvement"`,
        `"expected_improvement"`, `"noisy_expected_improvement"`,
        `"upper_confidence_bound"`, and `"probability_of_improvement"`.
    top_k
        Optional number of rows to return after ranking. If omitted, all
        candidates are returned.
    beta
        Exploration weight for upper confidence bound.
    candidate_data
        Optional metadata dataframe to carry through to the ranked output, such
        as formulas, SMILES strings, synthesis conditions, or sample IDs.
    constraints
        Optional finite-pool feasibility constraints applied after candidate
        metadata and acquisition scores are assembled.
    constraint_policy
        `"filter"` returns only feasible candidates. `"annotate"` keeps all
        candidates and adds feasibility columns.
    trust_region
        Optional descriptor-space trust region for local candidate filtering.
    trust_region_policy
        `"filter"` returns only candidates inside the trust region.
        `"annotate"` keeps all candidates and adds trust-region columns.
    duplicate_policy
        Optional duplicate-avoidance policy for already measured, selected, or
        queued candidates.
    duplicate_policy_action
        `"filter"` removes duplicate candidates. `"annotate"` keeps all
        candidates and adds duplicate-audit columns.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided")
    constraint_policy = _validate_constraint_policy(constraint_policy)

    import torch

    candidate_array, _, candidate_index = _as_numeric_matrix(
        X_candidates,
        name="X_candidates",
    )
    if candidate_array.shape[1] != surrogate.train_X.shape[-1]:
        raise ValueError(
            "X_candidates must have the same number of features as X_train "
            f"({candidate_array.shape[1]} != {surrogate.train_X.shape[-1]})"
        )
    ranked = _candidate_frame(
        X_candidates=X_candidates,
        candidate_data=candidate_data,
        candidate_index=candidate_index,
    )

    candidate_X = _to_tensor(
        torch,
        candidate_array,
        device=surrogate.train_X.device,
    )
    acquisition = _build_acquisition_function(
        acquisition_function,
        surrogate=surrogate,
        beta=beta,
    )

    surrogate.model.eval()
    with torch.no_grad():
        scores = acquisition(candidate_X.unsqueeze(-2)).detach().cpu().numpy().reshape(-1)
        posterior = surrogate.model.posterior(candidate_X)
        transformed_mean = posterior.mean.detach().cpu().numpy().reshape(-1)
        std = posterior.variance.clamp_min(0.0).sqrt().detach().cpu().numpy().reshape(-1)

    predicted_mean = transformed_mean if surrogate.maximize else -transformed_mean
    ranked["matgpr_predicted_mean"] = predicted_mean
    ranked["matgpr_predicted_std"] = std
    ranked["matgpr_acquisition"] = scores
    if constraints is not None:
        ranked = apply_candidate_constraints(ranked, constraints)
        if constraint_policy == "filter":
            keep_mask = ranked["matgpr_feasible"].to_numpy(dtype=bool)
            ranked = ranked.loc[keep_mask].copy()
            candidate_array = candidate_array[keep_mask]
    ranked = _apply_candidate_domain_policies(
        ranked,
        candidate_array,
        trust_region=trust_region,
        trust_region_policy=trust_region_policy,
        duplicate_policy=duplicate_policy,
        duplicate_policy_action=duplicate_policy_action,
    )
    ranked = ranked.sort_values("matgpr_acquisition", ascending=False).reset_index(drop=True)
    ranked.insert(0, "matgpr_rank", np.arange(1, ranked.shape[0] + 1))

    if top_k is not None:
        return ranked.head(top_k).reset_index(drop=True)
    return ranked


def rank_multi_objective_discrete_candidates(
    surrogate: MultiObjectiveBoTorchSurrogate,
    X_candidates: pd.DataFrame | np.ndarray,
    *,
    acquisition_function: str = "q_log_expected_hypervolume_improvement",
    top_k: int | None = None,
    candidate_data: pd.DataFrame | None = None,
    constraints: CandidateConstraint | list[CandidateConstraint] | tuple[CandidateConstraint, ...] | None = None,
    constraint_policy: str = "filter",
    trust_region: CandidateTrustRegion | None = None,
    trust_region_policy: str = "filter",
    duplicate_policy: CandidateDuplicatePolicy | None = None,
    duplicate_policy_action: str = "filter",
    mc_samples: int = 128,
    sampler_seed: int | None = 42,
) -> pd.DataFrame:
    """Rank a finite pool with multi-objective hypervolume improvement.

    The returned table is sorted by acquisition value and includes posterior
    mean and standard-deviation columns for every objective in original units.
    Acquisition functions operate in BoTorch maximization space, so
    minimize-type objectives are handled by the fitted surrogate.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided")
    constraint_policy = _validate_constraint_policy(constraint_policy)
    mc_samples = _validate_mc_samples(mc_samples)

    import torch

    candidate_array, _, candidate_index = _as_numeric_matrix(
        X_candidates,
        name="X_candidates",
    )
    if candidate_array.shape[1] != surrogate.train_X.shape[-1]:
        raise ValueError(
            "X_candidates must have the same number of features as X_train "
            f"({candidate_array.shape[1]} != {surrogate.train_X.shape[-1]})"
        )
    ranked = _candidate_frame(
        X_candidates=X_candidates,
        candidate_data=candidate_data,
        candidate_index=candidate_index,
    )

    candidate_X = _to_tensor(
        torch,
        candidate_array,
        device=surrogate.train_X.device,
    )
    acquisition = _build_multi_objective_acquisition_function(
        acquisition_function,
        surrogate=surrogate,
        mc_samples=mc_samples,
        sampler_seed=sampler_seed,
    )

    surrogate.model.eval()
    with torch.no_grad():
        scores = acquisition(candidate_X.unsqueeze(-2)).detach().cpu().numpy().reshape(-1)
        posterior = surrogate.model.posterior(candidate_X)
        objective_mean = posterior.mean.detach().cpu()
        std = posterior.variance.clamp_min(0.0).sqrt().detach().cpu().numpy()

    signs = surrogate.objective_signs.detach().cpu()
    predicted_mean = (objective_mean * signs).numpy()
    objective_columns = []
    for objective_index, objective_name in enumerate(surrogate.objective_names):
        slug = _slugify_objective_name(objective_name)
        mean_column = f"matgpr_predicted_mean_{slug}"
        std_column = f"matgpr_predicted_std_{slug}"
        ranked[mean_column] = predicted_mean[:, objective_index]
        ranked[std_column] = std[:, objective_index]
        objective_columns.append(mean_column)

    ranked["matgpr_acquisition"] = scores
    ranked["matgpr_predicted_pareto_front"] = pareto_front_mask(
        ranked,
        [
            ObjectiveSpec(
                name=surrogate.objective_names[index],
                column=column,
                goal=surrogate.objective_directions[index],
            )
            for index, column in enumerate(objective_columns)
        ],
    ).to_numpy(dtype=bool)

    if constraints is not None:
        ranked = apply_candidate_constraints(ranked, constraints)
        if constraint_policy == "filter":
            keep_mask = ranked["matgpr_feasible"].to_numpy(dtype=bool)
            ranked = ranked.loc[keep_mask].copy()
            candidate_array = candidate_array[keep_mask]
    ranked = _apply_candidate_domain_policies(
        ranked,
        candidate_array,
        trust_region=trust_region,
        trust_region_policy=trust_region_policy,
        duplicate_policy=duplicate_policy,
        duplicate_policy_action=duplicate_policy_action,
    )
    ranked = ranked.sort_values("matgpr_acquisition", ascending=False).reset_index(drop=True)
    ranked.insert(0, "matgpr_rank", np.arange(1, ranked.shape[0] + 1))

    if top_k is not None:
        return ranked.head(top_k).reset_index(drop=True)
    return ranked


def select_sequential_multi_objective_batch(
    surrogate: MultiObjectiveBoTorchSurrogate,
    X_candidates: pd.DataFrame | np.ndarray,
    *,
    top_k: int,
    acquisition_function: str = "q_log_expected_hypervolume_improvement",
    candidate_data: pd.DataFrame | None = None,
    constraints: CandidateConstraint | list[CandidateConstraint] | tuple[CandidateConstraint, ...] | None = None,
    constraint_policy: str = "filter",
    trust_region: CandidateTrustRegion | None = None,
    trust_region_policy: str = "filter",
    duplicate_policy: CandidateDuplicatePolicy | None = None,
    duplicate_policy_action: str = "filter",
    mc_samples: int = 128,
    sampler_seed: int | None = 42,
    return_all: bool = False,
) -> pd.DataFrame:
    """Select a greedy multi-objective batch from a finite candidate pool.

    Candidates are selected one at a time. After each selection, the selected
    candidates are treated as pending points and the hypervolume-improvement
    acquisition is recomputed for the remaining candidates. This is useful
    when recommending a small experimental batch, because the second and later
    choices account for what was already selected rather than simply taking the
    top individual acquisition scores.

    Parameters
    ----------
    surrogate
        Fitted multi-objective surrogate returned by
        `fit_multi_objective_botorch_surrogate`.
    X_candidates
        Numeric candidate features with shape `(n_candidates, n_features)`.
    top_k
        Maximum number of candidates to select.
    acquisition_function
        Multi-objective acquisition function name. Supported values match
        `rank_multi_objective_discrete_candidates`.
    candidate_data
        Optional metadata dataframe to carry through to the output.
    constraints
        Optional finite-pool feasibility constraints.
    constraint_policy
        `"filter"` selects only feasible candidates. `"annotate"` keeps all
        candidates and adds feasibility columns.
    mc_samples
        Number of Monte Carlo samples for hypervolume acquisition estimates.
    sampler_seed
        Optional seed for reproducible Sobol sampling.
    return_all
        If `True`, return the full annotated candidate table. Otherwise return
        only selected candidates in sequential batch order.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    constraint_policy = _validate_constraint_policy(constraint_policy)
    mc_samples = _validate_mc_samples(mc_samples)

    import torch

    candidate_array, _, candidate_index = _as_numeric_matrix(
        X_candidates,
        name="X_candidates",
    )
    if candidate_array.shape[1] != surrogate.train_X.shape[-1]:
        raise ValueError(
            "X_candidates must have the same number of features as X_train "
            f"({candidate_array.shape[1]} != {surrogate.train_X.shape[-1]})"
        )

    result = _candidate_frame(
        X_candidates=X_candidates,
        candidate_data=candidate_data,
        candidate_index=candidate_index,
    )
    candidate_X = _to_tensor(
        torch,
        candidate_array,
        device=surrogate.train_X.device,
    )
    single_scores, predicted_mean, predicted_std = _multi_objective_scores_and_predictions(
        surrogate,
        candidate_X,
        acquisition_function=acquisition_function,
        mc_samples=mc_samples,
        sampler_seed=sampler_seed,
        x_pending=None,
    )
    objective_columns = _add_multi_objective_prediction_columns(
        result,
        surrogate=surrogate,
        predicted_mean=predicted_mean,
        predicted_std=predicted_std,
    )
    result["matgpr_acquisition"] = single_scores
    result["matgpr_predicted_pareto_front"] = pareto_front_mask(
        result,
        [
            ObjectiveSpec(
                name=surrogate.objective_names[index],
                column=column,
                goal=surrogate.objective_directions[index],
            )
            for index, column in enumerate(objective_columns)
        ],
    ).to_numpy(dtype=bool)

    if constraints is not None:
        result = apply_candidate_constraints(result, constraints)
        if constraint_policy == "filter":
            keep_mask = result["matgpr_feasible"].to_numpy(dtype=bool)
            result = result.loc[keep_mask].copy()
            candidate_array = candidate_array[keep_mask]
            candidate_X = candidate_X[torch.as_tensor(keep_mask, device=candidate_X.device)]

    result, keep_mask = _apply_candidate_domain_policies_with_mask(
        result,
        candidate_array,
        trust_region=trust_region,
        trust_region_policy=trust_region_policy,
        duplicate_policy=duplicate_policy,
        duplicate_policy_action=duplicate_policy_action,
    )
    candidate_X = candidate_X[torch.as_tensor(keep_mask, device=candidate_X.device)]
    result = result.reset_index(drop=True)
    result.insert(0, "matgpr_rank", _rank_descending(result["matgpr_acquisition"]))
    result["matgpr_batch_selected"] = False
    result["matgpr_batch_order"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["matgpr_batch_score"] = np.nan

    if result.empty:
        return result.reset_index(drop=True)

    selected_positions, selected_scores = _greedy_multi_objective_batch_positions(
        surrogate,
        candidate_X,
        top_k=min(top_k, result.shape[0]),
        acquisition_function=acquisition_function,
        mc_samples=mc_samples,
        sampler_seed=sampler_seed,
    )
    for order, (position, score) in enumerate(
        zip(selected_positions, selected_scores),
        start=1,
    ):
        result.loc[position, "matgpr_batch_selected"] = True
        result.loc[position, "matgpr_batch_order"] = order
        result.loc[position, "matgpr_batch_score"] = score

    if return_all:
        return result.sort_values(
            ["matgpr_batch_selected", "matgpr_batch_order", "matgpr_rank"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)

    return (
        result[result["matgpr_batch_selected"]]
        .sort_values("matgpr_batch_order")
        .reset_index(drop=True)
    )


def suggest_next_experiments(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_candidates: pd.DataFrame | np.ndarray,
    *,
    candidate_data: pd.DataFrame | None = None,
    top_k: int = 5,
    acquisition_function: str = "log_expected_improvement",
    maximize: bool = True,
    noise_variance: float | np.ndarray | pd.Series | None = None,
    normalize_features: bool = True,
    standardize_target: bool = True,
    beta: float = 0.2,
    fit_model: bool = True,
    device: str | None = None,
    constraints: CandidateConstraint | list[CandidateConstraint] | tuple[CandidateConstraint, ...] | None = None,
    constraint_policy: str = "filter",
    trust_region: CandidateTrustRegion | None = None,
    trust_region_policy: str = "filter",
    duplicate_policy: CandidateDuplicatePolicy | None = None,
    duplicate_policy_action: str = "filter",
    batch_selection: str = "top",
    batch_feature_columns: list[str] | tuple[str, ...] | None = None,
    diversity_weight: float = 0.25,
    min_batch_distance: float | None = None,
) -> BayesianOptimizationResult:
    """Fit a surrogate and recommend the next materials to evaluate.

    Use this high-level function when the candidate space is a finite list of
    materials, compositions, molecules, polymers, or processing conditions that
    have already been featurized.

    Parameters
    ----------
    X_train
        Numeric features for materials with measured target values.
    y_train
        Measured target values.
    X_candidates
        Numeric features for unmeasured candidate materials.
    candidate_data
        Optional metadata for the candidates. This is returned with the
        recommendation table.
    top_k
        Number of candidates to recommend.
    acquisition_function
        Acquisition function used for ranking. See
        `rank_discrete_candidates`.
    maximize
        If `True`, recommend candidates expected to maximize the target. If
        `False`, recommend candidates expected to minimize the target.
    noise_variance
        Optional known target-noise variance for each training row or a scalar
        variance shared by all rows.
    normalize_features
        If `True`, use BoTorch's feature normalization transform.
    standardize_target
        If `True`, use BoTorch's target standardization transform.
    beta
        Exploration weight for upper confidence bound.
    fit_model
        If `True`, fit GP hyperparameters before ranking.
    device
        Optional torch device string.
    constraints
        Optional finite-pool feasibility constraints.
    constraint_policy
        `"filter"` recommends only feasible candidates. `"annotate"` ranks all
        candidates and marks feasibility.
    trust_region
        Optional descriptor-space trust region.
    trust_region_policy
        `"filter"` recommends only candidates inside the trust region.
        `"annotate"` keeps all candidates and marks trust-region membership.
    duplicate_policy
        Optional duplicate-avoidance policy.
    duplicate_policy_action
        `"filter"` removes duplicate candidates. `"annotate"` keeps all
        candidates and marks duplicates.
    batch_selection
        `"top"` returns the highest-acquisition candidates. `"diverse"` uses
        greedy diversity-aware batch selection.
    batch_feature_columns
        Numeric columns in the ranked candidate table used for diversity-aware
        batch selection.
    diversity_weight
        Diversity weight used when `batch_selection="diverse"`.
    min_batch_distance
        Optional minimum descriptor-space distance between selected batch
        members when `batch_selection="diverse"`.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    constraint_policy = _validate_constraint_policy(constraint_policy)
    batch_selection = _validate_batch_selection(batch_selection)

    surrogate = fit_botorch_surrogate(
        X_train,
        y_train,
        maximize=maximize,
        noise_variance=noise_variance,
        normalize_features=normalize_features,
        standardize_target=standardize_target,
        fit_model=fit_model,
        device=device,
    )
    ranked_candidates = rank_discrete_candidates(
        surrogate,
        X_candidates,
        acquisition_function=acquisition_function,
        top_k=None,
        beta=beta,
        candidate_data=candidate_data,
        constraints=constraints,
        constraint_policy=constraint_policy,
        trust_region=trust_region,
        trust_region_policy=trust_region_policy,
        duplicate_policy=duplicate_policy,
        duplicate_policy_action=duplicate_policy_action,
    )
    if batch_selection == "diverse":
        recommendations = select_diverse_batch(
            ranked_candidates,
            top_k=top_k,
            feature_columns=batch_feature_columns,
            diversity_weight=diversity_weight,
            min_distance=min_batch_distance,
            return_all=False,
        )
    else:
        recommendations = ranked_candidates.head(top_k).reset_index(drop=True)

    return BayesianOptimizationResult(
        recommendations=recommendations,
        ranked_candidates=ranked_candidates,
        surrogate=surrogate,
        acquisition_function=acquisition_function,
        maximize=maximize,
        top_k=top_k,
    )


def suggest_multi_objective_next_experiments(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
    X_candidates: pd.DataFrame | np.ndarray,
    *,
    objective_directions: str | Sequence[str] = "maximize",
    objective_names: Sequence[str] | None = None,
    reference_point: Sequence[float] | np.ndarray | pd.Series | None = None,
    candidate_data: pd.DataFrame | None = None,
    top_k: int = 5,
    acquisition_function: str = "q_log_expected_hypervolume_improvement",
    noise_variance: float | np.ndarray | pd.DataFrame | None = None,
    normalize_features: bool = True,
    standardize_targets: bool = True,
    fit_model: bool = True,
    device: str | None = None,
    constraints: CandidateConstraint | list[CandidateConstraint] | tuple[CandidateConstraint, ...] | None = None,
    constraint_policy: str = "filter",
    trust_region: CandidateTrustRegion | None = None,
    trust_region_policy: str = "filter",
    duplicate_policy: CandidateDuplicatePolicy | None = None,
    duplicate_policy_action: str = "filter",
    batch_selection: str = "top",
    batch_feature_columns: list[str] | tuple[str, ...] | None = None,
    diversity_weight: float = 0.25,
    min_batch_distance: float | None = None,
    mc_samples: int = 128,
    sampler_seed: int | None = 42,
) -> MultiObjectiveBayesianOptimizationResult:
    """Fit multi-objective surrogates and recommend finite-pool candidates.

    Use this when a next experiment must trade off two or more measured
    objectives, such as maximizing conductivity while minimizing degradation
    rate, toxicity, synthesis cost, or processing burden.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    constraint_policy = _validate_constraint_policy(constraint_policy)
    batch_selection = _validate_batch_selection(batch_selection, allow_sequential=True)

    surrogate = fit_multi_objective_botorch_surrogate(
        X_train,
        y_train,
        objective_directions=objective_directions,
        objective_names=objective_names,
        reference_point=reference_point,
        noise_variance=noise_variance,
        normalize_features=normalize_features,
        standardize_targets=standardize_targets,
        fit_model=fit_model,
        device=device,
    )
    ranked_candidates = rank_multi_objective_discrete_candidates(
        surrogate,
        X_candidates,
        acquisition_function=acquisition_function,
        top_k=None,
        candidate_data=candidate_data,
        constraints=constraints,
        constraint_policy=constraint_policy,
        trust_region=trust_region,
        trust_region_policy=trust_region_policy,
        duplicate_policy=duplicate_policy,
        duplicate_policy_action=duplicate_policy_action,
        mc_samples=mc_samples,
        sampler_seed=sampler_seed,
    )
    if batch_selection == "sequential":
        recommendations = select_sequential_multi_objective_batch(
            surrogate,
            X_candidates,
            top_k=top_k,
            acquisition_function=acquisition_function,
            candidate_data=candidate_data,
            constraints=constraints,
            constraint_policy=constraint_policy,
            trust_region=trust_region,
            trust_region_policy=trust_region_policy,
            duplicate_policy=duplicate_policy,
            duplicate_policy_action=duplicate_policy_action,
            mc_samples=mc_samples,
            sampler_seed=sampler_seed,
            return_all=False,
        )
    elif batch_selection == "diverse":
        recommendations = select_diverse_batch(
            ranked_candidates,
            top_k=top_k,
            feature_columns=batch_feature_columns,
            diversity_weight=diversity_weight,
            min_distance=min_batch_distance,
            return_all=False,
        )
    else:
        recommendations = ranked_candidates.head(top_k).reset_index(drop=True)

    return MultiObjectiveBayesianOptimizationResult(
        recommendations=recommendations,
        ranked_candidates=ranked_candidates,
        surrogate=surrogate,
        acquisition_function=acquisition_function,
        objective_names=surrogate.objective_names,
        objective_directions=surrogate.objective_directions,
        top_k=top_k,
    )


def _build_acquisition_function(
    name: str,
    *,
    surrogate: BoTorchSurrogate,
    beta: float,
) -> Any:
    normalized = _normalize_acquisition_name(name)
    if normalized == "log_expected_improvement":
        from botorch.acquisition.analytic import LogExpectedImprovement

        return LogExpectedImprovement(
            model=surrogate.model,
            best_f=surrogate.best_observed_objective,
        )
    if normalized == "log_noisy_expected_improvement":
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement

        return qLogNoisyExpectedImprovement(
            model=surrogate.model,
            X_baseline=surrogate.train_X,
        )
    if normalized == "expected_improvement":
        from botorch.acquisition.analytic import ExpectedImprovement

        return ExpectedImprovement(
            model=surrogate.model,
            best_f=surrogate.best_observed_objective,
        )
    if normalized == "noisy_expected_improvement":
        from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

        return qNoisyExpectedImprovement(
            model=surrogate.model,
            X_baseline=surrogate.train_X,
        )
    if normalized == "probability_of_improvement":
        from botorch.acquisition.analytic import ProbabilityOfImprovement

        return ProbabilityOfImprovement(
            model=surrogate.model,
            best_f=surrogate.best_observed_objective,
        )
    if normalized == "upper_confidence_bound":
        from botorch.acquisition.analytic import UpperConfidenceBound

        return UpperConfidenceBound(model=surrogate.model, beta=beta)
    raise ValueError(
        "acquisition_function must be one of "
        "'log_expected_improvement', 'log_noisy_expected_improvement', "
        "'expected_improvement', 'noisy_expected_improvement', "
        "'probability_of_improvement', or 'upper_confidence_bound'"
    )


def _build_multi_objective_acquisition_function(
    name: str,
    *,
    surrogate: MultiObjectiveBoTorchSurrogate,
    mc_samples: int,
    sampler_seed: int | None,
    x_pending: Any | None = None,
) -> Any:
    normalized = _normalize_multi_objective_acquisition_name(name)
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler

    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([mc_samples]),
        seed=sampler_seed,
    )
    if normalized == "q_log_expected_hypervolume_improvement":
        from botorch.acquisition.multi_objective.logei import (
            qLogExpectedHypervolumeImprovement,
        )
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            NondominatedPartitioning,
        )

        partitioning = NondominatedPartitioning(
            ref_point=surrogate.reference_point,
            Y=surrogate.objective_y,
        )
        return qLogExpectedHypervolumeImprovement(
            model=surrogate.model,
            ref_point=surrogate.reference_point,
            partitioning=partitioning,
            sampler=sampler,
            X_pending=x_pending,
        )
    if normalized == "q_log_noisy_expected_hypervolume_improvement":
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )

        return qLogNoisyExpectedHypervolumeImprovement(
            model=surrogate.model,
            ref_point=surrogate.reference_point,
            X_baseline=surrogate.train_X,
            sampler=sampler,
            X_pending=x_pending,
        )
    if normalized == "q_expected_hypervolume_improvement":
        from botorch.acquisition.multi_objective.monte_carlo import (
            qExpectedHypervolumeImprovement,
        )
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            NondominatedPartitioning,
        )

        partitioning = NondominatedPartitioning(
            ref_point=surrogate.reference_point,
            Y=surrogate.objective_y,
        )
        return qExpectedHypervolumeImprovement(
            model=surrogate.model,
            ref_point=surrogate.reference_point,
            partitioning=partitioning,
            sampler=sampler,
            X_pending=x_pending,
        )
    if normalized == "q_noisy_expected_hypervolume_improvement":
        from botorch.acquisition.multi_objective.monte_carlo import (
            qNoisyExpectedHypervolumeImprovement,
        )

        return qNoisyExpectedHypervolumeImprovement(
            model=surrogate.model,
            ref_point=surrogate.reference_point,
            X_baseline=surrogate.train_X,
            sampler=sampler,
            X_pending=x_pending,
        )
    raise ValueError(
        "acquisition_function must be one of "
        "'q_log_expected_hypervolume_improvement', "
        "'q_log_noisy_expected_hypervolume_improvement', "
        "'q_expected_hypervolume_improvement', or "
        "'q_noisy_expected_hypervolume_improvement'"
    )


def _multi_objective_scores_and_predictions(
    surrogate: MultiObjectiveBoTorchSurrogate,
    candidate_X: Any,
    *,
    acquisition_function: str,
    mc_samples: int,
    sampler_seed: int | None,
    x_pending: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    acquisition = _build_multi_objective_acquisition_function(
        acquisition_function,
        surrogate=surrogate,
        mc_samples=mc_samples,
        sampler_seed=sampler_seed,
        x_pending=x_pending,
    )

    surrogate.model.eval()
    with torch.no_grad():
        scores = acquisition(candidate_X.unsqueeze(-2)).detach().cpu().numpy().reshape(-1)
        posterior = surrogate.model.posterior(candidate_X)
        objective_mean = posterior.mean.detach().cpu()
        std = posterior.variance.clamp_min(0.0).sqrt().detach().cpu().numpy()

    signs = surrogate.objective_signs.detach().cpu()
    predicted_mean = (objective_mean * signs).numpy()
    return scores, predicted_mean, std


def _add_multi_objective_prediction_columns(
    data: pd.DataFrame,
    *,
    surrogate: MultiObjectiveBoTorchSurrogate,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
) -> list[str]:
    objective_columns = []
    for objective_index, objective_name in enumerate(surrogate.objective_names):
        slug = _slugify_objective_name(objective_name)
        mean_column = f"matgpr_predicted_mean_{slug}"
        std_column = f"matgpr_predicted_std_{slug}"
        data[mean_column] = predicted_mean[:, objective_index]
        data[std_column] = predicted_std[:, objective_index]
        objective_columns.append(mean_column)
    return objective_columns


def _greedy_multi_objective_batch_positions(
    surrogate: MultiObjectiveBoTorchSurrogate,
    candidate_X: Any,
    *,
    top_k: int,
    acquisition_function: str,
    mc_samples: int,
    sampler_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    selected_scores: list[float] = []
    remaining = np.ones(candidate_X.shape[0], dtype=bool)

    while len(selected) < top_k and np.any(remaining):
        x_pending = candidate_X[selected] if selected else None
        scores, _, _ = _multi_objective_scores_and_predictions(
            surrogate,
            candidate_X,
            acquisition_function=acquisition_function,
            mc_samples=mc_samples,
            sampler_seed=sampler_seed,
            x_pending=x_pending,
        )
        scores[~remaining] = -np.inf
        chosen = _argmax_stable(scores)
        if chosen is None:
            break
        selected.append(chosen)
        selected_scores.append(float(scores[chosen]))
        remaining[chosen] = False

    return (
        np.asarray(selected, dtype=int),
        np.asarray(selected_scores, dtype=float),
    )


def _normalize_acquisition_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "logei": "log_expected_improvement",
        "log_ei": "log_expected_improvement",
        "log_expected_improvement": "log_expected_improvement",
        "lognei": "log_noisy_expected_improvement",
        "log_noisy_ei": "log_noisy_expected_improvement",
        "log_nei": "log_noisy_expected_improvement",
        "log_noisy_expected_improvement": "log_noisy_expected_improvement",
        "q_log_nei": "log_noisy_expected_improvement",
        "q_log_noisy_expected_improvement": "log_noisy_expected_improvement",
        "qlognoisyexpectedimprovement": "log_noisy_expected_improvement",
        "qlognei": "log_noisy_expected_improvement",
        "qlog_noisy_expected_improvement": "log_noisy_expected_improvement",
        "ei": "expected_improvement",
        "expected_improvement": "expected_improvement",
        "nei": "noisy_expected_improvement",
        "noisy_ei": "noisy_expected_improvement",
        "noisy_expected_improvement": "noisy_expected_improvement",
        "q_nei": "noisy_expected_improvement",
        "qnoisyexpectedimprovement": "noisy_expected_improvement",
        "qnei": "noisy_expected_improvement",
        "q_noisy_expected_improvement": "noisy_expected_improvement",
        "pi": "probability_of_improvement",
        "probability_improvement": "probability_of_improvement",
        "probability_of_improvement": "probability_of_improvement",
        "ucb": "upper_confidence_bound",
        "upper_confidence_bound": "upper_confidence_bound",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported acquisition_function: {name!r}")
    return aliases[normalized]


def _normalize_multi_objective_acquisition_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ehvi": "q_expected_hypervolume_improvement",
        "qehvi": "q_expected_hypervolume_improvement",
        "q_ehvi": "q_expected_hypervolume_improvement",
        "expected_hypervolume_improvement": "q_expected_hypervolume_improvement",
        "q_expected_hypervolume_improvement": "q_expected_hypervolume_improvement",
        "nehvi": "q_noisy_expected_hypervolume_improvement",
        "qnehvi": "q_noisy_expected_hypervolume_improvement",
        "q_nehvi": "q_noisy_expected_hypervolume_improvement",
        "noisy_expected_hypervolume_improvement": (
            "q_noisy_expected_hypervolume_improvement"
        ),
        "q_noisy_expected_hypervolume_improvement": (
            "q_noisy_expected_hypervolume_improvement"
        ),
        "log_ehvi": "q_log_expected_hypervolume_improvement",
        "logehvi": "q_log_expected_hypervolume_improvement",
        "qlogehvi": "q_log_expected_hypervolume_improvement",
        "q_log_ehvi": "q_log_expected_hypervolume_improvement",
        "log_expected_hypervolume_improvement": (
            "q_log_expected_hypervolume_improvement"
        ),
        "q_log_expected_hypervolume_improvement": (
            "q_log_expected_hypervolume_improvement"
        ),
        "log_nehvi": "q_log_noisy_expected_hypervolume_improvement",
        "lognehvi": "q_log_noisy_expected_hypervolume_improvement",
        "qlognehvi": "q_log_noisy_expected_hypervolume_improvement",
        "q_log_nehvi": "q_log_noisy_expected_hypervolume_improvement",
        "log_noisy_expected_hypervolume_improvement": (
            "q_log_noisy_expected_hypervolume_improvement"
        ),
        "q_log_noisy_expected_hypervolume_improvement": (
            "q_log_noisy_expected_hypervolume_improvement"
        ),
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported multi-objective acquisition_function: {name!r}")
    return aliases[normalized]


def _validate_constraint_policy(policy: str) -> str:
    normalized = policy.strip().lower()
    if normalized not in {"filter", "annotate"}:
        raise ValueError("constraint_policy must be 'filter' or 'annotate'")
    return normalized


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


def _apply_candidate_domain_policies(
    candidates: pd.DataFrame,
    candidate_features: pd.DataFrame | np.ndarray,
    *,
    trust_region: CandidateTrustRegion | None,
    trust_region_policy: str,
    duplicate_policy: CandidateDuplicatePolicy | None,
    duplicate_policy_action: str,
) -> pd.DataFrame:
    result, _ = _apply_candidate_domain_policies_with_mask(
        candidates,
        candidate_features,
        trust_region=trust_region,
        trust_region_policy=trust_region_policy,
        duplicate_policy=duplicate_policy,
        duplicate_policy_action=duplicate_policy_action,
    )
    return result


def _apply_candidate_domain_policies_with_mask(
    candidates: pd.DataFrame,
    candidate_features: pd.DataFrame | np.ndarray,
    *,
    trust_region: CandidateTrustRegion | None,
    trust_region_policy: str,
    duplicate_policy: CandidateDuplicatePolicy | None,
    duplicate_policy_action: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")

    trust_region_policy = _validate_candidate_policy(trust_region_policy)
    duplicate_policy_action = _validate_candidate_policy(duplicate_policy_action)
    result = candidates.copy()
    keep_mask = np.ones(result.shape[0], dtype=bool)

    if trust_region is not None:
        if not isinstance(trust_region, CandidateTrustRegion):
            raise TypeError("trust_region must be a CandidateTrustRegion")
        distances, in_region = _trust_region_annotations(
            candidate_features,
            trust_region,
            expected_rows=result.shape[0],
        )
        result["matgpr_trust_region_distance"] = distances
        result["matgpr_in_trust_region"] = in_region
        if trust_region_policy == "filter":
            keep_mask &= in_region

    if duplicate_policy is not None:
        if not isinstance(duplicate_policy, CandidateDuplicatePolicy):
            raise TypeError("duplicate_policy must be a CandidateDuplicatePolicy")
        is_duplicate, distances, reasons = _duplicate_policy_annotations(
            result,
            candidate_features,
            duplicate_policy,
            expected_rows=result.shape[0],
        )
        result["matgpr_is_duplicate"] = is_duplicate
        result["matgpr_duplicate_reason"] = reasons
        if distances is not None:
            result["matgpr_duplicate_distance"] = distances
        if duplicate_policy_action == "filter":
            keep_mask &= ~is_duplicate

    return result.loc[keep_mask].reset_index(drop=True), keep_mask


def _trust_region_annotations(
    candidate_features: pd.DataFrame | np.ndarray,
    trust_region: CandidateTrustRegion,
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
    duplicate_policy: CandidateDuplicatePolicy,
    *,
    expected_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    is_duplicate = np.zeros(candidates.shape[0], dtype=bool)
    reason_parts: list[list[str]] = [[] for _ in range(candidates.shape[0])]
    distances = None

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
        existing_keys = set(
            _candidate_policy_row_keys(
                duplicate_policy.existing_candidates,
                duplicate_policy.key_columns,
                label="existing key columns",
            )
        )
        key_duplicates = np.asarray([key in existing_keys for key in candidate_keys], dtype=bool)
        is_duplicate |= key_duplicates
        for row_index, duplicate in enumerate(key_duplicates):
            if duplicate:
                reason_parts[row_index].append("key")

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
        distances = _minimum_scaled_distances(
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

    reasons = np.asarray(["; ".join(parts) for parts in reason_parts], dtype=object)
    return is_duplicate, distances, reasons


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
        if feature_columns is not None and matrix.ndim == 2 and len(feature_columns) != matrix.shape[1]:
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
            f"{name} rows must match candidate rows "
            f"({matrix.shape[0]} != {expected_rows})"
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
            requested
            for requested, resolved in zip(feature_columns, columns)
            if resolved is None
        ]
        if missing:
            raise ValueError(f"{name} feature columns are missing: {missing}")
    else:
        columns = tuple(
            column
            for column in data.columns
            if pd.api.types.is_numeric_dtype(data[column])
        )
        if not columns:
            raise ValueError(
                f"No numeric columns could be inferred from {name}; provide feature_columns"
            )

    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
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
                f"{name} length must match feature count "
                f"({scales.shape[0]} != {n_features})"
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


def _validate_batch_selection(
    batch_selection: str,
    *,
    allow_sequential: bool = False,
) -> str:
    normalized = batch_selection.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "top": "top",
        "diverse": "diverse",
    }
    if allow_sequential:
        aliases.update(
            {
                "greedy": "sequential",
                "sequential": "sequential",
                "sequential_hypervolume": "sequential",
                "greedy_hypervolume": "sequential",
            }
        )
    if normalized not in aliases:
        allowed = "'top', 'diverse', or 'sequential'" if allow_sequential else "'top' or 'diverse'"
        raise ValueError(f"batch_selection must be {allowed}")
    return aliases[normalized]


def _validate_mc_samples(mc_samples: int) -> int:
    if isinstance(mc_samples, bool) or not isinstance(mc_samples, Integral):
        raise TypeError("mc_samples must be an integer")
    mc_samples = int(mc_samples)
    if mc_samples < 1:
        raise ValueError("mc_samples must be at least 1")
    return mc_samples


def _as_constraint_tuple(
    constraints: CandidateConstraint | list[CandidateConstraint] | tuple[CandidateConstraint, ...],
) -> tuple[CandidateConstraint, ...]:
    if isinstance(constraints, CandidateConstraint):
        return (constraints,)
    try:
        constraint_tuple = tuple(constraints)
    except TypeError as exc:
        raise TypeError(
            "constraints must be a CandidateConstraint or an iterable of "
            "CandidateConstraint objects"
        ) from exc
    invalid = [
        type(constraint).__name__
        for constraint in constraint_tuple
        if not isinstance(constraint, CandidateConstraint)
    ]
    if invalid:
        raise TypeError(
            "constraints must contain only CandidateConstraint objects; "
            f"got {invalid}"
        )
    return constraint_tuple


def _numeric_column(data: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    if column not in data.columns:
        raise ValueError(f"{label} {column!r} is missing")
    numeric = pd.to_numeric(data[column], errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} {column!r} contains missing, non-numeric, or infinite values")
    return pd.Series(values, index=data.index, name=column)


def _nonnegative_numeric_column(data: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    numeric = _numeric_column(data, column, label=label)
    if (numeric < 0.0).any():
        raise ValueError(f"{label} {column!r} must contain non-negative values")
    return numeric


def _replicate_group_variance(
    data: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    min_variance: float,
    ddof: int,
) -> pd.Series:
    if group_column not in data.columns:
        raise ValueError(f"replicate_group_column {group_column!r} is missing")
    if data[group_column].isna().any():
        raise ValueError(f"replicate_group_column {group_column!r} contains missing values")

    target = _numeric_column(data, target_column, label="target_column")
    frame = pd.DataFrame(
        {
            "target": target.to_numpy(dtype=float),
            "group": data[group_column].to_numpy(dtype=object),
        }
    )
    variance_values = np.full(data.shape[0], np.nan, dtype=float)
    replicate_variances: list[float] = []

    for _, group_frame in frame.groupby("group", sort=False, dropna=False):
        values = group_frame["target"].to_numpy(dtype=float)
        if values.shape[0] > ddof:
            group_variance = float(np.var(values, ddof=ddof))
            if np.isfinite(group_variance):
                replicate_variances.append(group_variance)
        else:
            group_variance = np.nan
        variance_values[group_frame.index.to_numpy(dtype=int)] = group_variance

    fallback_variance = (
        float(np.mean(replicate_variances)) if replicate_variances else min_variance
    )
    return pd.Series(variance_values, index=data.index, dtype=float).fillna(fallback_variance)


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


def _as_numeric_matrix(
    values: pd.DataFrame | np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, tuple[str, ...] | None, np.ndarray]:
    if isinstance(values, pd.DataFrame):
        non_numeric = [
            str(column)
            for column in values.columns
            if not pd.api.types.is_numeric_dtype(values[column])
        ]
        if non_numeric:
            raise ValueError(
                f"{name} must contain only numeric columns; non-numeric columns: "
                f"{non_numeric}"
            )
        array = values.to_numpy(dtype=float)
        feature_names = tuple(str(column) for column in values.columns)
        index = values.index.to_numpy()
    else:
        array = np.asarray(values, dtype=float)
        feature_names = None
        if array.ndim >= 1:
            index = np.arange(array.shape[0])
        else:
            index = np.array([], dtype=int)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D numeric array or dataframe")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array, feature_names, index


def _as_target_vector(
    values: pd.Series | np.ndarray,
    *,
    expected_length: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] != expected_length:
        raise ValueError(
            f"{name} length must match X_train rows "
            f"({array.shape[0]} != {expected_length})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_target_matrix(
    values: pd.DataFrame | np.ndarray,
    *,
    expected_length: int,
    name: str,
) -> tuple[np.ndarray, tuple[str, ...] | None]:
    if isinstance(values, pd.DataFrame):
        non_numeric = [
            str(column)
            for column in values.columns
            if not pd.api.types.is_numeric_dtype(values[column])
        ]
        if non_numeric:
            raise ValueError(
                f"{name} must contain only numeric columns; non-numeric columns: "
                f"{non_numeric}"
            )
        array = values.to_numpy(dtype=float)
        objective_names = tuple(str(column) for column in values.columns)
    else:
        array = np.asarray(values, dtype=float)
        objective_names = None

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D numeric array or dataframe")
    if array.shape[0] != expected_length:
        raise ValueError(
            f"{name} rows must match X_train rows "
            f"({array.shape[0]} != {expected_length})"
        )
    if array.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two objectives")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array, objective_names


def _resolve_objective_names(
    objective_names: Sequence[str] | None,
    *,
    inferred_names: tuple[str, ...] | None,
    n_objectives: int,
) -> tuple[str, ...]:
    if objective_names is None:
        if inferred_names is not None:
            names = inferred_names
        else:
            names = tuple(f"objective_{index}" for index in range(n_objectives))
    else:
        names = tuple(str(name).strip() for name in objective_names)

    if len(names) != n_objectives:
        raise ValueError(
            "objective_names length must match number of objectives "
            f"({len(names)} != {n_objectives})"
        )
    if any(not name for name in names):
        raise ValueError("objective_names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("objective_names must be unique")
    slugs = [_slugify_objective_name(name) for name in names]
    if len(set(slugs)) != len(slugs):
        raise ValueError("objective_names must produce unique output column names")
    return names


def _resolve_objective_directions(
    objective_directions: str | Sequence[str],
    *,
    n_objectives: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(objective_directions, str):
        directions = tuple([_normalize_objective_direction(objective_directions)] * n_objectives)
    else:
        directions = tuple(_normalize_objective_direction(direction) for direction in objective_directions)

    if len(directions) != n_objectives:
        raise ValueError(
            "objective_directions length must match number of objectives "
            f"({len(directions)} != {n_objectives})"
        )
    signs = np.asarray([1.0 if direction == "maximize" else -1.0 for direction in directions])
    return directions, signs


def _normalize_objective_direction(direction: str) -> str:
    normalized = str(direction).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "max": "maximize",
        "maximize": "maximize",
        "higher": "maximize",
        "higher_is_better": "maximize",
        "min": "minimize",
        "minimize": "minimize",
        "lower": "minimize",
        "lower_is_better": "minimize",
    }
    if normalized not in aliases:
        raise ValueError("objective_directions must contain 'maximize' or 'minimize'")
    return aliases[normalized]


def _resolve_reference_point(
    reference_point: Sequence[float] | np.ndarray | pd.Series | None,
    *,
    objective_array: np.ndarray,
    signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if reference_point is None:
        minima = np.min(objective_array, axis=0)
        ranges = np.ptp(objective_array, axis=0)
        fallback = np.maximum(np.abs(minima) * 0.01, 1e-6)
        margins = np.where(ranges > 0.0, 0.1 * ranges, fallback)
        reference_objective = minima - margins
        reference_original = reference_objective * signs
    else:
        reference_original = np.asarray(reference_point, dtype=float).reshape(-1)
        if reference_original.shape[0] != objective_array.shape[1]:
            raise ValueError(
                "reference_point length must match number of objectives "
                f"({reference_original.shape[0]} != {objective_array.shape[1]})"
            )
        if not np.all(np.isfinite(reference_original)):
            raise ValueError("reference_point contains NaN or infinite values")
        reference_objective = reference_original * signs

    if np.any(reference_objective >= np.max(objective_array, axis=0)):
        raise ValueError(
            "reference_point must be worse than at least one observed value for "
            "every objective"
        )
    return reference_objective.astype(float), reference_original.astype(float)


def _as_noise_variance(
    value: float | np.ndarray | pd.Series,
    *,
    expected_length: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(expected_length, float(array))
    else:
        array = array.reshape(-1)
    if array.shape[0] != expected_length:
        raise ValueError(
            "noise_variance length must match X_train rows "
            f"({array.shape[0]} != {expected_length})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("noise_variance contains NaN or infinite values")
    if np.any(array < 0.0):
        raise ValueError("noise_variance must contain non-negative variances")
    return array


def _as_noise_variance_matrix(
    value: float | np.ndarray | pd.DataFrame,
    *,
    expected_length: int,
    n_objectives: int,
) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        array = value.to_numpy(dtype=float)
    else:
        array = np.asarray(value, dtype=float)

    if array.ndim == 0:
        array = np.full((expected_length, n_objectives), float(array))
    elif array.ndim == 1:
        array = array.reshape(-1)
        if array.shape[0] != expected_length:
            raise ValueError(
                "1D noise_variance length must match X_train rows "
                f"({array.shape[0]} != {expected_length})"
            )
        array = np.repeat(array.reshape(-1, 1), n_objectives, axis=1)
    elif array.ndim == 2:
        if array.shape != (expected_length, n_objectives):
            raise ValueError(
                "2D noise_variance shape must match "
                f"({expected_length}, {n_objectives}); got {array.shape}"
            )
    else:
        raise ValueError("noise_variance must be a scalar, vector, or 2D matrix")

    if not np.all(np.isfinite(array)):
        raise ValueError("noise_variance contains NaN or infinite values")
    if np.any(array < 0.0):
        raise ValueError("noise_variance must contain non-negative variances")
    return array


def _to_tensor(torch_module: Any, array: np.ndarray, *, device: Any | None = None) -> Any:
    return torch_module.as_tensor(array, dtype=torch_module.double, device=device)


def _slugify_objective_name(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "objective"


def _candidate_frame(
    *,
    X_candidates: pd.DataFrame | np.ndarray,
    candidate_data: pd.DataFrame | None,
    candidate_index: np.ndarray,
) -> pd.DataFrame:
    if candidate_data is not None:
        if candidate_data.shape[0] != candidate_index.shape[0]:
            raise ValueError(
                "candidate_data rows must match X_candidates rows "
                f"({candidate_data.shape[0]} != {candidate_index.shape[0]})"
            )
        frame = candidate_data.reset_index(drop=True).copy()
    elif isinstance(X_candidates, pd.DataFrame):
        frame = X_candidates.reset_index(drop=True).copy()
    else:
        frame = pd.DataFrame()

    frame = frame.drop(columns=[column for column in _OUTPUT_COLUMNS if column in frame.columns])
    frame.insert(0, "matgpr_candidate_index", candidate_index)
    return frame
