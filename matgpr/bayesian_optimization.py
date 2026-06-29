from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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
    "apply_candidate_constraints",
    "fit_botorch_surrogate",
    "rank_discrete_candidates",
    "select_diverse_batch",
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
    """

    model: Any
    train_X: Any
    train_y: Any
    objective_y: Any
    maximize: bool
    best_observed_objective: float
    feature_names: tuple[str, ...] | None = None


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
    if noise_variance is not None:
        noise_array = _as_noise_variance(
            noise_variance,
            expected_length=train_array.shape[0],
        )
        model_kwargs["train_Yvar"] = _to_tensor(
            torch,
            noise_array.reshape(-1, 1),
            device=torch_device,
        )
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
        `"log_expected_improvement"`, `"expected_improvement"`,
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
            ranked = ranked[ranked["matgpr_feasible"]].copy()
    ranked = ranked.sort_values("matgpr_acquisition", ascending=False).reset_index(drop=True)
    ranked.insert(0, "matgpr_rank", np.arange(1, ranked.shape[0] + 1))

    if top_k is not None:
        return ranked.head(top_k).reset_index(drop=True)
    return ranked


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
    if normalized == "expected_improvement":
        from botorch.acquisition.analytic import ExpectedImprovement

        return ExpectedImprovement(
            model=surrogate.model,
            best_f=surrogate.best_observed_objective,
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
        "'log_expected_improvement', 'expected_improvement', "
        "'probability_of_improvement', or 'upper_confidence_bound'"
    )


def _normalize_acquisition_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "logei": "log_expected_improvement",
        "log_ei": "log_expected_improvement",
        "log_expected_improvement": "log_expected_improvement",
        "ei": "expected_improvement",
        "expected_improvement": "expected_improvement",
        "pi": "probability_of_improvement",
        "probability_improvement": "probability_of_improvement",
        "probability_of_improvement": "probability_of_improvement",
        "ucb": "upper_confidence_bound",
        "upper_confidence_bound": "upper_confidence_bound",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported acquisition_function: {name!r}")
    return aliases[normalized]


def _validate_constraint_policy(policy: str) -> str:
    normalized = policy.strip().lower()
    if normalized not in {"filter", "annotate"}:
        raise ValueError("constraint_policy must be 'filter' or 'annotate'")
    return normalized


def _validate_batch_selection(batch_selection: str) -> str:
    normalized = batch_selection.strip().lower()
    if normalized not in {"top", "diverse"}:
        raise ValueError("batch_selection must be 'top' or 'diverse'")
    return normalized


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


def _to_tensor(torch_module: Any, array: np.ndarray, *, device: Any | None = None) -> Any:
    return torch_module.as_tensor(array, dtype=torch_module.double, device=device)


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
