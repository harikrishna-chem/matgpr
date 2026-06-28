from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .optional_dependencies import require_optional_dependency

_OUTPUT_COLUMNS = (
    "matgpr_candidate_index",
    "matgpr_rank",
    "matgpr_predicted_mean",
    "matgpr_predicted_std",
    "matgpr_acquisition",
)

__all__ = [
    "BayesianOptimizationResult",
    "BoTorchSurrogate",
    "fit_botorch_surrogate",
    "rank_discrete_candidates",
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
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided")

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
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

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
    )
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
