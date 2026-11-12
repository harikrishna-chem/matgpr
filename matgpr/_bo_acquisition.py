from __future__ import annotations

from typing import Any, Protocol

__all__: list[str] = []


class _SingleObjectiveSurrogateLike(Protocol):
    model: Any
    train_X: Any
    best_observed_objective: float


class _MultiObjectiveSurrogateLike(Protocol):
    model: Any
    train_X: Any
    objective_y: Any
    reference_point: Any


def _build_acquisition_function(
    name: str,
    *,
    surrogate: _SingleObjectiveSurrogateLike,
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
    surrogate: _MultiObjectiveSurrogateLike,
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
        "noisy_expected_hypervolume_improvement": ("q_noisy_expected_hypervolume_improvement"),
        "q_noisy_expected_hypervolume_improvement": ("q_noisy_expected_hypervolume_improvement"),
        "log_ehvi": "q_log_expected_hypervolume_improvement",
        "logehvi": "q_log_expected_hypervolume_improvement",
        "qlogehvi": "q_log_expected_hypervolume_improvement",
        "q_log_ehvi": "q_log_expected_hypervolume_improvement",
        "log_expected_hypervolume_improvement": ("q_log_expected_hypervolume_improvement"),
        "q_log_expected_hypervolume_improvement": ("q_log_expected_hypervolume_improvement"),
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
