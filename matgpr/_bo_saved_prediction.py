"""Deterministic acquisition policy for saved BO predictions.

This private module owns acquisition-intent resolution and the numerical
helpers used when candidate posterior means and standard deviations already
exist. Dataframe assembly, generated columns, stable ranking, and the public
API remain in :mod:`matgpr.bayesian_optimization`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .bayesian_optimization_specs import (
    BOAcquisitionIntent,
    BOObjectiveMode,
    BOObjectivePreference,
    BOOutputConstraintSpec,
)

__all__: list[str] = []

_SAVED_PREDICTION_ACQUISITIONS = {
    BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE,
    BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND,
    BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND,
    BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
    BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT,
    BOAcquisitionIntent.TARGET_PROBABILITY,
    BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
}
_UNCERTAINTY_REQUIRED_ACQUISITIONS = _SAVED_PREDICTION_ACQUISITIONS - {
    BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE,
}


def _resolve_saved_prediction_acquisition(
    value: BOAcquisitionIntent | str | None,
    objective: BOObjectivePreference,
    *,
    uncertainty_available: bool,
) -> tuple[BOAcquisitionIntent | None, BOAcquisitionIntent]:
    requested = None if value is None else BOAcquisitionIntent(value)
    if requested is None:
        if not uncertainty_available:
            effective = BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE
        elif objective.mode is BOObjectiveMode.MAXIMIZE:
            effective = BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND
        elif objective.mode is BOObjectiveMode.MINIMIZE:
            effective = BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND
        else:
            effective = BOAcquisitionIntent.TARGET_PROBABILITY
    else:
        effective = requested
    if effective not in _SAVED_PREDICTION_ACQUISITIONS:
        supported = ", ".join(sorted(item.value for item in _SAVED_PREDICTION_ACQUISITIONS))
        raise ValueError(
            f"acquisition intent {effective.value!r} is not supported for saved "
            f"prediction scoring; supported intents: {supported}"
        )
    if effective in _UNCERTAINTY_REQUIRED_ACQUISITIONS and not uncertainty_available:
        raise ValueError(
            f"predicted_std is required for acquisition intent {effective.value!r}; "
            "request 'posterior_mean_preference' explicitly for deterministic scoring"
        )
    if effective is BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND and (
        objective.mode is not BOObjectiveMode.MAXIMIZE
    ):
        raise ValueError("upper_confidence_bound requires a maximize objective")
    if effective is BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND and (
        objective.mode is not BOObjectiveMode.MINIMIZE
    ):
        raise ValueError("lower_confidence_bound requires a minimize objective")
    if effective is BOAcquisitionIntent.TARGET_PROBABILITY and objective.mode not in {
        BOObjectiveMode.TARGET_VALUE,
        BOObjectiveMode.TARGET_RANGE,
    }:
        raise ValueError("target_probability requires a target_value or target_range objective")
    if effective in {
        BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
        BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT,
        BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
    } and objective.mode not in {BOObjectiveMode.MAXIMIZE, BOObjectiveMode.MINIMIZE}:
        raise ValueError(f"{effective.value} requires a maximize or minimize objective")
    return requested, effective


def _saved_prediction_acquisition_values(
    mean: np.ndarray,
    std: np.ndarray,
    objective: BOObjectivePreference,
    *,
    acquisition_intent: BOAcquisitionIntent,
    beta: float,
    incumbent_value: float | None,
    improvement_threshold: float,
) -> np.ndarray:
    if acquisition_intent is BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE:
        return _posterior_mean_preference(mean, objective)
    if acquisition_intent is BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND:
        return mean + beta * std
    if acquisition_intent is BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND:
        return -(mean - beta * std)
    if acquisition_intent in {
        BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
        BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
    }:
        return _normal_expected_improvement(
            mean,
            std,
            objective=objective,
            incumbent_value=incumbent_value,
            improvement_threshold=improvement_threshold,
        )
    if acquisition_intent is BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT:
        return _normal_probability_of_improvement(
            mean,
            std,
            objective=objective,
            incumbent_value=incumbent_value,
            improvement_threshold=improvement_threshold,
        )
    if acquisition_intent is BOAcquisitionIntent.TARGET_PROBABILITY:
        lower, upper = _objective_event_interval(objective)
        return _normal_interval_probability(
            mean,
            std,
            lower_bound=lower,
            upper_bound=upper,
        )
    raise ValueError(f"Unsupported saved-prediction acquisition: {acquisition_intent.value}")


def _posterior_mean_preference(
    mean: np.ndarray,
    objective: BOObjectivePreference,
) -> np.ndarray:
    if objective.mode is BOObjectiveMode.MAXIMIZE:
        return mean.astype(float)
    if objective.mode is BOObjectiveMode.MINIMIZE:
        return -mean.astype(float)
    if objective.mode is BOObjectiveMode.TARGET_VALUE:
        return -np.abs(mean.astype(float) - float(objective.target_value))
    lower, upper = _objective_event_interval(objective)
    below = np.maximum(float(lower) - mean, 0.0)
    above = np.maximum(mean - float(upper), 0.0)
    return -(below + above)


def _normal_expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    objective: BOObjectivePreference,
    incumbent_value: float | None,
    improvement_threshold: float,
) -> np.ndarray:
    incumbent = _required_incumbent_value(incumbent_value)
    improvement = (
        mean - incumbent - improvement_threshold
        if objective.mode is BOObjectiveMode.MAXIMIZE
        else incumbent - mean - improvement_threshold
    )
    deterministic = std <= 1e-12
    values = np.maximum(improvement, 0.0)
    active = ~deterministic
    if np.any(active):
        z = improvement[active] / std[active]
        values[active] = improvement[active] * _standard_normal_cdf(z) + std[
            active
        ] * _standard_normal_pdf(z)
    return values


def _normal_probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    objective: BOObjectivePreference,
    incumbent_value: float | None,
    improvement_threshold: float,
) -> np.ndarray:
    incumbent = _required_incumbent_value(incumbent_value)
    improvement = (
        mean - incumbent - improvement_threshold
        if objective.mode is BOObjectiveMode.MAXIMIZE
        else incumbent - mean - improvement_threshold
    )
    deterministic = std <= 1e-12
    values = (improvement > 0.0).astype(float)
    active = ~deterministic
    if np.any(active):
        values[active] = _standard_normal_cdf(improvement[active] / std[active])
    return values


def _apply_saved_prediction_feasibility(
    acquisition_base: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    objective: BOObjectivePreference,
    constraints: Sequence[BOOutputConstraintSpec],
    acquisition_intent: BOAcquisitionIntent,
    incumbent_value: float | None,
    improvement_threshold: float,
    feasibility_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    base_name = (
        BOAcquisitionIntent.EXPECTED_IMPROVEMENT.value
        if acquisition_intent is BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT
        else acquisition_intent.value
    )
    if not constraints:
        return acquisition_base, acquisition_base, base_name

    if acquisition_intent in {
        BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT,
        BOAcquisitionIntent.TARGET_PROBABILITY,
    }:
        objective_interval = (
            _improvement_event_interval(
                objective,
                incumbent_value=incumbent_value,
                improvement_threshold=improvement_threshold,
            )
            if acquisition_intent is BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT
            else _objective_event_interval(objective)
        )
        constraint_interval = _constraint_event_interval(constraints)
        lower, upper = _intersect_intervals(objective_interval, constraint_interval)
        intersection_probability = _normal_interval_probability(
            mean,
            std,
            lower_bound=lower,
            upper_bound=upper,
        )
        return intersection_probability, acquisition_base, f"intersection_{base_name}"

    objective_component = (
        _unit_interval_scores(acquisition_base)
        if acquisition_intent
        in {
            BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE,
            BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND,
            BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND,
        }
        else acquisition_base
    )
    return (
        objective_component * feasibility_probability,
        objective_component,
        f"feasibility_weighted_{base_name}",
    )


def _objective_event_interval(
    objective: BOObjectivePreference,
) -> tuple[float | None, float | None]:
    if objective.mode is BOObjectiveMode.TARGET_RANGE:
        return objective.target_lower_bound, objective.target_upper_bound
    if objective.mode is BOObjectiveMode.TARGET_VALUE:
        tolerance = _objective_target_tolerance(objective)
        if tolerance is None or tolerance <= 0.0:
            raise ValueError(
                "target_probability with target_value requires a positive target_tolerance"
            )
        target = float(objective.target_value)
        return target - tolerance, target + tolerance
    raise ValueError("objective does not define a closed target event")


def _improvement_event_interval(
    objective: BOObjectivePreference,
    *,
    incumbent_value: float | None,
    improvement_threshold: float,
) -> tuple[float | None, float | None]:
    incumbent = _required_incumbent_value(incumbent_value)
    if objective.mode is BOObjectiveMode.MAXIMIZE:
        return incumbent + improvement_threshold, None
    return None, incumbent - improvement_threshold


def _constraint_event_interval(
    constraints: Sequence[BOOutputConstraintSpec],
) -> tuple[float | None, float | None]:
    lower_values = [float(item.lower_bound) for item in constraints if item.lower_bound is not None]
    upper_values = [float(item.upper_bound) for item in constraints if item.upper_bound is not None]
    return (
        max(lower_values) if lower_values else None,
        min(upper_values) if upper_values else None,
    )


def _intersect_intervals(
    first: tuple[float | None, float | None],
    second: tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    lower_values = [value for value in (first[0], second[0]) if value is not None]
    upper_values = [value for value in (first[1], second[1]) if value is not None]
    lower = max(lower_values) if lower_values else None
    upper = min(upper_values) if upper_values else None
    if lower is not None and upper is not None and lower > upper:
        return 1.0, 0.0
    return lower, upper


def _required_incumbent_value(value: float | None) -> float:
    if value is None:
        raise ValueError("incumbent_value is required for EI/PI acquisition")
    incumbent = float(value)
    if not np.isfinite(incumbent):
        raise ValueError("incumbent_value must be finite")
    return incumbent


def _normal_interval_probability(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
    inclusive: bool = True,
    lower_inclusive: bool | None = None,
    upper_inclusive: bool | None = None,
) -> np.ndarray:
    if lower_bound is None and upper_bound is None:
        return np.ones(mean.shape[0], dtype=float)

    mean = mean.astype(float)
    std = np.maximum(std.astype(float), 0.0)
    lower_closed = inclusive if lower_inclusive is None else lower_inclusive
    upper_closed = inclusive if upper_inclusive is None else upper_inclusive
    deterministic = std <= 1e-12
    probability = np.zeros(mean.shape[0], dtype=float)

    if np.any(~deterministic):
        active_mean = mean[~deterministic]
        active_std = std[~deterministic]
        lower_probability = (
            0.0
            if lower_bound is None
            else _standard_normal_cdf((float(lower_bound) - active_mean) / active_std)
        )
        upper_probability = (
            1.0
            if upper_bound is None
            else _standard_normal_cdf((float(upper_bound) - active_mean) / active_std)
        )
        probability[~deterministic] = np.clip(
            upper_probability - lower_probability,
            0.0,
            1.0,
        )

    if np.any(deterministic):
        lower_ok = (
            np.ones(np.count_nonzero(deterministic), dtype=bool)
            if lower_bound is None
            else (
                mean[deterministic] >= float(lower_bound)
                if lower_closed
                else mean[deterministic] > float(lower_bound)
            )
        )
        upper_ok = (
            np.ones(np.count_nonzero(deterministic), dtype=bool)
            if upper_bound is None
            else (
                mean[deterministic] <= float(upper_bound)
                if upper_closed
                else mean[deterministic] < float(upper_bound)
            )
        )
        probability[deterministic] = (lower_ok & upper_ok).astype(float)
    return probability


def _joint_output_constraint_probability(
    mean: np.ndarray,
    std: np.ndarray,
    constraints: Sequence[BOOutputConstraintSpec],
) -> np.ndarray:
    lower_bounds = [
        float(constraint.lower_bound)
        for constraint in constraints
        if constraint.lower_bound is not None
    ]
    upper_bounds = [
        float(constraint.upper_bound)
        for constraint in constraints
        if constraint.upper_bound is not None
    ]
    lower_bound = max(lower_bounds) if lower_bounds else None
    upper_bound = min(upper_bounds) if upper_bounds else None
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        return np.zeros(mean.shape[0], dtype=float)

    lower_inclusive = all(
        constraint.inclusive
        for constraint in constraints
        if constraint.lower_bound is not None and float(constraint.lower_bound) == lower_bound
    )
    upper_inclusive = all(
        constraint.inclusive
        for constraint in constraints
        if constraint.upper_bound is not None and float(constraint.upper_bound) == upper_bound
    )
    return _normal_interval_probability(
        mean,
        std,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_inclusive=lower_inclusive,
        upper_inclusive=upper_inclusive,
    )


def _normal_target_closeness(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    target: float,
) -> np.ndarray:
    mean = mean.astype(float)
    std = np.maximum(std.astype(float), 0.0)
    deterministic = std <= 1e-12
    score = np.zeros(mean.shape[0], dtype=float)
    if np.any(~deterministic):
        z = (mean[~deterministic] - target) / std[~deterministic]
        score[~deterministic] = np.exp(-0.5 * z**2)
    if np.any(deterministic):
        score[deterministic] = np.isclose(mean[deterministic], target).astype(float)
    return score


def _standard_normal_cdf(values: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(values)


def _standard_normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values**2) / np.sqrt(2.0 * np.pi)


def _unit_interval_scores(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if np.isclose(min_value, max_value):
        return np.ones_like(values, dtype=float)
    scaled = (values - min_value) / (max_value - min_value)
    return np.clip(scaled, 0.0, 1.0)


def _objective_target_tolerance(objective: BOObjectivePreference) -> float | None:
    if objective.target_tolerance is not None:
        return float(objective.target_tolerance)
    for key in ("tolerance", "target_tolerance", "absolute_tolerance"):
        if key in objective.metadata:
            tolerance = float(objective.metadata[key])
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("objective target tolerance must be finite and non-negative")
            return tolerance
    return None


def _validate_single_target_output_constraints(
    objective: BOObjectivePreference,
    constraints: Sequence[BOOutputConstraintSpec],
) -> None:
    mismatched = [
        constraint.target_name
        for constraint in constraints
        if constraint.target_name != objective.target_name
    ]
    if mismatched:
        raise ValueError(
            "single-target BO preference scoring only supports output constraints "
            f"on the modeled objective {objective.target_name!r}; got {mismatched}"
        )
