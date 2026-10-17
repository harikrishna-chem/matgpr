"""Reusable Bayesian-optimization intent and search-space specifications.

This module defines JSON-friendly records for describing materials BO problems
before a concrete optimizer is selected. The records are intentionally light:
they capture objectives, controllable variables, output constraints, and the
recommended acquisition-function families without fitting a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BOAcquisitionIntent",
    "BOFeatureSpec",
    "BOFeatureType",
    "BOObjectiveMode",
    "BOObjectivePreference",
    "BOOptimizationMode",
    "BOOptimizationSpec",
    "BOOutputConstraintSpec",
    "recommend_bo_acquisition_intents",
    "recommend_saved_prediction_acquisition_intents",
]


class BOOptimizationMode(str, Enum):
    """High-level BO workflow requested by the user."""

    FINITE_POOL_RANKING = "finite_pool_ranking"
    FEATURE_OPTIMIZATION = "feature_optimization"


class BOFeatureType(str, Enum):
    """Type of a controllable input variable."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    FIXED = "fixed"


class BOObjectiveMode(str, Enum):
    """How the target output should be optimized."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET_VALUE = "target_value"
    TARGET_RANGE = "target_range"


class BOAcquisitionIntent(str, Enum):
    """Acquisition-function family suitable for a BO request."""

    POSTERIOR_MEAN_PREFERENCE = "posterior_mean_preference"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    NOISY_EXPECTED_IMPROVEMENT = "noisy_expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    LOWER_CONFIDENCE_BOUND = "lower_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    TARGET_PROBABILITY = "target_probability"
    CONSTRAINED_EXPECTED_IMPROVEMENT = "constrained_expected_improvement"
    CONSTRAINED_NOISY_EXPECTED_IMPROVEMENT = "constrained_noisy_expected_improvement"
    MULTI_OBJECTIVE_EHVI = "multi_objective_expected_hypervolume_improvement"
    SCALARIZED_EXPECTED_IMPROVEMENT = "scalarized_expected_improvement"
    THOMPSON_SAMPLING = "thompson_sampling"
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    COST_AWARE_EXPECTED_IMPROVEMENT = "cost_aware_expected_improvement"


@dataclass(frozen=True, slots=True)
class BOFeatureSpec:
    """Specification for one controllable or fixed BO input feature.

    Parameters
    ----------
    name
        Input feature or process variable name.
    variable_type
        Whether the feature is continuous, integer, categorical, or fixed.
    lower_bound, upper_bound
        Numeric bounds for continuous and integer variables.
    values
        Allowed category values for categorical variables.
    fixed_value
        Value for fixed context variables that should not be optimized.
    units
        Optional units shown to users and persisted in audit records.
    transform
        Optional transform name such as ``"log"`` or ``"standardized"``.
    metadata
        Extra JSON-safe information for applications.
    """

    name: str
    variable_type: BOFeatureType | str = BOFeatureType.CONTINUOUS
    lower_bound: float | None = None
    upper_bound: float | None = None
    values: Sequence[Any] | None = None
    fixed_value: Any | None = None
    units: str | None = None
    transform: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        variable_type = _coerce_enum(BOFeatureType, self.variable_type)
        lower_bound = _optional_finite_float(self.lower_bound, "lower_bound")
        upper_bound = _optional_finite_float(self.upper_bound, "upper_bound")
        values = None if self.values is None else tuple(self.values)

        if lower_bound is not None and upper_bound is not None:
            if lower_bound > upper_bound:
                raise ValueError("lower_bound must be <= upper_bound")

        if variable_type in {BOFeatureType.CONTINUOUS, BOFeatureType.INTEGER}:
            if lower_bound is None or upper_bound is None:
                raise ValueError(
                    f"{variable_type.value} feature requires lower_bound and upper_bound"
                )
            if variable_type is BOFeatureType.INTEGER and (
                not lower_bound.is_integer() or not upper_bound.is_integer()
            ):
                raise ValueError("integer feature bounds must be integer-valued")
        elif variable_type is BOFeatureType.CATEGORICAL:
            if not values:
                raise ValueError("categorical feature requires non-empty values")
        elif variable_type is BOFeatureType.FIXED and self.fixed_value is None:
            raise ValueError("fixed feature requires fixed_value")

        object.__setattr__(self, "name", _clean_text(self.name, "name"))
        object.__setattr__(self, "variable_type", variable_type)
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "units", _clean_optional_text(self.units))
        object.__setattr__(self, "transform", _clean_optional_text(self.transform))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_controllable(self) -> bool:
        """Whether this feature should be optimized by the BO routine."""

        return self.variable_type is not BOFeatureType.FIXED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe feature specification."""

        return {
            "name": self.name,
            "variable_type": self.variable_type.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "values": _json_safe(self.values),
            "fixed_value": _json_safe(self.fixed_value),
            "units": self.units,
            "transform": self.transform,
            "is_controllable": self.is_controllable,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BOObjectivePreference:
    """Target objective preference for a BO campaign.

    Use ``TARGET_VALUE`` or ``TARGET_RANGE`` for process-development settings
    where the best experiment is near a specification window rather than at an
    extreme value. ``target_tolerance`` is an optional positive half-width
    around ``target_value`` in the declared target units.
    """

    target_name: str
    mode: BOObjectiveMode | str = BOObjectiveMode.MAXIMIZE
    target_value: float | None = None
    target_tolerance: float | None = None
    target_lower_bound: float | None = None
    target_upper_bound: float | None = None
    weight: float = 1.0
    units: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = _coerce_enum(BOObjectiveMode, self.mode)
        target_value = _optional_finite_float(self.target_value, "target_value")
        target_tolerance = _optional_finite_float(
            self.target_tolerance,
            "target_tolerance",
        )
        target_lower_bound = _optional_finite_float(
            self.target_lower_bound,
            "target_lower_bound",
        )
        target_upper_bound = _optional_finite_float(
            self.target_upper_bound,
            "target_upper_bound",
        )
        if target_lower_bound is not None and target_upper_bound is not None:
            if target_lower_bound > target_upper_bound:
                raise ValueError("target_lower_bound must be <= target_upper_bound")
        if mode is BOObjectiveMode.TARGET_VALUE and target_value is None:
            raise ValueError("target_value mode requires target_value")
        if target_tolerance is not None:
            if mode is not BOObjectiveMode.TARGET_VALUE:
                raise ValueError("target_tolerance is only valid for target_value objectives")
            if target_tolerance <= 0.0:
                raise ValueError("target_tolerance must be finite and positive")
        if mode is BOObjectiveMode.TARGET_RANGE and (
            target_lower_bound is None or target_upper_bound is None
        ):
            raise ValueError("target_range mode requires target_lower_bound and target_upper_bound")

        weight = _positive_finite_float(self.weight, "weight")
        object.__setattr__(self, "target_name", _clean_text(self.target_name))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "target_value", target_value)
        object.__setattr__(self, "target_tolerance", target_tolerance)
        object.__setattr__(self, "target_lower_bound", target_lower_bound)
        object.__setattr__(self, "target_upper_bound", target_upper_bound)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "units", _clean_optional_text(self.units))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe objective preference."""

        return {
            "target_name": self.target_name,
            "mode": self.mode.value,
            "target_value": self.target_value,
            "target_tolerance": self.target_tolerance,
            "target_lower_bound": self.target_lower_bound,
            "target_upper_bound": self.target_upper_bound,
            "weight": self.weight,
            "units": self.units,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BOOutputConstraintSpec:
    """Constraint on a predicted output or measured property."""

    target_name: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    probability_threshold: float | None = None
    inclusive: bool = True
    units: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lower_bound = _optional_finite_float(self.lower_bound, "lower_bound")
        upper_bound = _optional_finite_float(self.upper_bound, "upper_bound")
        if lower_bound is None and upper_bound is None:
            raise ValueError("output constraint requires at least one bound")
        if lower_bound is not None and upper_bound is not None:
            if lower_bound > upper_bound:
                raise ValueError("lower_bound must be <= upper_bound")
        probability_threshold = _optional_unit_interval_float(
            self.probability_threshold,
            "probability_threshold",
        )
        object.__setattr__(self, "target_name", _clean_text(self.target_name))
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)
        object.__setattr__(self, "probability_threshold", probability_threshold)
        object.__setattr__(self, "inclusive", bool(self.inclusive))
        object.__setattr__(self, "units", _clean_optional_text(self.units))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe output-constraint specification."""

        return {
            "target_name": self.target_name,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "probability_threshold": self.probability_threshold,
            "inclusive": self.inclusive,
            "units": self.units,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BOOptimizationSpec:
    """Complete app-neutral specification for a BO request."""

    mode: BOOptimizationMode | str
    objectives: Sequence[BOObjectivePreference | Mapping[str, Any]]
    features: Sequence[BOFeatureSpec | Mapping[str, Any]] = ()
    output_constraints: Sequence[BOOutputConstraintSpec | Mapping[str, Any]] = ()
    acquisition_intent: BOAcquisitionIntent | str | None = None
    batch_size: int = 1
    random_seed: int | None = None
    cost_column: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = _coerce_enum(BOOptimizationMode, self.mode)
        objectives = tuple(_coerce_objective(item) for item in self.objectives)
        features = tuple(_coerce_feature(item) for item in self.features)
        output_constraints = tuple(
            _coerce_output_constraint(item) for item in self.output_constraints
        )
        if not objectives:
            raise ValueError("BOOptimizationSpec requires at least one objective")
        duplicate_objectives = _duplicate_names(objective.target_name for objective in objectives)
        if duplicate_objectives:
            raise ValueError(
                f"objective target names must be unique; duplicates: {duplicate_objectives}"
            )
        duplicate_features = _duplicate_names(feature.name for feature in features)
        if duplicate_features:
            raise ValueError(f"feature names must be unique; duplicates: {duplicate_features}")
        if mode is BOOptimizationMode.FEATURE_OPTIMIZATION:
            if not any(feature.is_controllable for feature in features):
                raise ValueError("feature_optimization requires at least one controllable feature")
        acquisition_intent = (
            None
            if self.acquisition_intent is None
            else _coerce_enum(BOAcquisitionIntent, self.acquisition_intent)
        )
        batch_size = _positive_int(self.batch_size, "batch_size")
        random_seed = None if self.random_seed is None else int(self.random_seed)

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "output_constraints", output_constraints)
        object.__setattr__(self, "acquisition_intent", acquisition_intent)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(
            self,
            "cost_column",
            _clean_optional_text(self.cost_column),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Objective target names in request order."""

        return tuple(objective.target_name for objective in self.objectives)

    @property
    def controllable_feature_names(self) -> tuple[str, ...]:
        """Input features that are allowed to vary during optimization."""

        return tuple(feature.name for feature in self.features if feature.is_controllable)

    @property
    def fixed_feature_names(self) -> tuple[str, ...]:
        """Input features held fixed during optimization."""

        return tuple(
            feature.name
            for feature in self.features
            if feature.variable_type is BOFeatureType.FIXED
        )

    def recommended_acquisition_intents(self) -> tuple[BOAcquisitionIntent, ...]:
        """Return acquisition families that match the request semantics."""

        return recommend_bo_acquisition_intents(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe BO optimization specification."""

        return {
            "mode": self.mode.value,
            "objectives": [objective.to_dict() for objective in self.objectives],
            "features": [feature.to_dict() for feature in self.features],
            "output_constraints": [constraint.to_dict() for constraint in self.output_constraints],
            "acquisition_intent": (
                None if self.acquisition_intent is None else self.acquisition_intent.value
            ),
            "recommended_acquisition_intents": [
                intent.value for intent in self.recommended_acquisition_intents()
            ],
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "cost_column": self.cost_column,
            "objective_names": list(self.objective_names),
            "controllable_feature_names": list(self.controllable_feature_names),
            "fixed_feature_names": list(self.fixed_feature_names),
            "metadata": _json_safe(self.metadata),
        }


def recommend_bo_acquisition_intents(
    spec: BOOptimizationSpec | Mapping[str, Any],
) -> tuple[BOAcquisitionIntent, ...]:
    """Recommend acquisition families for a BO request.

    The recommendation is policy-light and optimizer-independent. Private
    platforms can layer cost, compute budget, saved-model, or user-interface
    policy on top while keeping the scientific semantics reusable.
    """

    optimization_spec = _coerce_optimization_spec(spec)
    if optimization_spec.acquisition_intent is not None:
        return (optimization_spec.acquisition_intent,)

    objective_modes = {objective.mode for objective in optimization_spec.objectives}
    intents: list[BOAcquisitionIntent] = []

    if len(optimization_spec.objectives) > 1:
        intents.extend(
            [
                BOAcquisitionIntent.MULTI_OBJECTIVE_EHVI,
                BOAcquisitionIntent.SCALARIZED_EXPECTED_IMPROVEMENT,
            ]
        )
    elif objective_modes & {BOObjectiveMode.TARGET_VALUE, BOObjectiveMode.TARGET_RANGE}:
        intents.append(BOAcquisitionIntent.TARGET_PROBABILITY)
    elif optimization_spec.output_constraints:
        intents.extend(
            [
                BOAcquisitionIntent.CONSTRAINED_NOISY_EXPECTED_IMPROVEMENT,
                BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
            ]
        )
    else:
        first_objective = optimization_spec.objectives[0]
        if first_objective.mode is BOObjectiveMode.MINIMIZE:
            intents.extend(
                [
                    BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND,
                    BOAcquisitionIntent.NOISY_EXPECTED_IMPROVEMENT,
                ]
            )
        else:
            intents.extend(
                [
                    BOAcquisitionIntent.NOISY_EXPECTED_IMPROVEMENT,
                    BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
                    BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND,
                ]
            )

    if optimization_spec.output_constraints and not any(
        intent
        in {
            BOAcquisitionIntent.CONSTRAINED_NOISY_EXPECTED_IMPROVEMENT,
            BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
        }
        for intent in intents
    ):
        intents.append(BOAcquisitionIntent.CONSTRAINED_NOISY_EXPECTED_IMPROVEMENT)
    if optimization_spec.batch_size > 1:
        intents.append(BOAcquisitionIntent.THOMPSON_SAMPLING)
    if optimization_spec.cost_column is not None:
        intents.append(BOAcquisitionIntent.COST_AWARE_EXPECTED_IMPROVEMENT)

    return tuple(dict.fromkeys(intents))


def recommend_saved_prediction_acquisition_intents(
    spec: BOOptimizationSpec | Mapping[str, Any],
) -> tuple[BOAcquisitionIntent, ...]:
    """Recommend acquisitions supported by saved mean/uncertainty scoring.

    This deliberately excludes acquisitions that require posterior sampling,
    fantasy models, joint batch optimization, costs, or multi-output models.
    """

    optimization_spec = _coerce_optimization_spec(spec)
    if len(optimization_spec.objectives) != 1:
        return ()
    requested = optimization_spec.acquisition_intent
    supported = {
        BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE,
        BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
        BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND,
        BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND,
        BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT,
        BOAcquisitionIntent.TARGET_PROBABILITY,
        BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
    }
    if requested is not None:
        return (requested,) if requested in supported else ()

    objective = optimization_spec.objectives[0]
    if objective.mode in {BOObjectiveMode.TARGET_VALUE, BOObjectiveMode.TARGET_RANGE}:
        return (
            BOAcquisitionIntent.TARGET_PROBABILITY,
            BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE,
        )
    if objective.mode is BOObjectiveMode.MINIMIZE:
        return (
            BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND,
            BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
            BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT,
            BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE,
        )
    return (
        BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND,
        BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
        BOAcquisitionIntent.PROBABILITY_OF_IMPROVEMENT,
        BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE,
    )


def _coerce_optimization_spec(
    value: BOOptimizationSpec | Mapping[str, Any],
) -> BOOptimizationSpec:
    if isinstance(value, BOOptimizationSpec):
        return value
    if isinstance(value, Mapping):
        return BOOptimizationSpec(**dict(value))
    raise TypeError("spec must be a BOOptimizationSpec or mapping")


def _coerce_feature(value: BOFeatureSpec | Mapping[str, Any]) -> BOFeatureSpec:
    if isinstance(value, BOFeatureSpec):
        return value
    if isinstance(value, Mapping):
        return BOFeatureSpec(**dict(value))
    raise TypeError("features must contain BOFeatureSpec or mapping values")


def _coerce_objective(
    value: BOObjectivePreference | Mapping[str, Any],
) -> BOObjectivePreference:
    if isinstance(value, BOObjectivePreference):
        return value
    if isinstance(value, Mapping):
        return BOObjectivePreference(**dict(value))
    raise TypeError("objectives must contain BOObjectivePreference or mapping values")


def _coerce_output_constraint(
    value: BOOutputConstraintSpec | Mapping[str, Any],
) -> BOOutputConstraintSpec:
    if isinstance(value, BOOutputConstraintSpec):
        return value
    if isinstance(value, Mapping):
        return BOOutputConstraintSpec(**dict(value))
    raise TypeError("output_constraints must contain BOOutputConstraintSpec or mapping values")


def _coerce_enum(enum_cls: type[Enum], value: Enum | str) -> Any:
    if isinstance(value, enum_cls):
        return value
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return enum_cls(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_cls)
        raise ValueError(f"{enum_cls.__name__} must be one of: {allowed}") from exc


def _clean_text(value: Any, field_name: str = "value") -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _clean_optional_text(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_finite_float(value: Any | None, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _positive_finite_float(value: Any, field_name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return numeric


def _optional_unit_interval_float(value: Any | None, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be in the interval [0, 1]")
    return numeric


def _positive_int(value: Any, field_name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if integer < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return integer


def _duplicate_names(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        frame = value.astype(object).where(pd.notna(value), None)
        return [_json_safe(row) for row in frame.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
