"""Focused tests for saved-prediction BO acquisition policy."""

from __future__ import annotations

import importlib
from math import erf, exp, pi, sqrt

import numpy as np
import pytest

import matgpr.bayesian_optimization as bo
from matgpr.bayesian_optimization_specs import (
    BOAcquisitionIntent,
    BOObjectiveMode,
    BOObjectivePreference,
    BOOutputConstraintSpec,
)

pytestmark = pytest.mark.unit

_MOVED_POLICY_NAMES = (
    "_SAVED_PREDICTION_ACQUISITIONS",
    "_UNCERTAINTY_REQUIRED_ACQUISITIONS",
    "_resolve_saved_prediction_acquisition",
    "_saved_prediction_acquisition_values",
    "_posterior_mean_preference",
    "_normal_expected_improvement",
    "_normal_probability_of_improvement",
    "_apply_saved_prediction_feasibility",
    "_objective_event_interval",
    "_improvement_event_interval",
    "_constraint_event_interval",
    "_intersect_intervals",
    "_required_incumbent_value",
    "_normal_interval_probability",
    "_joint_output_constraint_probability",
    "_normal_target_closeness",
    "_standard_normal_cdf",
    "_standard_normal_pdf",
    "_unit_interval_scores",
    "_objective_target_tolerance",
    "_validate_single_target_output_constraints",
)


def _objective(mode: BOObjectiveMode | str, **kwargs: object) -> BOObjectivePreference:
    return BOObjectivePreference(target_name="property", mode=mode, **kwargs)


@pytest.mark.parametrize(
    ("objective", "uncertainty_available", "expected"),
    [
        (_objective("maximize"), True, BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND),
        (_objective("minimize"), True, BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND),
        (
            _objective("target_value", target_value=1.0, target_tolerance=0.2),
            True,
            BOAcquisitionIntent.TARGET_PROBABILITY,
        ),
        (
            _objective("target_range", target_lower_bound=0.8, target_upper_bound=1.2),
            True,
            BOAcquisitionIntent.TARGET_PROBABILITY,
        ),
        (_objective("maximize"), False, BOAcquisitionIntent.POSTERIOR_MEAN_PREFERENCE),
    ],
)
def test_resolve_saved_prediction_acquisition_defaults(
    objective: BOObjectivePreference,
    uncertainty_available: bool,
    expected: BOAcquisitionIntent,
) -> None:
    requested, effective = bo._resolve_saved_prediction_acquisition(
        None,
        objective,
        uncertainty_available=uncertainty_available,
    )

    assert requested is None
    assert effective is expected


@pytest.mark.parametrize(
    ("intent", "objective", "uncertainty_available", "message"),
    [
        (
            BOAcquisitionIntent.NOISY_EXPECTED_IMPROVEMENT,
            _objective("maximize"),
            True,
            "not supported",
        ),
        (
            BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND,
            _objective("minimize"),
            True,
            "requires a maximize objective",
        ),
        (
            BOAcquisitionIntent.LOWER_CONFIDENCE_BOUND,
            _objective("maximize"),
            True,
            "requires a minimize objective",
        ),
        (
            BOAcquisitionIntent.TARGET_PROBABILITY,
            _objective("maximize"),
            True,
            "requires a target_value or target_range objective",
        ),
        (
            BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
            _objective("target_value", target_value=1.0),
            True,
            "requires a maximize or minimize objective",
        ),
        (
            BOAcquisitionIntent.EXPECTED_IMPROVEMENT,
            _objective("maximize"),
            False,
            "predicted_std is required",
        ),
    ],
)
def test_resolve_saved_prediction_acquisition_rejects_invalid_combinations(
    intent: BOAcquisitionIntent,
    objective: BOObjectivePreference,
    uncertainty_available: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bo._resolve_saved_prediction_acquisition(
            intent,
            objective,
            uncertainty_available=uncertainty_available,
        )


@pytest.mark.parametrize(
    ("objective", "mean", "expected"),
    [
        (_objective("maximize"), [1.0, 2.0], [1.0, 2.0]),
        (_objective("minimize"), [1.0, 2.0], [-1.0, -2.0]),
        (_objective("target_value", target_value=2.0), [1.0, 2.0], [-1.0, 0.0]),
        (
            _objective("target_range", target_lower_bound=1.0, target_upper_bound=3.0),
            [0.0, 2.0, 4.0],
            [-1.0, 0.0, -1.0],
        ),
    ],
)
def test_posterior_mean_preference_covers_all_objective_modes(
    objective: BOObjectivePreference,
    mean: list[float],
    expected: list[float],
) -> None:
    np.testing.assert_allclose(
        bo._posterior_mean_preference(np.asarray(mean), objective),
        expected,
    )


def test_expected_and_probability_of_improvement_match_closed_form() -> None:
    mean = np.array([1.8, 1.0, 1.6])
    std = np.array([0.0, 0.0, 0.5])
    objective = _objective("maximize")
    z = -0.2
    cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    pdf = exp(-0.5 * z**2) / sqrt(2.0 * pi)

    expected_improvement = bo._normal_expected_improvement(
        mean,
        std,
        objective=objective,
        incumbent_value=1.5,
        improvement_threshold=0.2,
    )
    probability = bo._normal_probability_of_improvement(
        mean,
        std,
        objective=objective,
        incumbent_value=1.5,
        improvement_threshold=0.2,
    )

    np.testing.assert_allclose(expected_improvement, [0.1, 0.0, -0.1 * cdf + 0.5 * pdf])
    np.testing.assert_allclose(probability, [1.0, 0.0, cdf])


def test_deterministic_improvement_skips_normal_distribution_terms() -> None:
    objective = _objective("maximize")
    mean = np.array([2.0, 1.0])
    std = np.zeros(2)

    np.testing.assert_array_equal(
        bo._normal_expected_improvement(
            mean,
            std,
            objective=objective,
            incumbent_value=1.5,
            improvement_threshold=0.0,
        ),
        [0.5, 0.0],
    )
    np.testing.assert_array_equal(
        bo._normal_probability_of_improvement(
            mean,
            std,
            objective=objective,
            incumbent_value=1.5,
            improvement_threshold=0.0,
        ),
        [1.0, 0.0],
    )


def test_acquisition_value_dispatch_rejects_unsupported_internal_intent() -> None:
    with pytest.raises(ValueError, match="Unsupported saved-prediction acquisition"):
        bo._saved_prediction_acquisition_values(
            np.array([1.0]),
            np.array([0.1]),
            _objective("maximize"),
            acquisition_intent=BOAcquisitionIntent.NOISY_EXPECTED_IMPROVEMENT,
            beta=0.2,
            incumbent_value=0.5,
            improvement_threshold=0.0,
        )


def test_interval_probability_handles_open_closed_and_uncertain_values() -> None:
    mean = np.array([0.0, 1.0, 2.0, 0.0])
    std = np.array([0.0, 0.0, 0.0, 1.0])

    closed = bo._normal_interval_probability(
        mean,
        std,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    open_interval = bo._normal_interval_probability(
        mean,
        std,
        lower_bound=0.0,
        upper_bound=1.0,
        inclusive=False,
    )

    np.testing.assert_allclose(closed[:3], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(open_interval[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(closed[3], 0.3413447460685429)


def test_unbounded_interval_probability_is_one() -> None:
    np.testing.assert_array_equal(
        bo._normal_interval_probability(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            lower_bound=None,
            upper_bound=None,
        ),
        [1.0, 1.0],
    )


def test_event_intervals_and_intersection_preserve_impossible_ranges() -> None:
    target = _objective("target_value", target_value=2.0, target_tolerance=0.25)
    maximize = _objective("maximize")

    assert bo._objective_event_interval(target) == (1.75, 2.25)
    assert bo._improvement_event_interval(
        maximize,
        incumbent_value=1.0,
        improvement_threshold=0.2,
    ) == (1.2, None)
    assert bo._improvement_event_interval(
        _objective("minimize"),
        incumbent_value=1.5,
        improvement_threshold=0.2,
    ) == (None, 1.3)
    assert bo._intersect_intervals((None, None), (None, None)) == (None, None)
    assert bo._intersect_intervals((0.0, 1.0), (2.0, 3.0)) == (1.0, 0.0)


def test_objective_event_interval_requires_a_closed_target_event() -> None:
    with pytest.raises(ValueError, match="positive target_tolerance"):
        bo._objective_event_interval(_objective("target_value", target_value=2.0))
    with pytest.raises(ValueError, match="does not define a closed target event"):
        bo._objective_event_interval(_objective("maximize"))


def test_target_tolerance_metadata_and_incumbent_validation_are_explicit() -> None:
    objective = _objective("target_value", target_value=2.0, metadata={"tolerance": 0.3})
    invalid = _objective("target_value", target_value=2.0, metadata={"tolerance": -0.1})

    assert bo._objective_target_tolerance(objective) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="target tolerance"):
        bo._objective_target_tolerance(invalid)
    with pytest.raises(ValueError, match="incumbent_value is required"):
        bo._required_incumbent_value(None)
    with pytest.raises(ValueError, match="incumbent_value must be finite"):
        bo._required_incumbent_value(np.nan)

    absolute = _objective(
        "target_value",
        target_value=2.0,
        metadata={"absolute_tolerance": 0.4},
    )
    missing = _objective("target_value", target_value=2.0)
    assert bo._objective_target_tolerance(absolute) == pytest.approx(0.4)
    assert bo._objective_target_tolerance(missing) is None


def test_unit_interval_scores_cover_empty_constant_and_varying_values() -> None:
    np.testing.assert_array_equal(
        bo._unit_interval_scores(np.array([np.nan, np.inf])),
        [0.0, 0.0],
    )
    np.testing.assert_array_equal(bo._unit_interval_scores(np.array([2.0, 2.0])), [1.0, 1.0])
    np.testing.assert_allclose(bo._unit_interval_scores(np.array([1.0, 2.0, 3.0])), [0, 0.5, 1])


def test_joint_constraint_probability_and_target_validation() -> None:
    constraints = (
        BOOutputConstraintSpec(target_name="property", lower_bound=-1.0),
        BOOutputConstraintSpec(target_name="property", upper_bound=1.0),
    )
    probability = bo._joint_output_constraint_probability(
        np.array([0.0]),
        np.array([1.0]),
        constraints,
    )

    np.testing.assert_allclose(probability, [0.6826894921370859])
    bo._validate_single_target_output_constraints(_objective("maximize"), constraints)
    with pytest.raises(ValueError, match="single-target"):
        bo._validate_single_target_output_constraints(
            _objective("maximize"),
            (BOOutputConstraintSpec(target_name="other", lower_bound=0.0),),
        )


def test_constraint_intervals_and_conflicts_are_explicit() -> None:
    lower_only = (BOOutputConstraintSpec(target_name="property", lower_bound=0.0),)
    upper_only = (BOOutputConstraintSpec(target_name="property", upper_bound=1.0),)
    conflicting = (
        BOOutputConstraintSpec(target_name="property", lower_bound=2.0),
        BOOutputConstraintSpec(target_name="property", upper_bound=1.0),
    )

    assert bo._constraint_event_interval(lower_only) == (0.0, None)
    assert bo._constraint_event_interval(upper_only) == (None, 1.0)
    np.testing.assert_array_equal(
        bo._joint_output_constraint_probability(
            np.array([1.5]),
            np.array([0.2]),
            conflicting,
        ),
        [0.0],
    )


def test_target_closeness_handles_deterministic_and_uncertain_values() -> None:
    np.testing.assert_allclose(
        bo._normal_target_closeness(
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            target=1.0,
        ),
        [1.0, np.exp(-0.5)],
    )
    np.testing.assert_array_equal(
        bo._normal_target_closeness(
            np.array([1.0, 2.0]),
            np.zeros(2),
            target=1.0,
        ),
        [1.0, 0.0],
    )
    np.testing.assert_allclose(
        bo._normal_target_closeness(
            np.array([1.0, 2.0]),
            np.ones(2),
            target=1.0,
        ),
        [1.0, np.exp(-0.5)],
    )


def test_feasibility_weighting_keeps_objective_component_nonnegative() -> None:
    constraints = (BOOutputConstraintSpec(target_name="property", lower_bound=-10.5),)
    acquisition, component, name = bo._apply_saved_prediction_feasibility(
        np.array([-10.0, -11.0]),
        np.array([-10.0, -11.0]),
        np.zeros(2),
        objective=_objective("maximize"),
        constraints=constraints,
        acquisition_intent=BOAcquisitionIntent.UPPER_CONFIDENCE_BOUND,
        incumbent_value=None,
        improvement_threshold=0.0,
        feasibility_probability=np.array([1.0, 0.0]),
    )

    np.testing.assert_array_equal(component, [1.0, 0.0])
    np.testing.assert_array_equal(acquisition, [1.0, 0.0])
    assert name == "feasibility_weighted_upper_confidence_bound"


def test_constrained_expected_improvement_uses_expected_improvement_name() -> None:
    constraint = (BOOutputConstraintSpec(target_name="property", lower_bound=0.0),)
    acquisition, component, name = bo._apply_saved_prediction_feasibility(
        np.array([0.4]),
        np.array([1.0]),
        np.array([0.2]),
        objective=_objective("maximize"),
        constraints=constraint,
        acquisition_intent=BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
        incumbent_value=0.8,
        improvement_threshold=0.0,
        feasibility_probability=np.array([0.5]),
    )

    np.testing.assert_array_equal(component, [0.4])
    np.testing.assert_array_equal(acquisition, [0.2])
    assert name == "feasibility_weighted_expected_improvement"


def test_parent_facade_adopts_internal_saved_prediction_policy_module() -> None:
    policy = importlib.import_module("matgpr._bo_saved_prediction")

    for name in _MOVED_POLICY_NAMES:
        assert getattr(bo, name) is getattr(policy, name)
