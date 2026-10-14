from __future__ import annotations

import json

import pytest

from matgpr import (
    BOAcquisitionIntent,
    BOFeatureSpec,
    BOFeatureType,
    BOObjectiveMode,
    BOObjectivePreference,
    BOOptimizationMode,
    BOOptimizationSpec,
    BOOutputConstraintSpec,
    recommend_bo_acquisition_intents,
)


def test_bo_optimization_spec_serializes_feature_optimization_request() -> None:
    spec = BOOptimizationSpec(
        mode="feature_optimization",
        objectives=(
            {
                "target_name": "relative_density",
                "mode": "target_range",
                "target_lower_bound": 0.985,
                "target_upper_bound": 0.995,
                "units": "fraction",
            },
        ),
        features=(
            {
                "name": "laser_power_W",
                "variable_type": "continuous",
                "lower_bound": 120.0,
                "upper_bound": 360.0,
                "units": "W",
            },
            {
                "name": "scan_speed_mm_s",
                "variable_type": BOFeatureType.CONTINUOUS,
                "lower_bound": 200.0,
                "upper_bound": 1400.0,
                "units": "mm/s",
            },
            {
                "name": "alloy_family",
                "variable_type": "fixed",
                "fixed_value": "316L",
            },
        ),
        output_constraints=(
            {
                "target_name": "porosity",
                "upper_bound": 0.01,
                "probability_threshold": 0.9,
            },
        ),
        batch_size=4,
        metadata={"domain": "additive_manufacturing"},
    )

    payload = spec.to_dict()

    assert spec.mode is BOOptimizationMode.FEATURE_OPTIMIZATION
    assert spec.objectives[0].mode is BOObjectiveMode.TARGET_RANGE
    assert spec.controllable_feature_names == ("laser_power_W", "scan_speed_mm_s")
    assert spec.fixed_feature_names == ("alloy_family",)
    assert payload["recommended_acquisition_intents"] == [
        BOAcquisitionIntent.TARGET_PROBABILITY.value,
        BOAcquisitionIntent.CONSTRAINED_NOISY_EXPECTED_IMPROVEMENT.value,
        BOAcquisitionIntent.THOMPSON_SAMPLING.value,
    ]
    assert payload["features"][0]["is_controllable"] is True
    assert payload["features"][2]["fixed_value"] == "316L"
    assert json.dumps(payload)


def test_bo_acquisition_recommendations_include_constraints_and_cost() -> None:
    spec = BOOptimizationSpec(
        mode=BOOptimizationMode.FINITE_POOL_RANKING,
        objectives=(BOObjectivePreference("yield_strength", mode="maximize"),),
        output_constraints=(BOOutputConstraintSpec("ductility", lower_bound=0.08),),
        cost_column="experiment_cost",
    )

    intents = recommend_bo_acquisition_intents(spec)

    assert intents == (
        BOAcquisitionIntent.CONSTRAINED_NOISY_EXPECTED_IMPROVEMENT,
        BOAcquisitionIntent.CONSTRAINED_EXPECTED_IMPROVEMENT,
        BOAcquisitionIntent.COST_AWARE_EXPECTED_IMPROVEMENT,
    )


def test_bo_feature_spec_validates_variable_type_requirements() -> None:
    with pytest.raises(ValueError, match="requires lower_bound"):
        BOFeatureSpec(name="temperature_C", variable_type="continuous")

    with pytest.raises(ValueError, match="categorical"):
        BOFeatureSpec(name="solvent", variable_type="categorical")

    with pytest.raises(ValueError, match="fixed_value"):
        BOFeatureSpec(name="machine_id", variable_type="fixed")


def test_bo_objective_preference_validates_target_modes() -> None:
    with pytest.raises(ValueError, match="target_value"):
        BOObjectivePreference(target_name="band_gap", mode="target_value")

    with pytest.raises(ValueError, match="target_lower_bound"):
        BOObjectivePreference(target_name="density", mode="target_range")


def test_bo_feature_optimization_requires_controllable_feature() -> None:
    with pytest.raises(ValueError, match="controllable feature"):
        BOOptimizationSpec(
            mode="feature_optimization",
            objectives=(BOObjectivePreference("conductivity"),),
            features=(BOFeatureSpec("material_id", variable_type="fixed", fixed_value="A"),),
        )
