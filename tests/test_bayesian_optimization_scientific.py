"""Analytical scientific checks for finite-pool Bayesian optimization."""

from __future__ import annotations

from math import erf, exp, pi, sqrt

import numpy as np
import pandas as pd
import pytest

from matgpr import (
    CandidateConstraint,
    CandidateDuplicatePolicy,
    ObjectiveSpec,
    apply_candidate_constraints,
    apply_candidate_duplicate_policy,
    pareto_front_mask,
    rank_bo_candidate_predictions,
    rank_multi_objective_candidates,
    scalarize_objectives,
    select_diverse_batch,
)

pytestmark = pytest.mark.scientific


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return exp(-0.5 * value**2) / sqrt(2.0 * pi)


def test_confidence_bounds_match_closed_form_and_objective_direction() -> None:
    candidate_data = pd.DataFrame({"candidate_id": ["a", "b"]})

    ucb = rank_bo_candidate_predictions(
        predicted_mean=(1.0, 0.9),
        predicted_std=(0.2, 0.5),
        objective={"target_name": "strength", "mode": "maximize"},
        acquisition_intent="upper_confidence_bound",
        beta=2.0,
        candidate_data=candidate_data,
    ).set_index("candidate_id")
    lcb = rank_bo_candidate_predictions(
        predicted_mean=(1.0, 1.2),
        predicted_std=(0.1, 0.8),
        objective={"target_name": "cost", "mode": "minimize"},
        acquisition_intent="lower_confidence_bound",
        beta=0.5,
        candidate_data=candidate_data,
    ).set_index("candidate_id")

    np.testing.assert_allclose(
        ucb.loc[["a", "b"], "matgpr_acquisition_base"],
        [1.4, 1.9],
    )
    np.testing.assert_allclose(
        lcb.loc[["a", "b"], "matgpr_acquisition_base"],
        [-0.95, -0.8],
    )
    assert ucb.loc["b", "matgpr_rank"] == 1
    assert lcb.loc["b", "matgpr_rank"] == 1


@pytest.mark.parametrize(
    ("mode", "means", "expected_improvements"),
    [
        ("maximize", (1.0, 1.8), (0.0, 0.1 * _normal_cdf(0.2) + 0.5 * _normal_pdf(0.2))),
        ("minimize", (2.0, 1.2), (0.0, 0.1 * _normal_cdf(0.2) + 0.5 * _normal_pdf(0.2))),
    ],
)
def test_expected_improvement_matches_normal_closed_form(
    mode: str,
    means: tuple[float, float],
    expected_improvements: tuple[float, float],
) -> None:
    ranked = rank_bo_candidate_predictions(
        predicted_mean=means,
        predicted_std=(0.0, 0.5),
        objective={"target_name": "property", "mode": mode},
        acquisition_intent="expected_improvement",
        incumbent_value=1.5,
        improvement_threshold=0.2,
        candidate_data=pd.DataFrame({"candidate_id": ["deterministic", "uncertain"]}),
    ).set_index("candidate_id")

    np.testing.assert_allclose(
        ranked.loc[
            ["deterministic", "uncertain"],
            "matgpr_acquisition_base",
        ],
        expected_improvements,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("mode", "means"),
    [
        ("maximize", (1.8, 1.0, 1.6)),
        ("minimize", (1.2, 2.0, 1.4)),
    ],
)
def test_probability_of_improvement_handles_deterministic_and_uncertain_predictions(
    mode: str,
    means: tuple[float, float, float],
) -> None:
    ranked = rank_bo_candidate_predictions(
        predicted_mean=means,
        predicted_std=(0.0, 0.0, 0.5),
        objective={"target_name": "property", "mode": mode},
        acquisition_intent="probability_of_improvement",
        incumbent_value=1.5,
        improvement_threshold=0.2,
        candidate_data=pd.DataFrame({"candidate_id": ["better", "worse", "uncertain"]}),
    ).set_index("candidate_id")

    np.testing.assert_allclose(
        ranked.loc[["better", "worse", "uncertain"], "matgpr_acquisition_base"],
        [1.0, 0.0, _normal_cdf(-0.2)],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("bound", [np.nan, np.inf, -np.inf])
def test_candidate_constraints_reject_nonfinite_numeric_bounds(bound: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        CandidateConstraint(name="temperature", column="temperature_c", lower_bound=bound)


def test_candidate_constraint_boundaries_and_missing_policy_are_explicit() -> None:
    candidates = pd.DataFrame({"temperature_c": [20.0, 20.1, np.nan]})
    constraint = CandidateConstraint(
        name="strict_lower_temperature",
        column="temperature_c",
        lower_bound=20.0,
        inclusive=False,
        allow_missing=True,
    )

    constrained = apply_candidate_constraints(candidates, constraint)

    assert constrained["matgpr_feasible"].tolist() == [False, True, True]


def test_duplicate_filtering_preserves_new_candidate_order() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["measured", "near_measured", "new_a", "new_b"],
            "x": [0.0, 0.05, 2.0, 3.0],
        }
    )
    policy = CandidateDuplicatePolicy(
        existing_candidates=pd.DataFrame({"candidate_id": ["measured"], "x": [0.0]}),
        key_columns=("candidate_id",),
        feature_columns=("x",),
        feature_tolerance=0.1,
    )

    filtered = apply_candidate_duplicate_policy(candidates, policy)

    assert filtered["candidate_id"].tolist() == ["new_a", "new_b"]


def test_equal_acquisition_scores_keep_original_candidate_order() -> None:
    ranked = rank_bo_candidate_predictions(
        predicted_mean=(1.0, 1.0, 1.0),
        objective={"target_name": "strength", "mode": "maximize"},
        acquisition_intent="posterior_mean_preference",
        candidate_data=pd.DataFrame({"candidate_id": ["first", "second", "third"]}),
    )

    assert ranked["candidate_id"].tolist() == ["first", "second", "third"]
    assert ranked["matgpr_rank"].tolist() == [1, 2, 3]


def test_diversity_selection_is_stable_then_selects_farthest_candidate() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["first", "near", "far"],
            "matgpr_acquisition": [1.0, 1.0, 1.0],
            "descriptor": [0.0, 0.1, 10.0],
        }
    )

    batch = select_diverse_batch(
        candidates,
        top_k=2,
        feature_columns=("descriptor",),
        diversity_weight=1.0,
        standardize_features=False,
    )

    assert batch["candidate_id"].tolist() == ["first", "far"]
    np.testing.assert_allclose(batch["matgpr_diversity_distance"].iloc[1], 10.0)


@pytest.mark.parametrize("top_k", [True, 1.5])
def test_finite_pool_top_k_requires_a_real_integer(top_k: object) -> None:
    candidate_data = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "matgpr_acquisition": [1.0, 0.5],
            "descriptor": [0.0, 1.0],
        }
    )

    with pytest.raises(TypeError, match="top_k must be an integer"):
        rank_bo_candidate_predictions(
            predicted_mean=(1.0, 0.5),
            objective={"target_name": "strength", "mode": "maximize"},
            acquisition_intent="posterior_mean_preference",
            top_k=top_k,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="top_k must be an integer"):
        select_diverse_batch(
            candidate_data,
            top_k=top_k,  # type: ignore[arg-type]
            diversity_weight=0.0,
        )
    with pytest.raises(TypeError, match="top_k must be an integer"):
        rank_multi_objective_candidates(
            candidate_data.assign(cost=[2.0, 1.0]),
            (
                ObjectiveSpec("strength", "matgpr_acquisition", goal="maximize"),
                ObjectiveSpec("cost", "cost", goal="minimize"),
            ),
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_multi_objective_mixed_directions_have_exact_pareto_and_utility_results() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["strong", "cheap", "balanced", "dominated"],
            "strength": [10.0, 8.0, 9.0, 8.0],
            "cost": [10.0, 2.0, 6.0, 8.0],
        }
    )
    objectives = (
        ObjectiveSpec("strength", "strength", goal="maximize", weight=1.0),
        ObjectiveSpec("cost", "cost", goal="minimize", weight=1.0),
    )

    mask = pareto_front_mask(candidates, objectives)
    scores = scalarize_objectives(candidates, objectives)
    ranked = rank_multi_objective_candidates(candidates, objectives)

    assert mask.tolist() == [True, True, True, False]
    np.testing.assert_allclose(scores.to_numpy(), [0.5, 0.5, 0.5, 0.125])
    assert ranked["candidate_id"].tolist() == ["strong", "cheap", "balanced", "dominated"]
