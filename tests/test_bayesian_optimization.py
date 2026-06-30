from __future__ import annotations

import importlib.util
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from matgpr import (
    BayesianOptimizationResult,
    BORecommendationAudit,
    BoTorchSurrogate,
    CandidateConstraint,
    CandidateDuplicatePolicy,
    CandidateTrustRegion,
    MultiObjectiveBayesianOptimizationResult,
    apply_candidate_constraints,
    apply_candidate_duplicate_policy,
    apply_candidate_trust_region,
    fit_botorch_surrogate,
    fit_multi_objective_botorch_surrogate,
    observation_noise_variance,
    rank_discrete_candidates,
    rank_multi_objective_discrete_candidates,
    select_diverse_batch,
    select_sequential_multi_objective_batch,
    suggest_multi_objective_next_experiments,
    suggest_next_experiments,
    summarize_bo_recommendation_audit,
)


class BayesianOptimizationApiTests(unittest.TestCase):
    def test_observation_noise_variance_from_reported_uncertainty_columns(self):
        data = pd.DataFrame(
            {
                "target_variance": [0.0, 0.04],
                "target_std": [0.1, 0.2],
                "target_sem": [0.01, 0.02],
            }
        )

        variance = observation_noise_variance(
            data,
            variance_column="target_variance",
            min_variance=1e-6,
        )
        std_variance = observation_noise_variance(data, std_column="target_std")
        sem_variance = observation_noise_variance(data, sem_column="target_sem")

        np.testing.assert_allclose(variance.to_numpy(), [1e-6, 0.04])
        np.testing.assert_allclose(std_variance.to_numpy(), [0.01, 0.04])
        np.testing.assert_allclose(sem_variance.to_numpy(), [0.0001, 0.0004])
        self.assertEqual(variance.name, "matgpr_noise_variance")

    def test_observation_noise_variance_from_replicate_groups(self):
        data = pd.DataFrame(
            {
                "material_id": ["a", "a", "b", "b", "c"],
                "conductivity": [1.0, 1.2, 2.0, 2.4, 5.0],
            }
        )

        variance = observation_noise_variance(
            data,
            replicate_group_column="material_id",
            target_column="conductivity",
        )

        np.testing.assert_allclose(
            variance.to_numpy(),
            [0.02, 0.02, 0.08, 0.08, 0.05],
            rtol=1e-12,
        )

    def test_observation_noise_variance_validation_errors_are_explicit(self):
        data = pd.DataFrame(
            {
                "target_variance": [0.01, 0.02],
                "target_std": [0.1, -0.2],
                "group": ["a", "b"],
                "target": [1.0, 2.0],
            }
        )

        with self.assertRaises(ValueError):
            observation_noise_variance(data)
        with self.assertRaises(ValueError):
            observation_noise_variance(
                data,
                variance_column="target_variance",
                std_column="target_std",
            )
        with self.assertRaises(ValueError):
            observation_noise_variance(data, std_column="target_std")
        with self.assertRaises(ValueError):
            observation_noise_variance(data, replicate_group_column="group")
        with self.assertRaises(ValueError):
            observation_noise_variance(
                data,
                replicate_group_column="missing_group",
                target_column="target",
            )

    def test_candidate_constraints_annotate_feasibility_and_violations(self):
        candidates = pd.DataFrame(
            {
                "temperature_c": [25.0, 80.0, 120.0],
                "solvent_class": ["green", "green", "restricted"],
            }
        )
        constrained = apply_candidate_constraints(
            candidates,
            [
                CandidateConstraint(
                    name="temperature_window",
                    column="temperature_c",
                    lower_bound=20.0,
                    upper_bound=100.0,
                ),
                CandidateConstraint(
                    name="allowed_solvent",
                    column="solvent_class",
                    allowed_values=("green",),
                ),
            ],
        )

        self.assertEqual(constrained["matgpr_feasible"].tolist(), [True, True, False])
        self.assertEqual(
            constrained["matgpr_constraint_violations"].tolist(),
            ["", "", "temperature_window; allowed_solvent"],
        )

    def test_candidate_constraints_can_use_any_constraint_when_requested(self):
        candidates = pd.DataFrame(
            {
                "temperature_c": [10.0, 80.0, 120.0],
                "solvent_class": ["restricted", "green", "restricted"],
            }
        )
        constrained = apply_candidate_constraints(
            candidates,
            [
                CandidateConstraint(
                    name="temperature_window",
                    column="temperature_c",
                    lower_bound=20.0,
                    upper_bound=100.0,
                ),
                CandidateConstraint(
                    name="allowed_solvent",
                    column="solvent_class",
                    allowed_values=("green",),
                ),
            ],
            require_all=False,
        )

        self.assertEqual(constrained["matgpr_feasible"].tolist(), [False, True, False])
        self.assertEqual(
            constrained["matgpr_constraint_violations"].tolist(),
            [
                "temperature_window; allowed_solvent",
                "",
                "temperature_window; allowed_solvent",
            ],
        )

    def test_candidate_trust_region_filters_and_annotates_distances(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["near", "far"],
                "descriptor_x": [0.2, 2.0],
                "descriptor_y": [0.1, 0.0],
            }
        )
        trust_region = CandidateTrustRegion(
            centers=pd.DataFrame({"descriptor_x": [0.0], "descriptor_y": [0.0]}),
            radius=0.5,
            feature_columns=("descriptor_x", "descriptor_y"),
        )

        annotated = apply_candidate_trust_region(
            candidates,
            trust_region,
            policy="annotate",
        )
        filtered = apply_candidate_trust_region(candidates, trust_region)

        np.testing.assert_allclose(
            annotated["matgpr_trust_region_distance"].to_numpy(),
            [np.sqrt(0.05), 2.0],
        )
        self.assertEqual(annotated["matgpr_in_trust_region"].tolist(), [True, False])
        self.assertEqual(filtered["material_id"].tolist(), ["near"])

    def test_candidate_duplicate_policy_filters_exact_key_duplicates(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["already_measured", "new_candidate"],
                "descriptor_x": [0.0, 1.0],
            }
        )
        duplicate_policy = CandidateDuplicatePolicy(
            existing_candidates=pd.DataFrame({"material_id": ["already_measured"]}),
            key_columns=("material_id",),
        )

        annotated = apply_candidate_duplicate_policy(
            candidates,
            duplicate_policy,
            policy="annotate",
        )
        filtered = apply_candidate_duplicate_policy(candidates, duplicate_policy)

        self.assertEqual(annotated["matgpr_is_duplicate"].tolist(), [True, False])
        self.assertEqual(annotated["matgpr_duplicate_reason"].tolist(), ["key", ""])
        self.assertEqual(filtered["material_id"].tolist(), ["new_candidate"])

    def test_candidate_duplicate_policy_filters_feature_tolerance_duplicates(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["near_existing", "far_candidate"],
                "descriptor_x": [0.1, 1.0],
                "descriptor_y": [0.1, 0.0],
            }
        )
        duplicate_policy = CandidateDuplicatePolicy(
            existing_candidates=pd.DataFrame(
                {"descriptor_x": [0.0], "descriptor_y": [0.0]}
            ),
            feature_columns=("descriptor_x", "descriptor_y"),
            feature_tolerance=0.2,
        )

        annotated = apply_candidate_duplicate_policy(
            candidates,
            duplicate_policy,
            policy="annotate",
        )
        filtered = apply_candidate_duplicate_policy(candidates, duplicate_policy)

        np.testing.assert_allclose(
            annotated["matgpr_duplicate_distance"].to_numpy(),
            [np.sqrt(0.02), 1.0],
        )
        self.assertEqual(annotated["matgpr_is_duplicate"].tolist(), [True, False])
        self.assertEqual(annotated["matgpr_duplicate_reason"].tolist(), ["feature", ""])
        self.assertEqual(filtered["material_id"].tolist(), ["far_candidate"])

    def test_candidate_domain_policy_validation_errors_are_explicit(self):
        candidates = pd.DataFrame({"material_id": ["a"], "descriptor_x": [0.0]})

        with self.assertRaises(ValueError):
            CandidateTrustRegion(
                centers=pd.DataFrame({"descriptor_x": [0.0]}),
                radius=-1.0,
            )
        with self.assertRaises(ValueError):
            CandidateDuplicatePolicy(existing_candidates=candidates)
        with self.assertRaises(ValueError):
            CandidateDuplicatePolicy(
                existing_candidates=candidates,
                feature_tolerance=-0.1,
            )
        with self.assertRaises(ValueError):
            apply_candidate_trust_region(
                candidates,
                CandidateTrustRegion(
                    centers=pd.DataFrame({"descriptor_x": [0.0]}),
                    radius=1.0,
                ),
                policy="drop",
            )
        with self.assertRaises(ValueError):
            apply_candidate_duplicate_policy(
                candidates,
                CandidateDuplicatePolicy(
                    existing_candidates=pd.DataFrame({"material_id": ["a"]}),
                    key_columns=("material_id",),
                ),
                policy="drop",
            )

    def test_select_diverse_batch_can_return_score_only_top_k(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["a", "b", "c"],
                "matgpr_acquisition": [0.8, 1.0, 0.9],
            }
        )

        batch = select_diverse_batch(
            candidates,
            top_k=2,
            diversity_weight=0.0,
        )

        self.assertEqual(batch["material_id"].tolist(), ["b", "c"])
        self.assertEqual(batch["matgpr_batch_order"].tolist(), [1, 2])
        self.assertTrue(batch["matgpr_batch_selected"].all())

    def test_select_diverse_batch_prefers_distant_candidates_when_weighted(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["near_best", "near_duplicate", "medium", "far"],
                "matgpr_acquisition": [1.0, 0.99, 0.92, 0.85],
                "descriptor_x": [0.0, 0.02, 5.0, 10.0],
            }
        )

        batch = select_diverse_batch(
            candidates,
            top_k=2,
            feature_columns=("descriptor_x",),
            diversity_weight=1.0,
            standardize_features=False,
        )

        self.assertEqual(batch["material_id"].tolist(), ["near_best", "far"])
        self.assertGreater(batch["matgpr_diversity_distance"].iloc[1], 9.0)

    def test_select_diverse_batch_min_distance_can_return_smaller_batch(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["a", "b", "c"],
                "matgpr_acquisition": [1.0, 0.95, 0.90],
                "descriptor_x": [0.0, 0.1, 0.2],
            }
        )

        batch = select_diverse_batch(
            candidates,
            top_k=3,
            feature_columns=("descriptor_x",),
            min_distance=1.0,
            standardize_features=False,
        )

        self.assertEqual(batch["material_id"].tolist(), ["a"])

    def test_select_diverse_batch_can_return_full_annotated_table(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["a", "b", "c"],
                "matgpr_acquisition": [1.0, 0.9, 0.8],
            }
        )

        annotated = select_diverse_batch(
            candidates,
            top_k=2,
            diversity_weight=0.0,
            return_all=True,
        )

        self.assertEqual(annotated["matgpr_batch_selected"].tolist(), [True, True, False])
        self.assertEqual(annotated["material_id"].tolist(), ["a", "b", "c"])

    def test_summarize_bo_recommendation_audit_reports_scores_and_policies(self):
        ranked = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c", "d"],
                "matgpr_rank": [1, 2, 3, 4],
                "matgpr_acquisition": [0.9, 0.7, 0.4, 0.1],
                "matgpr_predicted_mean": [1.8, 1.5, 1.1, 0.8],
                "matgpr_predicted_std": [0.2, 0.5, 0.1, 0.4],
                "matgpr_feasible": [True, False, True, True],
                "matgpr_constraint_violations": ["", "temperature_window", "", ""],
                "matgpr_in_trust_region": [True, False, True, True],
                "matgpr_trust_region_distance": [0.1, 1.2, 0.3, 0.4],
                "matgpr_is_duplicate": [False, False, True, False],
                "matgpr_duplicate_reason": ["", "", "key", ""],
                "matgpr_batch_selected": [True, False, False, True],
                "matgpr_batch_order": [1, pd.NA, pd.NA, 2],
            }
        )
        recommendations = ranked.loc[[0, 3]].reset_index(drop=True)

        audit = summarize_bo_recommendation_audit(
            recommendations,
            ranked_candidates=ranked,
            candidate_count=6,
            identifier_columns=("candidate_id",),
        )

        self.assertIsInstance(audit, BORecommendationAudit)
        overview = audit.overview_frame().iloc[0]
        self.assertEqual(overview["matgpr_input_candidates"], 6)
        self.assertEqual(overview["matgpr_ranked_candidates"], 4)
        self.assertEqual(overview["matgpr_recommended_candidates"], 2)
        self.assertEqual(overview["matgpr_filtered_out_candidates"], 2)
        self.assertAlmostEqual(overview["matgpr_feasible_recommended_fraction"], 1.0)
        self.assertEqual(overview["matgpr_duplicate_recommended_count"], 0)

        policy_summary = audit.policy_summary_frame()
        self.assertEqual(
            set(policy_summary["policy"]),
            {
                "pre_rank_filtering",
                "constraints",
                "trust_region",
                "duplicate_policy",
                "batch_selection",
            },
        )
        constraint_row = policy_summary.loc[policy_summary["policy"] == "constraints"].iloc[0]
        self.assertEqual(constraint_row["fail_count"], 1)
        self.assertIn("temperature_window", constraint_row["details"])

        score_summary = audit.score_summary_frame()
        acquisition_row = score_summary.loc[
            score_summary["column"] == "matgpr_acquisition"
        ].iloc[0]
        self.assertAlmostEqual(acquisition_row["recommended_mean"], 0.5)
        self.assertAlmostEqual(acquisition_row["ranked_max"], 0.9)

        recommendation_audit = audit.recommendation_frame()
        self.assertIn("matgpr_audit_note", recommendation_audit.columns)
        self.assertIn("rank 1", recommendation_audit["matgpr_audit_note"].iloc[0])
        self.assertIn("not duplicate", recommendation_audit["matgpr_audit_note"].iloc[0])
        self.assertIn("matgpr_acquisition_percentile", recommendation_audit.columns)
        self.assertIn("matgpr_uncertainty_percentile", recommendation_audit.columns)

    def test_summarize_bo_recommendation_audit_accepts_bo_result(self):
        ranked = pd.DataFrame(
            {
                "candidate_id": ["a", "b"],
                "matgpr_rank": [1, 2],
                "matgpr_acquisition": [1.0, 0.5],
                "matgpr_predicted_mean": [2.0, 1.0],
                "matgpr_predicted_std": [0.2, 0.3],
            }
        )
        surrogate = BoTorchSurrogate(
            model=object(),
            train_X=np.zeros((2, 1)),
            train_y=np.array([[0.0], [1.0]]),
            objective_y=np.array([[0.0], [1.0]]),
            maximize=True,
            best_observed_objective=1.0,
            feature_names=("x",),
        )
        result = BayesianOptimizationResult(
            recommendations=ranked.head(1),
            ranked_candidates=ranked,
            surrogate=surrogate,
            acquisition_function="expected_improvement",
            maximize=True,
            top_k=1,
        )

        audit = summarize_bo_recommendation_audit(result)
        overview = audit.overview_frame().iloc[0]

        self.assertEqual(overview["matgpr_acquisition_function"], "expected_improvement")
        self.assertEqual(overview["matgpr_top_k"], 1)
        self.assertEqual(overview["matgpr_objective_mode"], "single_objective")

    def test_summarize_bo_recommendation_audit_validation_errors_are_explicit(self):
        ranked = pd.DataFrame(
            {
                "candidate_id": ["a", "b"],
                "matgpr_acquisition": [1.0, 0.5],
            }
        )

        with self.assertRaises(TypeError):
            summarize_bo_recommendation_audit([{"candidate_id": "a"}])
        with self.assertRaises(ValueError):
            summarize_bo_recommendation_audit(
                ranked.head(1),
                ranked_candidates=ranked,
                candidate_count=1,
            )
        with self.assertRaises(ValueError):
            summarize_bo_recommendation_audit(
                ranked.head(1),
                ranked_candidates=ranked,
                identifier_columns=("missing",),
            )

    def test_botorch_missing_dependency_has_clear_install_message(self):
        with patch(
            "matgpr.optional_dependencies.importlib.import_module",
            side_effect=ImportError("missing botorch"),
        ):
            with self.assertRaises(ImportError) as context:
                fit_botorch_surrogate(
                    pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
                    pd.Series([0.0, 1.0, 0.5]),
                )

        message = str(context.exception)
        self.assertIn("BoTorch Bayesian optimization", message)
        self.assertIn("optional dependency `botorch`", message)
        self.assertIn("matgpr[bo]", message)

    def test_multi_objective_botorch_missing_dependency_has_clear_install_message(self):
        with patch(
            "matgpr.optional_dependencies.importlib.import_module",
            side_effect=ImportError("missing botorch"),
        ):
            with self.assertRaises(ImportError) as context:
                fit_multi_objective_botorch_surrogate(
                    pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
                    pd.DataFrame(
                        {
                            "performance": [0.0, 1.0, 0.5],
                            "cost": [4.0, 3.0, 2.0],
                        }
                    ),
                )

        message = str(context.exception)
        self.assertIn("BoTorch Bayesian optimization", message)
        self.assertIn("optional dependency `botorch`", message)
        self.assertIn("matgpr[bo]", message)

    def test_rank_discrete_candidates_validates_feature_count_before_botorch_use(self):
        fake_train_X = types.SimpleNamespace(shape=(3, 2))
        surrogate = BoTorchSurrogate(
            model=object(),
            train_X=fake_train_X,
            train_y=np.array([[0.0], [1.0], [2.0]]),
            objective_y=np.array([[0.0], [1.0], [2.0]]),
            maximize=True,
            best_observed_objective=2.0,
            feature_names=("x0", "x1"),
        )

        with self.assertRaises(ValueError) as context:
            rank_discrete_candidates(
                surrogate,
                pd.DataFrame({"only_one_feature": [0.5, 1.5]}),
            )

        self.assertIn("same number of features", str(context.exception))

    def test_rank_multi_objective_candidates_validates_feature_count_before_botorch_use(self):
        fake_train_X = types.SimpleNamespace(shape=(3, 2))
        surrogate = types.SimpleNamespace(train_X=fake_train_X)

        with self.assertRaises(ValueError) as context:
            rank_multi_objective_discrete_candidates(
                surrogate,
                pd.DataFrame({"only_one_feature": [0.5, 1.5]}),
            )

        self.assertIn("same number of features", str(context.exception))

    def test_candidate_constraint_validation_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            CandidateConstraint(name="", column="x", lower_bound=0.0)
        with self.assertRaises(ValueError):
            CandidateConstraint(name="missing_rule", column="", lower_bound=0.0)
        with self.assertRaises(ValueError):
            CandidateConstraint(name="empty_rule", column="x")
        with self.assertRaises(ValueError):
            CandidateConstraint(name="bad_bounds", column="x", lower_bound=2.0, upper_bound=1.0)
        with self.assertRaises(ValueError):
            apply_candidate_constraints(
                pd.DataFrame({"x": [1.0]}),
                CandidateConstraint(name="missing_column", column="y", lower_bound=0.0),
            )

    def test_select_diverse_batch_validation_errors_are_explicit(self):
        candidates = pd.DataFrame(
            {
                "matgpr_acquisition": [1.0, 0.9],
                "descriptor": [0.0, 1.0],
            }
        )

        with self.assertRaises(ValueError):
            select_diverse_batch(candidates, top_k=0)
        with self.assertRaises(ValueError):
            select_diverse_batch(candidates, top_k=1, diversity_weight=-0.1)
        with self.assertRaises(ValueError):
            select_diverse_batch(candidates, top_k=1, min_distance=-1.0)
        with self.assertRaises(ValueError):
            select_diverse_batch(candidates.drop(columns=["matgpr_acquisition"]), top_k=1)
        with self.assertRaises(ValueError):
            select_diverse_batch(
                candidates,
                top_k=1,
                feature_columns=("missing_descriptor",),
            )
        with self.assertRaises(ValueError):
            select_diverse_batch(
                pd.DataFrame({"matgpr_acquisition": [1.0], "label": ["not_numeric"]}),
                top_k=1,
                feature_columns=("label",),
            )

    def test_rank_discrete_candidates_rejects_invalid_constraint_policy_before_botorch_use(self):
        fake_train_X = types.SimpleNamespace(shape=(3, 1))
        surrogate = BoTorchSurrogate(
            model=object(),
            train_X=fake_train_X,
            train_y=np.array([[0.0], [1.0], [2.0]]),
            objective_y=np.array([[0.0], [1.0], [2.0]]),
            maximize=True,
            best_observed_objective=2.0,
            feature_names=("x",),
        )

        with self.assertRaises(ValueError) as context:
            rank_discrete_candidates(
                surrogate,
                pd.DataFrame({"x": [0.5, 1.5]}),
                constraint_policy="drop",
            )

        self.assertIn("constraint_policy", str(context.exception))

    def test_suggest_next_experiments_rejects_invalid_batch_selection_before_botorch_use(self):
        with self.assertRaises(ValueError) as context:
            suggest_next_experiments(
                pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
                pd.Series([0.0, 1.0, 0.5]),
                pd.DataFrame({"x": [0.25, 0.75]}),
                batch_selection="clustered",
            )

        self.assertIn("batch_selection", str(context.exception))

        with self.assertRaises(ValueError) as context:
            suggest_next_experiments(
                pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
                pd.Series([0.0, 1.0, 0.5]),
                pd.DataFrame({"x": [0.25, 0.75]}),
                batch_selection="sequential",
            )

        self.assertIn("'top' or 'diverse'", str(context.exception))

    def test_suggest_multi_objective_next_experiments_rejects_invalid_batch_selection_before_botorch_use(self):
        with self.assertRaises(ValueError) as context:
            suggest_multi_objective_next_experiments(
                pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
                pd.DataFrame(
                    {
                        "performance": [0.0, 1.0, 0.5],
                        "cost": [4.0, 3.0, 2.0],
                    }
                ),
                pd.DataFrame({"x": [0.25, 0.75]}),
                batch_selection="clustered",
            )

        self.assertIn("batch_selection", str(context.exception))

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_suggest_next_experiments_ranks_candidate_pool_when_botorch_is_available(self):
        X_train = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
        y_train = pd.Series([0.0, 0.4, 1.0])
        X_candidates = pd.DataFrame({"x": [0.25, 0.75, 1.25]})
        candidate_data = pd.DataFrame(
            {
                "material_id": ["candidate_a", "candidate_b", "candidate_c"],
            }
        )

        result = suggest_next_experiments(
            X_train,
            y_train,
            X_candidates,
            candidate_data=candidate_data,
            top_k=2,
            acquisition_function="log_expected_improvement",
            fit_model=False,
        )

        self.assertIsInstance(result, BayesianOptimizationResult)
        self.assertEqual(result.recommendations.shape[0], 2)
        self.assertEqual(result.ranked_candidates.shape[0], 3)
        self.assertIn("material_id", result.recommendations.columns)
        self.assertIn("matgpr_predicted_mean", result.recommendations.columns)
        self.assertIn("matgpr_predicted_std", result.recommendations.columns)
        self.assertIn("matgpr_acquisition", result.recommendations.columns)

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_suggest_next_experiments_applies_duplicate_policy(self):
        X_train = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
        y_train = pd.Series([0.0, 0.4, 1.0])
        X_candidates = pd.DataFrame({"x": [0.25, 0.75, 1.25]})
        candidate_data = pd.DataFrame(
            {
                "material_id": ["already_seen", "candidate_b", "candidate_c"],
            }
        )
        duplicate_policy = CandidateDuplicatePolicy(
            existing_candidates=pd.DataFrame({"material_id": ["already_seen"]}),
            key_columns=("material_id",),
        )

        result = suggest_next_experiments(
            X_train,
            y_train,
            X_candidates,
            candidate_data=candidate_data,
            top_k=3,
            duplicate_policy=duplicate_policy,
            fit_model=False,
        )

        self.assertEqual(result.ranked_candidates.shape[0], 2)
        self.assertNotIn("already_seen", result.ranked_candidates["material_id"].tolist())
        self.assertEqual(result.recommendations.shape[0], 2)
        self.assertIn("matgpr_is_duplicate", result.ranked_candidates.columns)

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_fit_botorch_surrogate_stores_known_noise_variance(self):
        X_train = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
        y_train = pd.Series([0.0, 0.4, 1.0])
        noise = observation_noise_variance(
            pd.DataFrame({"target_std": [0.1, 0.2, 0.3]}),
            std_column="target_std",
        )

        surrogate = fit_botorch_surrogate(
            X_train,
            y_train,
            noise_variance=noise,
            fit_model=False,
        )

        self.assertIsNotNone(surrogate.noise_variance)
        np.testing.assert_allclose(
            surrogate.noise_variance.detach().cpu().numpy().reshape(-1),
            [0.01, 0.04, 0.09],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_suggest_next_experiments_supports_noisy_expected_improvement(self):
        X_train = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
        y_train = pd.Series([0.0, 0.4, 1.0])
        X_candidates = pd.DataFrame({"x": [0.25, 0.75]})
        noise = observation_noise_variance(
            pd.DataFrame({"target_std": [0.1, 0.1, 0.1]}),
            std_column="target_std",
        )

        result = suggest_next_experiments(
            X_train,
            y_train,
            X_candidates,
            noise_variance=noise,
            acquisition_function="log_noisy_expected_improvement",
            top_k=1,
            fit_model=False,
        )

        self.assertEqual(result.recommendations.shape[0], 1)
        self.assertIn("matgpr_acquisition", result.recommendations.columns)

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_suggest_multi_objective_next_experiments_ranks_candidate_pool(self):
        X_train = pd.DataFrame({"x": [0.0, 0.3, 0.7, 1.0]})
        y_train = pd.DataFrame(
            {
                "performance": [0.1, 0.4, 0.8, 1.0],
                "cost": [4.0, 2.5, 2.0, 3.0],
            }
        )
        X_candidates = pd.DataFrame({"x": [0.15, 0.55, 0.9]})
        candidate_data = pd.DataFrame({"material_id": ["a", "b", "c"]})

        result = suggest_multi_objective_next_experiments(
            X_train,
            y_train,
            X_candidates,
            objective_directions=("maximize", "minimize"),
            candidate_data=candidate_data,
            top_k=2,
            acquisition_function="q_log_expected_hypervolume_improvement",
            fit_model=False,
            mc_samples=16,
            sampler_seed=7,
        )

        self.assertIsInstance(result, MultiObjectiveBayesianOptimizationResult)
        self.assertEqual(result.recommendations.shape[0], 2)
        self.assertEqual(result.ranked_candidates.shape[0], 3)
        self.assertEqual(result.objective_names, ("performance", "cost"))
        self.assertEqual(result.objective_directions, ("maximize", "minimize"))
        self.assertIn("matgpr_predicted_mean_performance", result.recommendations.columns)
        self.assertIn("matgpr_predicted_std_cost", result.recommendations.columns)
        self.assertIn("matgpr_predicted_pareto_front", result.ranked_candidates.columns)
        self.assertIn("matgpr_acquisition", result.recommendations.columns)
        self.assertIn("material_id", result.recommendations.columns)

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_suggest_multi_objective_next_experiments_supports_sequential_batch(self):
        X_train = pd.DataFrame({"x": [0.0, 0.3, 0.7, 1.0]})
        y_train = pd.DataFrame(
            {
                "performance": [0.1, 0.4, 0.8, 1.0],
                "cost": [4.0, 2.5, 2.0, 3.0],
            }
        )
        X_candidates = pd.DataFrame({"x": [0.15, 0.35, 0.55, 0.9]})
        candidate_data = pd.DataFrame({"material_id": ["a", "b", "c", "d"]})

        result = suggest_multi_objective_next_experiments(
            X_train,
            y_train,
            X_candidates,
            objective_directions=("maximize", "minimize"),
            candidate_data=candidate_data,
            top_k=3,
            batch_selection="greedy",
            acquisition_function="q_log_expected_hypervolume_improvement",
            fit_model=False,
            mc_samples=16,
            sampler_seed=11,
        )

        self.assertEqual(result.recommendations.shape[0], 3)
        self.assertEqual(result.recommendations["matgpr_batch_order"].tolist(), [1, 2, 3])
        self.assertTrue(result.recommendations["matgpr_batch_selected"].all())
        self.assertTrue(np.isfinite(result.recommendations["matgpr_batch_score"]).all())
        self.assertEqual(result.recommendations["material_id"].nunique(), 3)
        self.assertIn("matgpr_rank", result.recommendations.columns)

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_select_sequential_multi_objective_batch_can_return_annotated_pool(self):
        X_train = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
        y_train = pd.DataFrame(
            {
                "strength": [1.0, 2.0, 3.0],
                "toxicity": [0.5, 0.4, 0.8],
            }
        )
        X_candidates = pd.DataFrame({"x": [0.2, 0.6, 0.9]})
        candidate_data = pd.DataFrame({"material_id": ["a", "b", "c"]})
        surrogate = fit_multi_objective_botorch_surrogate(
            X_train,
            y_train,
            objective_directions=("maximize", "minimize"),
            fit_model=False,
        )

        annotated = select_sequential_multi_objective_batch(
            surrogate,
            X_candidates,
            top_k=2,
            candidate_data=candidate_data,
            mc_samples=16,
            sampler_seed=13,
            return_all=True,
        )

        selected = annotated[annotated["matgpr_batch_selected"]]

        self.assertEqual(annotated.shape[0], 3)
        self.assertEqual(selected["matgpr_batch_order"].tolist(), [1, 2])
        self.assertIn("matgpr_predicted_mean_strength", annotated.columns)
        self.assertIn("matgpr_predicted_std_toxicity", annotated.columns)
        self.assertTrue(np.isfinite(selected["matgpr_batch_score"]).all())

    @unittest.skipUnless(
        importlib.util.find_spec("botorch") is not None,
        "BoTorch is optional and not installed",
    )
    def test_multi_objective_surrogate_stores_noise_and_reference_point(self):
        X_train = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
        y_train = pd.DataFrame(
            {
                "strength": [1.0, 2.0, 3.0],
                "toxicity": [0.5, 0.4, 0.8],
            }
        )
        noise = pd.DataFrame(
            {
                "strength_noise": [0.01, 0.01, 0.02],
                "toxicity_noise": [0.001, 0.002, 0.003],
            }
        )

        surrogate = fit_multi_objective_botorch_surrogate(
            X_train,
            y_train,
            objective_directions=("maximize", "minimize"),
            reference_point=(0.0, 1.0),
            noise_variance=noise,
            fit_model=False,
        )

        self.assertEqual(surrogate.objective_names, ("strength", "toxicity"))
        self.assertIsNotNone(surrogate.noise_variance)
        np.testing.assert_allclose(
            surrogate.noise_variance.detach().cpu().numpy(),
            [[0.01, 0.001], [0.01, 0.002], [0.02, 0.003]],
        )
        np.testing.assert_allclose(
            surrogate.reference_point_original.detach().cpu().numpy(),
            [0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
