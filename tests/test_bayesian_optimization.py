from __future__ import annotations

import importlib.util
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from matgpr import (
    BayesianOptimizationResult,
    BoTorchSurrogate,
    CandidateConstraint,
    apply_candidate_constraints,
    fit_botorch_surrogate,
    rank_discrete_candidates,
    select_diverse_batch,
    suggest_next_experiments,
)


class BayesianOptimizationApiTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
