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
    fit_botorch_surrogate,
    rank_discrete_candidates,
    suggest_next_experiments,
)


class BayesianOptimizationApiTests(unittest.TestCase):
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
