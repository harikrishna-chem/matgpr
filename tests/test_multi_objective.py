from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    ObjectiveSpec,
    pareto_front_mask,
    rank_multi_objective_candidates,
    scalarize_objectives,
    select_pareto_front,
)


class MultiObjectiveTests(unittest.TestCase):
    def test_pareto_front_mask_handles_maximize_and_minimize_objectives(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["a", "b", "c", "d"],
                "predicted_strength": [10.0, 9.0, 8.0, 6.0],
                "estimated_cost": [5.0, 3.0, 8.0, 1.0],
            }
        )
        objectives = [
            ObjectiveSpec("strength", "predicted_strength", goal="maximize"),
            ObjectiveSpec("cost", "estimated_cost", goal="minimize"),
        ]

        mask = pareto_front_mask(candidates, objectives)
        front = select_pareto_front(candidates, objectives)

        self.assertEqual(mask.tolist(), [True, True, False, True])
        self.assertEqual(front["material_id"].tolist(), ["a", "b", "d"])

    def test_scalarize_objectives_returns_weighted_normalized_utility(self):
        candidates = pd.DataFrame(
            {
                "predicted_strength": [10.0, 8.0],
                "estimated_cost": [5.0, 1.0],
            }
        )
        objectives = [
            ObjectiveSpec("strength", "predicted_strength", goal="maximize", weight=0.75),
            ObjectiveSpec("cost", "estimated_cost", goal="minimize", weight=0.25),
        ]

        score = scalarize_objectives(candidates, objectives)

        np.testing.assert_allclose(score.to_numpy(), [0.75, 0.25])
        self.assertEqual(score.name, "matgpr_multi_objective_score")

    def test_rank_multi_objective_candidates_adds_scores_front_and_utilities(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["a", "b", "c", "d"],
                "predicted_strength": [10.0, 9.0, 8.0, 6.0],
                "estimated_cost": [5.0, 3.0, 8.0, 1.0],
            }
        )
        objectives = [
            ObjectiveSpec("Predicted Strength", "predicted_strength", "maximize", 0.7),
            ObjectiveSpec("Estimated Cost", "estimated_cost", "minimize", 0.3),
        ]

        ranked = rank_multi_objective_candidates(candidates, objectives)

        self.assertEqual(ranked["material_id"].tolist(), ["a", "b", "c", "d"])
        self.assertEqual(ranked["matgpr_multi_objective_rank"].tolist(), [1, 2, 3, 4])
        self.assertEqual(ranked["matgpr_pareto_front"].tolist(), [True, True, False, True])
        self.assertIn("matgpr_objective_predicted_strength_utility", ranked.columns)
        self.assertIn("matgpr_objective_estimated_cost_utility", ranked.columns)
        self.assertGreater(
            ranked["matgpr_multi_objective_score"].iloc[0],
            ranked["matgpr_multi_objective_score"].iloc[-1],
        )

    def test_rank_multi_objective_candidates_can_return_top_k_without_utilities(self):
        candidates = pd.DataFrame(
            {
                "material_id": ["a", "b", "c"],
                "performance": [0.8, 0.9, 0.1],
                "toxicity": [0.4, 0.9, 0.1],
            }
        )
        objectives = [
            ObjectiveSpec("performance", "performance", "maximize", 1.0),
            ObjectiveSpec("toxicity", "toxicity", "minimize", 1.0),
        ]

        ranked = rank_multi_objective_candidates(
            candidates,
            objectives,
            top_k=2,
            include_objective_utilities=False,
        )

        self.assertEqual(ranked.shape[0], 2)
        self.assertNotIn("matgpr_objective_performance_utility", ranked.columns)

    def test_constant_objective_receives_equal_normalized_utility(self):
        candidates = pd.DataFrame(
            {
                "performance": [1.0, 2.0],
                "cost": [5.0, 5.0],
            }
        )
        objectives = [
            ObjectiveSpec("performance", "performance", "maximize", 1.0),
            ObjectiveSpec("cost", "cost", "minimize", 1.0),
        ]

        ranked = rank_multi_objective_candidates(candidates, objectives)

        np.testing.assert_allclose(
            ranked["matgpr_objective_cost_utility"].to_numpy(),
            [1.0, 1.0],
        )

    def test_multi_objective_validation_errors_are_explicit(self):
        data = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})

        with self.assertRaises(ValueError):
            ObjectiveSpec("", "x")
        with self.assertRaises(ValueError):
            ObjectiveSpec("x", "")
        with self.assertRaises(ValueError):
            ObjectiveSpec("x", "x", goal="target")
        with self.assertRaises(ValueError):
            ObjectiveSpec("x", "x", weight=-1.0)
        with self.assertRaises(ValueError):
            scalarize_objectives(data, [ObjectiveSpec("x", "x", weight=0.0)])
        with self.assertRaises(ValueError):
            rank_multi_objective_candidates(data, [ObjectiveSpec("missing", "missing")])
        with self.assertRaises(ValueError):
            rank_multi_objective_candidates(data, [ObjectiveSpec("label", "label")])
        with self.assertRaises(ValueError):
            rank_multi_objective_candidates(
                data,
                [
                    ObjectiveSpec("a-b", "x"),
                    ObjectiveSpec("a b", "x2"),
                ],
            )
        with self.assertRaises(ValueError):
            rank_multi_objective_candidates(
                data,
                [ObjectiveSpec("x", "x")],
                score_column="same",
                pareto_column="same",
            )


if __name__ == "__main__":
    unittest.main()
