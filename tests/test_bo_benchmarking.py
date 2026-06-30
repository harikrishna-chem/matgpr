from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    BOBenchmarkComparison,
    BOBenchmarkResult,
    BOBenchmarkStrategy,
    compare_bo_strategies,
    simulate_bo_strategy,
    summarize_bo_benchmark,
)


class BayesianOptimizationBenchmarkTests(unittest.TestCase):
    def test_simulate_bo_strategy_replays_score_column_strategy(self):
        candidates = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c", "d"],
                "target": [0.2, 0.5, 1.0, 0.7],
                "expected_improvement": [0.1, 0.4, 0.9, 0.6],
            }
        )

        result = simulate_bo_strategy(
            candidates,
            BOBenchmarkStrategy(
                name="expected_improvement",
                score_column="expected_improvement",
            ),
            target_column="target",
            candidate_id_column="candidate_id",
            budget=2,
        )

        self.assertIsInstance(result, BOBenchmarkResult)
        self.assertEqual(result.history["matgpr_candidate_id"].tolist(), ["c", "d"])
        self.assertEqual(result.history["matgpr_best_so_far"].tolist(), [1.0, 1.0])
        self.assertEqual(result.summary.loc[0, "matgpr_optimum_candidate_id"], "c")
        self.assertEqual(result.summary.loc[0, "matgpr_hit_optimum"], True)
        self.assertEqual(result.summary.loc[0, "matgpr_evaluations_to_optimum"], 1)
        self.assertEqual(result.summary.loc[0, "matgpr_final_regret"], 0.0)

    def test_simulate_bo_strategy_supports_minimization_and_initial_observations(self):
        candidates = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c", "d"],
                "cost": [10.0, 5.0, 1.0, 3.0],
                "predicted_cost": [8.0, 4.0, 2.0, 3.5],
            }
        )

        result = simulate_bo_strategy(
            candidates,
            BOBenchmarkStrategy(
                name="low_predicted_cost",
                score_column="predicted_cost",
                direction="minimize",
            ),
            target_column="cost",
            maximize=False,
            candidate_id_column="candidate_id",
            initial_observed=("a",),
            budget=2,
            batch_size=2,
        )

        self.assertEqual(result.history["matgpr_candidate_id"].tolist(), ["c", "d"])
        self.assertEqual(result.history["matgpr_iteration"].tolist(), [1, 1])
        self.assertEqual(result.history["matgpr_best_before_batch"].tolist(), [10.0, 10.0])
        self.assertEqual(result.summary.loc[0, "matgpr_goal"], "minimize")
        self.assertEqual(result.summary.loc[0, "matgpr_final_best"], 1.0)
        self.assertEqual(result.summary.loc[0, "matgpr_final_regret"], 0.0)

    def test_simulate_bo_strategy_counts_initial_optimum_as_already_found(self):
        candidates = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c"],
                "target": [0.2, 1.0, 0.4],
                "score": [0.1, 0.2, 0.9],
            }
        )

        result = simulate_bo_strategy(
            candidates,
            BOBenchmarkStrategy("score", score_column="score"),
            target_column="target",
            candidate_id_column="candidate_id",
            initial_observed=("b",),
            budget=1,
        )

        self.assertEqual(result.summary.loc[0, "matgpr_hit_optimum"], True)
        self.assertEqual(result.summary.loc[0, "matgpr_evaluations_to_optimum"], 0)
        self.assertEqual(result.summary.loc[0, "matgpr_final_regret"], 0.0)

    def test_compare_bo_strategies_repeats_and_summarizes_results(self):
        candidates = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c", "d"],
                "target": [0.0, 0.4, 1.0, 0.8],
                "good_score": [0.0, 0.4, 1.0, 0.8],
                "bad_score": [1.0, 0.8, 0.1, 0.2],
            }
        )

        comparison = compare_bo_strategies(
            candidates,
            [
                BOBenchmarkStrategy("good", score_column="good_score"),
                BOBenchmarkStrategy("bad", score_column="bad_score"),
                BOBenchmarkStrategy("random"),
            ],
            target_column="target",
            candidate_id_column="candidate_id",
            budget=2,
            n_repeats=3,
            random_state=7,
        )
        summary = comparison.summary_by_strategy()

        self.assertIsInstance(comparison, BOBenchmarkComparison)
        self.assertEqual(comparison.summary.shape[0], 9)
        self.assertEqual(comparison.history["matgpr_repeat"].nunique(), 3)
        self.assertEqual(summary.loc[0, "matgpr_strategy"], "good")
        self.assertEqual(summary.loc[0, "matgpr_hit_optimum_rate"], 1.0)
        self.assertEqual(summary.loc[0, "matgpr_final_regret_mean"], 0.0)
        self.assertIn("random", summary["matgpr_strategy"].tolist())

    def test_simulate_bo_strategy_accepts_custom_score_function(self):
        candidates = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c"],
                "target": [0.1, 1.0, 0.5],
                "score": [0.2, 0.9, 0.4],
            }
        )

        def score_remaining(observed, remaining, rng):
            del observed, rng
            return remaining["score"]

        result = simulate_bo_strategy(
            candidates,
            BOBenchmarkStrategy("callable", score_function=score_remaining),
            target_column="target",
            candidate_id_column="candidate_id",
            budget=1,
        )

        self.assertEqual(result.history["matgpr_candidate_id"].tolist(), ["b"])
        self.assertEqual(result.summary.loc[0, "matgpr_hit_optimum"], True)

    def test_summarize_bo_benchmark_validation_errors_are_explicit(self):
        with self.assertRaises(TypeError):
            summarize_bo_benchmark([{"matgpr_strategy": "x"}])
        with self.assertRaises(ValueError):
            summarize_bo_benchmark(pd.DataFrame({"matgpr_strategy": ["x"]}))

    def test_bo_benchmark_validation_errors_are_explicit(self):
        candidates = pd.DataFrame(
            {
                "candidate_id": ["a", "a"],
                "target": [0.0, 1.0],
                "score": [0.1, 0.2],
            }
        )

        with self.assertRaises(ValueError):
            BOBenchmarkStrategy("", score_column="score")
        with self.assertRaises(ValueError):
            BOBenchmarkStrategy(
                "bad",
                score_column="score",
                score_function=lambda observed, remaining, rng: np.ones(len(remaining)),
            )
        with self.assertRaises(ValueError):
            simulate_bo_strategy(
                candidates,
                BOBenchmarkStrategy("score", score_column="score"),
                target_column="target",
                candidate_id_column="candidate_id",
            )
        with self.assertRaises(ValueError):
            simulate_bo_strategy(
                pd.DataFrame({"candidate_id": ["a"], "target": [np.nan]}),
                BOBenchmarkStrategy("random"),
                target_column="target",
                candidate_id_column="candidate_id",
            )
        with self.assertRaises(ValueError):
            simulate_bo_strategy(
                pd.DataFrame({"target": [0.0], "score": [1.0]}),
                BOBenchmarkStrategy("score", score_column="score"),
                target_column="target",
                budget=0,
            )
        with self.assertRaises(ValueError):
            simulate_bo_strategy(
                pd.DataFrame(
                    {
                        "candidate_id": ["a", "b"],
                        "target": [0.0, 1.0],
                        "score": [0.1, 0.2],
                    }
                ),
                BOBenchmarkStrategy("score", score_column="score"),
                target_column="target",
                candidate_id_column="candidate_id",
                initial_observed=("a", "a"),
            )
        with self.assertRaises(ValueError):
            compare_bo_strategies(
                pd.DataFrame({"target": [0.0], "score": [1.0]}),
                [
                    BOBenchmarkStrategy("duplicate", score_column="score"),
                    BOBenchmarkStrategy("duplicate", score_column="score"),
                ],
                target_column="target",
            )


if __name__ == "__main__":
    unittest.main()
