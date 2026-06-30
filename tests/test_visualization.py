from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

from matgpr import (
    plot_bo_benchmark_trace,
    plot_bo_campaign_progress,
    plot_bo_regret_trace,
    plot_learning_curve,
)


class LearningCurvePlotTests(unittest.TestCase):
    def test_plot_learning_curve_supports_count_axis_and_train_test_splits(self):
        rows = pd.DataFrame(
            {
                "model": ["physics_gpr"] * 4,
                "train_size": [0.25, 0.25, 0.50, 0.50],
                "train_size_percent": [25.0, 25.0, 50.0, 50.0],
                "n_train": [4, 4, 8, 8],
                "train_RMSE": [0.20, 0.24, 0.16, 0.18],
                "test_RMSE": [0.42, 0.46, 0.31, 0.35],
            }
        )

        fig, ax, summary = plot_learning_curve(
            rows,
            metric="RMSE",
            split=("train", "test"),
            x_axis="count",
        )

        self.assertEqual(ax.get_xlabel(), "Train set size (samples)")
        self.assertEqual(ax.get_ylabel(), "RMSE")
        self.assertEqual(set(summary["metric_col"]), {"train_RMSE", "test_RMSE"})
        self.assertEqual(set(summary["n_train"]), {4, 8})
        self.assertEqual(summary.shape[0], 4)
        plt.close(fig)

    def test_plot_learning_curve_supports_explicit_metric_column(self):
        rows = pd.DataFrame(
            {
                "model": ["standard_gpr", "standard_gpr"],
                "train_size": [0.25, 0.50],
                "test_R2": [0.55, 0.72],
            }
        )

        fig, ax, summary = plot_learning_curve(rows, metric_col="test_R2")

        self.assertEqual(ax.get_xlabel(), "Train set size (%)")
        self.assertEqual(set(summary["train_size_percent"]), {25.0, 50.0})
        self.assertEqual(summary["metric_col"].unique().tolist(), ["test_R2"])
        plt.close(fig)

    def test_plot_learning_curve_requires_count_column_for_count_axis(self):
        rows = pd.DataFrame(
            {
                "model": ["standard_gpr"],
                "train_size": [0.25],
                "test_RMSE": [0.45],
            }
        )

        with self.assertRaises(KeyError):
            plot_learning_curve(rows, metric="RMSE", x_axis="count")

    def test_plot_bo_benchmark_trace_aggregates_repeated_strategies(self):
        history = pd.DataFrame(
            {
                "matgpr_strategy": [
                    "physics_prior",
                    "physics_prior",
                    "physics_prior",
                    "physics_prior",
                    "random",
                    "random",
                    "random",
                    "random",
                ],
                "matgpr_repeat": [0, 0, 1, 1, 0, 0, 1, 1],
                "matgpr_evaluation": [1, 2, 1, 2, 1, 2, 1, 2],
                "matgpr_best_so_far": [1.0, 1.4, 1.2, 1.5, 0.6, 1.1, 0.8, 1.0],
                "matgpr_simple_regret": [0.5, 0.1, 0.3, 0.0, 0.9, 0.4, 0.7, 0.5],
            }
        )

        fig, ax, summary = plot_bo_benchmark_trace(
            history,
            strategies=["physics_prior", "random"],
            std_style="errorbar",
        )

        self.assertEqual(ax.get_xlabel(), "Evaluated candidates")
        self.assertEqual(ax.get_ylabel(), "Best value found")
        self.assertEqual(set(summary["matgpr_strategy"]), {"physics_prior", "random"})
        physics_eval_1 = summary.loc[
            (summary["matgpr_strategy"] == "physics_prior")
            & (summary["matgpr_evaluation"] == 1.0)
        ].iloc[0]
        self.assertAlmostEqual(physics_eval_1["value_mean"], 1.1)
        self.assertEqual(physics_eval_1["n_runs"], 2)
        plt.close(fig)

    def test_plot_bo_regret_trace_uses_simple_regret_column(self):
        history = pd.DataFrame(
            {
                "matgpr_strategy": ["random", "random"],
                "matgpr_evaluation": [1, 2],
                "matgpr_best_so_far": [0.2, 0.5],
                "matgpr_simple_regret": [0.8, 0.5],
            }
        )

        fig, ax, summary = plot_bo_regret_trace(history)

        self.assertEqual(ax.get_ylabel(), "Simple regret")
        self.assertEqual(summary["value_column"].unique().tolist(), ["matgpr_simple_regret"])
        plt.close(fig)

    def test_plot_bo_campaign_progress_counts_record_types(self):
        log = pd.DataFrame(
            {
                "matgpr_campaign_id": ["c1", "c1", "c1", "c1", "c2"],
                "matgpr_iteration": [0, 0, 1, 1, 0],
                "matgpr_record_type": [
                    "recommendation",
                    "selection",
                    "recommendation",
                    "observation",
                    "recommendation",
                ],
                "candidate_id": ["a", "a", "b", "b", "x"],
            }
        )

        fig, ax, summary = plot_bo_campaign_progress(log, campaign_id="c1")

        self.assertEqual(ax.get_ylabel(), "Logged records")
        self.assertEqual(summary["matgpr_record_count"].sum(), 4)
        self.assertEqual(
            set(summary["matgpr_record_type"]),
            {"recommendation", "selection", "observation"},
        )
        plt.close(fig)

    def test_plot_bo_campaign_progress_tracks_best_observed_target(self):
        log = pd.DataFrame(
            {
                "matgpr_campaign_id": ["c1", "c1", "c1", "c1"],
                "matgpr_iteration": [0, 0, 1, 1],
                "matgpr_record_type": [
                    "observation",
                    "observation",
                    "observation",
                    "recommendation",
                ],
                "conductivity": [0.8, 1.1, 1.4, None],
            }
        )

        fig, ax, summary = plot_bo_campaign_progress(
            log,
            campaign_id="c1",
            target_column="conductivity",
            maximize=True,
        )

        self.assertEqual(ax.get_ylabel(), "conductivity")
        self.assertEqual(summary["matgpr_best_so_far"].tolist(), [1.1, 1.4])
        self.assertEqual(summary["matgpr_observation_count"].tolist(), [2, 1])
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
