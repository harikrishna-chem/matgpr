from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

from matgpr import plot_learning_curve


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


if __name__ == "__main__":
    unittest.main()
