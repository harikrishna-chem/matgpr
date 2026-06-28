from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import matplotlib.pyplot as plt
import numpy as np

from matgpr import (
    calibration_curve,
    gaussian_nlpd,
    interval_coverage,
    plot_uncertainty_calibration,
    plot_uncertainty_vs_error,
    prediction_interval_bounds,
    standardized_residuals,
    uncertainty_diagnostics,
    uncertainty_error_correlation,
)


class UncertaintyMetricTests(unittest.TestCase):
    def test_prediction_interval_bounds_match_normal_quantile(self):
        lower, upper = prediction_interval_bounds([0.0], [1.0], confidence_level=0.95)

        self.assertAlmostEqual(lower[0], -1.959963984540054, places=6)
        self.assertAlmostEqual(upper[0], 1.959963984540054, places=6)

    def test_interval_coverage_reports_observed_and_expected_values(self):
        y_true = np.array([0.0, 0.5, 2.5])
        y_pred = np.zeros(3)
        y_std = np.ones(3)

        metrics = interval_coverage(y_true, y_pred, y_std, confidence_level=0.68)

        self.assertEqual(metrics["confidence_level"], 0.68)
        self.assertAlmostEqual(metrics["observed_coverage"], 2 / 3)
        self.assertIn("mean_interval_width", metrics)

    def test_calibration_curve_has_expected_columns(self):
        curve = calibration_curve(
            [0.0, 0.1, 1.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            confidence_levels=[0.5, 0.9],
        )

        self.assertEqual(curve.shape[0], 2)
        self.assertEqual(curve["expected_coverage"].tolist(), [0.5, 0.9])
        self.assertIn("observed_coverage", curve.columns)
        self.assertIn("coverage_error", curve.columns)

    def test_calibration_curve_validates_confidence_levels(self):
        with self.assertRaises(ValueError):
            calibration_curve([0.0], [0.0], [1.0], confidence_levels=[])
        with self.assertRaises(ValueError):
            calibration_curve([0.0], [0.0], [1.0], confidence_levels=[0.5, 1.0])

    def test_gaussian_nlpd_supports_reductions(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 1.0])
        y_std = np.ones(2)

        values = gaussian_nlpd(y_true, y_pred, y_std, reduction="none")
        mean_value = gaussian_nlpd(y_true, y_pred, y_std)
        sum_value = gaussian_nlpd(y_true, y_pred, y_std, reduction="sum")

        self.assertEqual(values.shape, (2,))
        self.assertAlmostEqual(mean_value, 0.5 * np.log(2.0 * np.pi))
        self.assertAlmostEqual(sum_value, float(np.sum(values)))

    def test_standardized_residuals_and_diagnostics(self):
        y_true = np.array([0.0, 2.0, 4.0])
        y_pred = np.array([0.0, 1.0, 5.0])
        y_std = np.array([0.5, 1.0, 2.0])

        residuals = standardized_residuals(y_true, y_pred, y_std)
        diagnostics = uncertainty_diagnostics(y_true, y_pred, y_std, confidence_level=0.9)

        self.assertTrue(np.allclose(residuals, [0.0, 1.0, -0.5]))
        self.assertIn("NLPD", diagnostics)
        self.assertIn("uncertainty_error_spearman", diagnostics)
        self.assertEqual(diagnostics["expected_coverage"], 0.9)

    def test_uncertainty_error_correlation_and_validation(self):
        correlation = uncertainty_error_correlation(
            [0.0, 1.0, 2.0],
            [0.0, 1.5, 3.0],
            [0.1, 0.5, 1.0],
        )

        self.assertGreater(correlation, 0.0)
        with self.assertRaises(ValueError):
            prediction_interval_bounds([0.0], [0.0])
        with self.assertRaises(ValueError):
            interval_coverage([0.0], [0.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            uncertainty_error_correlation([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], method="kendall")


class UncertaintyPlotTests(unittest.TestCase):
    def test_uncertainty_calibration_plot_returns_curve(self):
        fig, ax, curve = plot_uncertainty_calibration(
            [0.0, 0.2, 1.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            confidence_levels=[0.5, 0.9],
        )

        self.assertEqual(curve.shape[0], 2)
        self.assertEqual(ax.get_xlabel(), "Expected coverage")
        plt.close(fig)

    def test_uncertainty_vs_error_plot_returns_diagnostics(self):
        fig, ax, diagnostics = plot_uncertainty_vs_error(
            [0.0, 1.0, 2.0],
            [0.0, 1.5, 3.0],
            [0.1, 0.5, 1.0],
        )

        self.assertIn("NLPD", diagnostics)
        self.assertEqual(ax.get_ylabel(), "Absolute prediction error")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
