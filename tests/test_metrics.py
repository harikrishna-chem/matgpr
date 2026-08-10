from __future__ import annotations

import json
import unittest

import numpy as np

from matgpr import (
    json_safe_metrics,
    json_safe_regression_metrics,
    json_safe_train_test_regression_metrics,
    regression_metrics,
)


class MetricTests(unittest.TestCase):
    def test_regression_metrics_keep_nan_for_undefined_one_sample_metrics(self):
        metrics = regression_metrics([1.0], [1.2])

        self.assertTrue(np.isnan(metrics["R2"]))
        self.assertTrue(np.isnan(metrics["r"]))

    def test_json_safe_regression_metrics_serializes_one_sample_metrics(self):
        metrics = json_safe_regression_metrics([1.0], [1.2])

        self.assertIsNone(metrics["R2"])
        self.assertIsNone(metrics["r"])
        self.assertIn("metric_warnings", metrics)
        json.dumps(metrics, allow_nan=False)

    def test_json_safe_metrics_serializes_constant_target_correlation(self):
        metrics = regression_metrics([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        safe = json_safe_metrics(metrics)

        self.assertTrue(np.isnan(metrics["r"]))
        self.assertIsNone(safe["r"])
        json.dumps(safe, allow_nan=False)

    def test_json_safe_train_test_regression_metrics_serializes_nested_edge_cases(self):
        metrics = json_safe_train_test_regression_metrics(
            [1.0],
            [1.1],
            [2.0, 2.0],
            [2.0, 2.0],
        )

        self.assertIsNone(metrics["train_R2"])
        self.assertIsNone(metrics["train_r"])
        self.assertIsNone(metrics["test_r"])
        json.dumps(metrics, allow_nan=False)

    def test_json_safe_metrics_supports_nested_values(self):
        metrics = {
            "scalar": float("nan"),
            "nested": {"value": np.inf},
            "array": np.array([1.0, np.nan]),
        }

        safe = json_safe_metrics(metrics)

        self.assertIsNone(safe["scalar"])
        self.assertIsNone(safe["nested"]["value"])
        self.assertEqual(safe["array"], [1.0, None])
        json.dumps(safe, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
