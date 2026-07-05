from __future__ import annotations

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np

from matgpr import MultiFidelityGPRRegressor, fit_delta_multifidelity_gpr
from matgpr.multifidelity_gpr import DeltaMultiFidelityGPRResult, MultiFidelityGPRPrediction


def low_fidelity_function(x: np.ndarray) -> np.ndarray:
    values = x.reshape(-1)
    return np.sin(2.0 * np.pi * values) + 0.5 * values


def high_fidelity_function(x: np.ndarray) -> np.ndarray:
    values = x.reshape(-1)
    low = low_fidelity_function(x)
    return 1.25 * low + 0.2 + 0.15 * values**2


class DeltaMultiFidelityGPRTests(unittest.TestCase):
    def test_fit_predict_with_supplied_low_fidelity_values(self):
        x_high = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y_high = high_fidelity_function(x_high)
        low_high = low_fidelity_function(x_high)

        result = fit_delta_multifidelity_gpr(
            x_high,
            y_high,
            low_fidelity_high=low_high,
            correction_kernel="rbf",
            training_iter=3,
            verbose=False,
        )
        prediction = result.predict(
            x_high[:3],
            low_fidelity=low_high[:3],
            confidence_level=0.90,
        )

        self.assertIsInstance(result, DeltaMultiFidelityGPRResult)
        self.assertIsNone(result.low_fidelity_model)
        self.assertGreater(result.rho, 0.0)
        self.assertEqual(result.correction_target.shape, (10,))
        self.assertIsInstance(prediction, MultiFidelityGPRPrediction)
        self.assertEqual(prediction.mean.shape, (3,))
        self.assertEqual(prediction.std.shape, (3,))
        self.assertEqual(prediction.lower.shape, (3,))
        self.assertEqual(prediction.correction_mean.shape, (3,))
        self.assertEqual(prediction.low_fidelity_mean.shape, (3,))

    def test_fit_predict_with_internal_low_fidelity_surrogate(self):
        x_low = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
        y_low = low_fidelity_function(x_low)
        x_high = np.linspace(0.05, 0.95, 8).reshape(-1, 1)
        y_high = high_fidelity_function(x_high)

        result = fit_delta_multifidelity_gpr(
            x_high,
            y_high,
            X_low=x_low,
            y_low=y_low,
            correction_kernel="rbf",
            low_fidelity_kernel="rbf",
            training_iter=3,
            low_fidelity_training_iter=3,
            verbose=False,
        )
        prediction = result.predict(x_high[:4], confidence_level=0.80)

        self.assertIsNotNone(result.low_fidelity_model)
        self.assertEqual(prediction.mean.shape, (4,))
        self.assertEqual(prediction.std.shape, (4,))
        self.assertEqual(prediction.low_fidelity_std.shape, (4,))
        self.assertTrue(np.all(prediction.std >= prediction.correction_std))

    def test_exact_linear_fidelity_map_allows_zero_correction_variance(self):
        x_high = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        low_high = low_fidelity_function(x_high)
        y_high = 1.25 * low_high + 0.2

        result = fit_delta_multifidelity_gpr(
            x_high,
            y_high,
            low_fidelity_high=low_high,
            correction_kernel="rbf",
            training_iter=2,
            verbose=False,
        )
        prediction = result.predict(
            x_high[:2],
            low_fidelity=low_high[:2],
            confidence_level=0.90,
        )

        self.assertEqual(result.correction_target.shape, (8,))
        self.assertTrue(np.allclose(result.correction_target, 0.0, atol=1e-12))
        self.assertEqual(prediction.mean.shape, (2,))
        self.assertEqual(prediction.std.shape, (2,))

    def test_estimator_fit_predict_with_supplied_low_fidelity_values(self):
        x_high = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
        y_high = high_fidelity_function(x_high)
        low_high = low_fidelity_function(x_high)

        estimator = MultiFidelityGPRRegressor(
            correction_kernel="rbf",
            training_iter=3,
            random_state=11,
        )
        estimator.fit(x_high, y_high, low_fidelity=low_high)
        prediction = estimator.predict_distribution(
            x_high[:2],
            low_fidelity=low_high[:2],
            confidence_level=0.90,
        )
        score = estimator.score(x_high, y_high, low_fidelity=low_high)

        self.assertEqual(estimator.n_features_in_, 1)
        self.assertGreater(estimator.rho_, 0.0)
        self.assertEqual(estimator.correction_target_.shape, (9,))
        self.assertEqual(prediction.mean.shape, (2,))
        self.assertEqual(prediction.upper.shape, (2,))
        self.assertTrue(np.isfinite(score))

    def test_validation_errors_are_explicit(self):
        x_high = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        y_high = high_fidelity_function(x_high)
        low_high = low_fidelity_function(x_high)

        with self.assertRaisesRegex(ValueError, "Provide low_fidelity_high"):
            fit_delta_multifidelity_gpr(
                x_high,
                y_high,
                training_iter=1,
                verbose=False,
            )

        with self.assertRaisesRegex(ValueError, "either low_fidelity_high"):
            fit_delta_multifidelity_gpr(
                x_high,
                y_high,
                low_fidelity_high=low_high,
                X_low=x_high,
                y_low=low_high,
                training_iter=1,
                verbose=False,
            )

        with self.assertRaisesRegex(ValueError, "nonzero variance"):
            fit_delta_multifidelity_gpr(
                x_high,
                y_high,
                low_fidelity_high=np.ones_like(low_high),
                training_iter=1,
                verbose=False,
            )

        with self.assertRaisesRegex(ValueError, "X_low has 2 features"):
            fit_delta_multifidelity_gpr(
                x_high,
                y_high,
                X_low=np.column_stack([x_high.ravel(), x_high.ravel() ** 2]),
                y_low=low_high,
                training_iter=1,
                verbose=False,
            )

        result = fit_delta_multifidelity_gpr(
            x_high,
            y_high,
            low_fidelity_high=low_high,
            training_iter=1,
            verbose=False,
        )
        with self.assertRaisesRegex(ValueError, "low_fidelity is required"):
            result.predict(x_high[:2])


if __name__ == "__main__":
    unittest.main()
