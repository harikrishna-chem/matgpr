from __future__ import annotations

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np

from matgpr.heteroscedastic_gpr import (
    HeteroscedasticGPRResult,
    fit_heteroscedastic_gpr,
)


class HeteroscedasticGPRTests(unittest.TestCase):
    def test_fit_predict_returns_learned_noise_and_total_uncertainty(self):
        x = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y = np.sin(2.0 * np.pi * x).ravel()
        y = y + np.linspace(0.0, 0.2, x.shape[0])

        result = fit_heteroscedastic_gpr(
            x,
            y,
            signal_kernel="rbf",
            noise_kernel="rbf",
            signal_training_iter=2,
            noise_training_iter=2,
            signal_initial_noise=0.05,
            noise_initial_noise=0.05,
            verbose=False,
        )
        prediction = result.predict(x[:3], confidence_level=0.80)

        self.assertIsInstance(result, HeteroscedasticGPRResult)
        self.assertEqual(result.residuals.shape, (10,))
        self.assertEqual(result.log_noise_variance_targets.shape, (10,))
        self.assertEqual(prediction.mean.shape, (3,))
        self.assertEqual(prediction.std.shape, (3,))
        self.assertEqual(prediction.latent_std.shape, (3,))
        self.assertEqual(prediction.noise_std.shape, (3,))
        self.assertEqual(prediction.noise_variance.shape, (3,))
        self.assertEqual(prediction.log_noise_variance.shape, (3,))
        self.assertEqual(prediction.lower.shape, (3,))
        self.assertEqual(prediction.upper.shape, (3,))
        self.assertTrue(np.all(prediction.noise_variance > 0.0))
        self.assertTrue(np.all(prediction.std >= prediction.latent_std))

    def test_predict_without_total_std_still_returns_noise_estimate(self):
        x = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        y = x.ravel() ** 2

        result = fit_heteroscedastic_gpr(
            x,
            y,
            signal_training_iter=2,
            noise_training_iter=2,
            verbose=False,
        )
        prediction = result.predict(x[:2], return_std=False)

        self.assertEqual(prediction.mean.shape, (2,))
        self.assertIsNone(prediction.std)
        self.assertIsNone(prediction.latent_std)
        self.assertEqual(prediction.noise_std.shape, (2,))
        self.assertEqual(prediction.noise_variance.shape, (2,))

    def test_rejects_invalid_variance_floor(self):
        x = np.linspace(0.0, 1.0, 4).reshape(-1, 1)
        y = x.ravel()

        with self.assertRaises(ValueError):
            fit_heteroscedastic_gpr(
                x,
                y,
                residual_variance_floor=0.0,
                signal_training_iter=2,
                noise_training_iter=2,
                verbose=False,
            )

        with self.assertRaises(ValueError):
            fit_heteroscedastic_gpr(
                x,
                y,
                noise_variance_floor=-1.0,
                signal_training_iter=2,
                noise_training_iter=2,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
