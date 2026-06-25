from __future__ import annotations

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np
import torch

from matgpr.gpytorch_gpr import (
    GPyTorchGPRResult,
    PhysicsInformedMean,
    fit_gpytorch_gpr,
    predict_gpytorch_gpr,
    train_gpytorch_gpr,
)


def linear_physics_equation(features, parameters):
    return parameters["slope"] * features["temperature_c"] + parameters["intercept"]


class PhysicsInformedMeanTests(unittest.TestCase):
    def test_restores_feature_units_and_standardizes_target(self):
        mean = PhysicsInformedMean(
            equation=linear_physics_equation,
            feature_indices={"temperature_c": 0},
            learnable_parameters={"slope": 2.0},
            positive_parameters=("slope",),
            fixed_parameters={"intercept": 1.0},
            feature_means={"temperature_c": 100.0},
            feature_stds={"temperature_c": 10.0},
            target_mean=50.0,
            target_std=5.0,
        )

        x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        expected = torch.tensor(
            [(201.0 - 50.0) / 5.0, (221.0 - 50.0) / 5.0],
            dtype=torch.float64,
        )

        self.assertTrue(torch.allclose(mean(x), expected, atol=1e-6))
        self.assertAlmostEqual(mean.current_parameter_values()["slope"], 2.0, places=6)
        self.assertAlmostEqual(mean.current_parameter_values()["intercept"], 1.0, places=6)

    def test_rejects_positive_parameter_without_initial_value(self):
        with self.assertRaises(ValueError):
            PhysicsInformedMean(
                equation=linear_physics_equation,
                feature_indices={"temperature_c": 0},
                positive_parameters=("slope",),
            )

    def test_rejects_equation_output_with_wrong_shape(self):
        mean = PhysicsInformedMean(
            equation=lambda features, parameters: torch.tensor([1.0]),
            feature_indices={"x": 0},
        )

        with self.assertRaises(ValueError):
            mean(torch.zeros(3, 1))


class GPyTorchTrainingTests(unittest.TestCase):
    def test_fit_result_predicts_in_original_target_units(self):
        x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = 1.5 * x.ravel() + 0.2

        result = fit_gpytorch_gpr(
            x,
            y,
            kernel="rbf",
            training_iter=3,
            initial_noise=0.05,
            verbose=False,
        )
        prediction = result.predict(x[:2], confidence_level=0.90)

        self.assertIsInstance(result, GPyTorchGPRResult)
        self.assertEqual(len(result.loss_history), 3)
        self.assertEqual(prediction.mean.shape, (2,))
        self.assertEqual(prediction.std.shape, (2,))
        self.assertEqual(prediction.lower.shape, (2,))
        self.assertEqual(prediction.upper.shape, (2,))

    def test_train_wrapper_preserves_tuple_return(self):
        x = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        y = x.ravel() ** 2

        model, likelihood = train_gpytorch_gpr(
            x,
            y,
            kernel="matern",
            training_iter=2,
            verbose=False,
        )
        mean, std = predict_gpytorch_gpr(model, likelihood, x[:2])

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
