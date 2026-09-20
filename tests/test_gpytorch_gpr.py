from __future__ import annotations

import contextlib
import io
import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import gpytorch
import numpy as np
import torch

from matgpr.gpytorch_gpr import (
    ExactGPRModel,
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

    def test_rejects_feature_index_outside_training_matrix_width(self):
        mean = PhysicsInformedMean(
            equation=linear_physics_equation,
            feature_indices={"temperature_c": 2},
            learnable_parameters={"slope": 1.0, "intercept": 0.0},
        )

        with self.assertRaises(ValueError):
            fit_gpytorch_gpr(
                np.zeros((4, 1)),
                np.arange(4.0),
                mean_module=mean,
                training_iter=1,
                verbose=False,
            )


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

    def test_fit_rejects_nonfinite_training_values(self):
        x = np.array([[0.0], [1.0], [np.nan]])
        y = np.array([0.0, 1.0, 2.0])

        with self.assertRaises(ValueError):
            fit_gpytorch_gpr(x, y, training_iter=1, verbose=False)


class TargetStandardizationBufferTests(unittest.TestCase):
    """Target standardization must travel with the model, in original units."""

    @staticmethod
    def _training_data():
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 2))
        # Target far from zero and widely scaled, so standardized predictions
        # are obviously different from predictions in original units.
        y = 500.0 + 50.0 * x[:, 0]
        return x, y

    def test_target_standardization_is_registered_in_state_dict(self):
        x, y = self._training_data()
        result = fit_gpytorch_gpr(x, y, training_iter=5, verbose=False)

        state_dict = result.model.state_dict()

        self.assertIn("target_mean", state_dict)
        self.assertIn("target_std", state_dict)

    def test_predictions_stay_in_original_units_after_state_dict_roundtrip(self):
        x, y = self._training_data()
        result = fit_gpytorch_gpr(x, y, training_iter=5, verbose=False)
        expected, _ = predict_gpytorch_gpr(result.model, result.likelihood, x[:3])

        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        reloaded = ExactGPRModel(
            torch.tensor(x).double(),
            torch.tensor((y - y.mean()) / y.std()).double(),
            likelihood,
            ard_num_dims=2,
        ).double()
        reloaded.load_state_dict(result.model.state_dict())
        actual, _ = predict_gpytorch_gpr(reloaded, likelihood, x[:3])

        np.testing.assert_allclose(actual, expected)

    def test_predict_rejects_model_without_target_standardization(self):
        x, y = self._training_data()
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        foreign = _PlainExactGP(
            torch.tensor(x).double(),
            torch.tensor(y).double(),
            likelihood,
        ).double()

        with self.assertRaisesRegex(ValueError, "target_mean"):
            predict_gpytorch_gpr(foreign, likelihood, x[:3])


class _PlainExactGP(gpytorch.models.ExactGP):
    """A GPyTorch model built outside matgpr, without standardization buffers."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class TrainingVerbosityTests(unittest.TestCase):
    def test_fit_is_quiet_by_default_and_logs_when_verbose(self):
        x = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
        y = 2.0 * x[:, 0]

        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet):
            fit_gpytorch_gpr(x, y, training_iter=10)

        loud = io.StringIO()
        with contextlib.redirect_stdout(loud):
            fit_gpytorch_gpr(x, y, training_iter=10, verbose=True, log_every=5)

        self.assertEqual(quiet.getvalue(), "")
        self.assertIn("Loss", loud.getvalue())


if __name__ == "__main__":
    unittest.main()
