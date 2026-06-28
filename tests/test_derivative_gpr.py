from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    DerivativeObservationSet,
    MonotonicDerivativeConstraint,
    combine_derivative_observations,
    fit_derivative_constrained_gpr,
)


class DerivativeGPRTests(unittest.TestCase):
    def test_derivative_observation_set_validation_and_alpha(self):
        observations = DerivativeObservationSet(
            X=np.array([[0.0, 1.0], [1.0, 2.0]]),
            feature_indices=[0, 1],
            derivative_values=[1.0, -0.5],
            noise_std=[0.1, 0.2],
            labels=["dx0", "dx1"],
        )

        self.assertEqual(observations.n_observations, 2)
        self.assertEqual(observations.n_features, 2)
        self.assertTrue(np.array_equal(observations.feature_indices, np.array([0, 1])))
        self.assertTrue(np.allclose(observations.alpha, [0.01, 0.04]))
        self.assertTrue(np.array_equal(observations.labels, np.array(["dx0", "dx1"], dtype=object)))

    def test_monotonic_derivative_constraint_generates_dataframe_named_feature_observations(self):
        X = pd.DataFrame(
            {
                "temperature_k": [300.0, 400.0, 500.0],
                "descriptor": [0.1, 0.2, 0.3],
            }
        )
        constraint = MonotonicDerivativeConstraint(
            feature="temperature_k",
            direction="increasing",
            minimum_slope=0.01,
            noise_std=0.05,
            feature_min=350.0,
            label="positive_temperature_slope",
        )

        observations = constraint.generate(X)

        self.assertEqual(observations.n_observations, 2)
        self.assertTrue(np.allclose(observations.X[:, 0], [400.0, 500.0]))
        self.assertTrue(np.array_equal(observations.feature_indices, np.array([0, 0])))
        self.assertTrue(np.allclose(observations.derivative_values, [0.01, 0.01]))
        self.assertTrue(np.allclose(observations.alpha, 0.05**2))
        self.assertTrue(
            np.array_equal(
                observations.labels,
                np.array(["positive_temperature_slope", "positive_temperature_slope"], dtype=object),
            )
        )

    def test_derivative_observation_changes_prediction_direction(self):
        X_train = np.array([[0.0]])
        y_train = np.array([0.0])
        derivative_observations = DerivativeObservationSet(
            X=np.array([[0.0]]),
            feature_indices=0,
            derivative_values=[1.0],
            noise_std=1e-5,
        )

        derivative_model = fit_derivative_constrained_gpr(
            X_train,
            y_train,
            derivative_observations,
            length_scale=1.0,
            signal_variance=1.0,
            value_noise_std=1e-5,
            standardize_y=False,
        )
        unconstrained_model = fit_derivative_constrained_gpr(
            X_train,
            y_train,
            length_scale=1.0,
            signal_variance=1.0,
            value_noise_std=1e-5,
            standardize_y=False,
        )

        constrained_prediction = derivative_model.predict([[0.2]], confidence_level=0.95)
        unconstrained_prediction = unconstrained_model.predict([[0.2]])

        self.assertGreater(constrained_prediction.mean[0], 0.1)
        self.assertTrue(np.allclose(unconstrained_prediction.mean, [0.0]))
        self.assertGreater(constrained_prediction.std[0], 0.0)
        self.assertLess(constrained_prediction.lower[0], constrained_prediction.mean[0])
        self.assertGreater(constrained_prediction.upper[0], constrained_prediction.mean[0])

    def test_combines_derivative_observations_and_optimizes_hyperparameters(self):
        left = DerivativeObservationSet(
            X=np.array([[0.0], [0.5]]),
            feature_indices=0,
            derivative_values=[1.0, 1.0],
            noise_std=0.05,
            labels=["left", "left"],
        )
        right = DerivativeObservationSet(
            X=np.array([[1.0]]),
            feature_indices=0,
            derivative_values=[0.0],
            noise_std=0.1,
            labels=["right"],
        )
        combined = combine_derivative_observations(left, right)

        model = fit_derivative_constrained_gpr(
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            combined,
            length_scale=0.8,
            signal_variance=1.0,
            value_noise_std=0.05,
            optimize_hyperparameters=True,
            maxiter=5,
        )

        self.assertEqual(combined.n_observations, 3)
        self.assertTrue(np.array_equal(combined.labels, np.array(["left", "left", "right"], dtype=object)))
        self.assertEqual(model.length_scale.shape, (1,))
        self.assertGreater(model.signal_variance, 0.0)
        self.assertIsInstance(model.optimizer_success, bool)
        self.assertTrue(np.isfinite(model.log_marginal_likelihood))

    def test_validation_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            DerivativeObservationSet(
                X=np.array([[0.0]]),
                feature_indices=[0, 1],
                derivative_values=[1.0],
            )
        with self.assertRaises(ValueError):
            MonotonicDerivativeConstraint(feature=0, direction="sideways").generate(
                np.array([[0.0]])
            )
        with self.assertRaises(ValueError):
            fit_derivative_constrained_gpr(
                np.array([[0.0, 1.0]]),
                np.array([1.0]),
                DerivativeObservationSet(
                    X=np.array([[0.0]]),
                    feature_indices=0,
                    derivative_values=[1.0],
                ),
            )


if __name__ == "__main__":
    unittest.main()
