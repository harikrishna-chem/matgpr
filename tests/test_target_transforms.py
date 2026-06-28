from __future__ import annotations

import unittest

import numpy as np

from matgpr import (
    BoundedTargetTransform,
    IdentityTargetTransform,
    LogTargetTransform,
    PhysicsResidualTransform,
    StandardizedTargetTransform,
    make_target_transform,
)
from matgpr.gpytorch_gpr import GPyTorchPrediction


class TargetTransformTests(unittest.TestCase):
    def test_identity_transform_round_trip_prediction(self):
        transform = IdentityTargetTransform()
        prediction = GPyTorchPrediction(
            mean=np.array([1.0, 2.0]),
            std=np.array([0.1, 0.2]),
            lower=np.array([0.8, 1.6]),
            upper=np.array([1.2, 2.4]),
        )

        inverted = transform.inverse_prediction(prediction)

        self.assertTrue(np.allclose(transform.fit_transform([1.0, 2.0]), [1.0, 2.0]))
        self.assertTrue(np.allclose(inverted.mean, prediction.mean))
        self.assertTrue(np.allclose(inverted.std, prediction.std))
        self.assertTrue(np.allclose(inverted.lower, prediction.lower))
        self.assertTrue(np.allclose(inverted.upper, prediction.upper))

    def test_standardized_transform_round_trip_and_prediction(self):
        y = np.array([2.0, 4.0, 6.0])
        transform = StandardizedTargetTransform()

        transformed = transform.fit_transform(y)
        restored = transform.inverse_transform(transformed)
        prediction = GPyTorchPrediction(
            mean=np.array([0.0]),
            std=np.array([0.5]),
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
        )
        inverted = transform.inverse_prediction(prediction)

        self.assertAlmostEqual(float(np.mean(transformed)), 0.0)
        self.assertAlmostEqual(float(np.std(transformed)), 1.0)
        self.assertTrue(np.allclose(restored, y))
        self.assertTrue(np.allclose(inverted.mean, [4.0]))
        self.assertTrue(np.allclose(inverted.std, [np.std(y) * 0.5]))
        self.assertTrue(np.allclose(inverted.lower, [4.0 - np.std(y)]))
        self.assertTrue(np.allclose(inverted.upper, [4.0 + np.std(y)]))

    def test_log_transform_uses_lognormal_prediction_moments(self):
        transform = LogTargetTransform(offset=1.0)
        prediction = GPyTorchPrediction(
            mean=np.array([np.log(3.0)]),
            std=np.array([0.2]),
            lower=np.array([np.log(2.0)]),
            upper=np.array([np.log(4.0)]),
        )

        inverted = transform.inverse_prediction(prediction)
        expected_mean = np.exp(np.log(3.0) + 0.5 * 0.2**2) - 1.0
        expected_std = np.sqrt(np.expm1(0.2**2) * np.exp(2.0 * np.log(3.0) + 0.2**2))

        self.assertTrue(np.allclose(transform.inverse_transform(transform.transform([2.0])), [2.0]))
        self.assertTrue(np.allclose(inverted.mean, [expected_mean]))
        self.assertTrue(np.allclose(inverted.std, [expected_std]))
        self.assertTrue(np.allclose(inverted.lower, [1.0]))
        self.assertTrue(np.allclose(inverted.upper, [3.0]))

    def test_bounded_transform_round_trip_and_prediction_moments(self):
        transform = BoundedTargetTransform(
            lower_bound=0.0,
            upper_bound=100.0,
            n_quadrature_points=25,
        )
        y = np.array([10.0, 50.0, 90.0])

        transformed = transform.fit_transform(y)
        restored = transform.inverse_transform(transformed)
        prediction = GPyTorchPrediction(
            mean=np.array([0.0]),
            std=np.array([0.25]),
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
        )
        inverted = transform.inverse_prediction(prediction)

        self.assertTrue(np.allclose(restored, y))
        self.assertGreater(inverted.mean[0], 0.0)
        self.assertLess(inverted.mean[0], 100.0)
        self.assertGreater(inverted.std[0], 0.0)
        self.assertTrue(np.allclose(inverted.lower, transform.inverse_transform([-1.0])))
        self.assertTrue(np.allclose(inverted.upper, transform.inverse_transform([1.0])))

    def test_bounded_transform_validates_physical_interval(self):
        with self.assertRaises(ValueError):
            BoundedTargetTransform(lower_bound=1.0, upper_bound=1.0)
        with self.assertRaises(ValueError):
            BoundedTargetTransform(lower_bound=0.0, upper_bound=1.0).fit([0.0, 0.5])
        with self.assertRaises(ValueError):
            BoundedTargetTransform(lower_bound=0.0, upper_bound=1.0).fit([0.5, 1.0])

    def test_physics_residual_transform_adds_baseline_back(self):
        y = np.array([10.0, 12.0, 15.0])
        baseline = np.array([9.0, 11.5, 14.0])
        transform = PhysicsResidualTransform()

        residual = transform.fit_transform(y, baseline=baseline)
        restored = transform.inverse_transform(residual, baseline=baseline)
        prediction = GPyTorchPrediction(
            mean=np.array([0.5, 1.0]),
            std=np.array([0.2, 0.3]),
            lower=np.array([0.1, 0.4]),
            upper=np.array([0.9, 1.6]),
        )
        inverted = transform.inverse_prediction(prediction, baseline=np.array([20.0, 30.0]))

        self.assertTrue(np.allclose(residual, [1.0, 0.5, 1.0]))
        self.assertTrue(np.allclose(restored, y))
        self.assertTrue(np.allclose(inverted.mean, [20.5, 31.0]))
        self.assertTrue(np.allclose(inverted.std, prediction.std))
        self.assertTrue(np.allclose(inverted.lower, [20.1, 30.4]))
        self.assertTrue(np.allclose(inverted.upper, [20.9, 31.6]))

    def test_factory_and_validation_errors(self):
        self.assertIsInstance(make_target_transform("standard"), StandardizedTargetTransform)
        self.assertIsInstance(make_target_transform("log", offset=1.0), LogTargetTransform)
        self.assertIsInstance(make_target_transform("positive"), LogTargetTransform)
        self.assertIsInstance(
            make_target_transform("bounded", lower_bound=0.0, upper_bound=1.0),
            BoundedTargetTransform,
        )
        self.assertIsInstance(make_target_transform("physics-residual"), PhysicsResidualTransform)

        with self.assertRaises(ValueError):
            make_target_transform("unknown")
        with self.assertRaises(ValueError):
            StandardizedTargetTransform().fit([1.0, 1.0])
        with self.assertRaises(ValueError):
            LogTargetTransform().transform([0.0])
        with self.assertRaises(ValueError):
            PhysicsResidualTransform().transform([1.0, 2.0], baseline=[1.0])


if __name__ == "__main__":
    unittest.main()
