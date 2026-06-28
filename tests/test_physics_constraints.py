from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    KnownLimitConstraint,
    MonotonicTrendConstraint,
    VirtualObservationSet,
    append_virtual_observations,
    combine_virtual_observations,
)


class PhysicsConstraintTests(unittest.TestCase):
    def test_known_limit_constraint_generates_dataframe_named_feature_anchors(self):
        X = pd.DataFrame(
            {
                "time_s": [1.0, 2.0, 3.0],
                "descriptor": [0.2, 0.4, 0.6],
            }
        )
        constraint = KnownLimitConstraint(
            feature="time_s",
            limit_value=0.0,
            target_value=0.0,
            noise_std=0.02,
            label="zero_time",
        )

        virtual = constraint.generate(X)

        self.assertEqual(virtual.n_observations, 3)
        self.assertEqual(virtual.n_features, 2)
        self.assertTrue(np.allclose(virtual.X[:, 0], 0.0))
        self.assertTrue(np.allclose(virtual.X[:, 1], X["descriptor"].to_numpy()))
        self.assertTrue(np.allclose(virtual.y, 0.0))
        self.assertTrue(np.allclose(virtual.alpha, 0.02**2))
        self.assertTrue(np.array_equal(virtual.labels, np.array(["zero_time"] * 3, dtype=object)))

    def test_known_limit_constraint_accepts_callable_target(self):
        X = np.array(
            [
                [300.0, 0.5],
                [350.0, 1.5],
            ]
        )
        constraint = KnownLimitConstraint(
            feature=0,
            limit_value=273.15,
            target_value=lambda x_virtual: 2.0 * x_virtual[:, 1],
        )

        virtual = constraint.generate(X)

        self.assertTrue(np.allclose(virtual.X[:, 0], 273.15))
        self.assertTrue(np.allclose(virtual.y, [1.0, 3.0]))

    def test_monotonic_trend_constraint_generates_increasing_local_anchors(self):
        X = np.array(
            [
                [0.0, 10.0],
                [1.0, 20.0],
                [2.0, 30.0],
            ]
        )
        y = np.array([5.0, 6.0, 7.0])
        constraint = MonotonicTrendConstraint(
            feature=0,
            direction="increasing",
            step=0.5,
            minimum_slope=2.0,
            noise_std=0.3,
        )

        virtual = constraint.generate(X, y)

        self.assertTrue(np.allclose(virtual.X[:, 0], X[:, 0] + 0.5))
        self.assertTrue(np.allclose(virtual.X[:, 1], X[:, 1]))
        self.assertTrue(np.allclose(virtual.y, y + 1.0))
        self.assertTrue(np.allclose(virtual.alpha, 0.3**2))
        self.assertTrue(np.array_equal(virtual.labels, np.array(["monotonic_increasing"] * 3, dtype=object)))

    def test_monotonic_trend_constraint_supports_decreasing_trends_and_bounds(self):
        X = pd.DataFrame(
            {
                "temperature_k": [90.0, 95.0, 99.0],
                "descriptor": [1.0, 2.0, 3.0],
            }
        )
        y = np.array([10.0, 9.0, 8.0])
        constraint = MonotonicTrendConstraint(
            feature="temperature_k",
            direction="decreasing",
            step=5.0,
            minimum_slope=0.2,
            feature_max=100.0,
            noise_std=[0.1, 0.2, 0.3],
            label="arrhenius_decrease",
        )

        virtual = constraint.generate(X, y)

        self.assertEqual(virtual.n_observations, 2)
        self.assertTrue(np.allclose(virtual.X[:, 0], [95.0, 100.0]))
        self.assertTrue(np.allclose(virtual.y, [9.0, 8.0]))
        self.assertTrue(np.allclose(virtual.alpha, [0.01, 0.04]))
        self.assertTrue(np.array_equal(virtual.labels, np.array(["arrhenius_decrease"] * 2, dtype=object)))

    def test_append_virtual_observations_preserves_dataframe_and_returns_alpha(self):
        X = pd.DataFrame(
            {
                "time_s": [1.0, 2.0],
                "descriptor": [0.2, 0.4],
            }
        )
        y = np.array([1.0, 2.0])
        known_limit = KnownLimitConstraint(
            feature="time_s",
            limit_value=0.0,
            target_value=0.0,
            noise_std=0.05,
            label="zero_time",
        ).generate(X)
        monotonic = MonotonicTrendConstraint(
            feature="time_s",
            direction="increasing",
            step=1.0,
            minimum_slope=0.5,
            noise_std=0.2,
        ).generate(X, y)
        combined = combine_virtual_observations(known_limit, monotonic)

        augmented = append_virtual_observations(
            X,
            y,
            combined,
            base_alpha=1e-6,
        )

        self.assertIsInstance(augmented.X, pd.DataFrame)
        self.assertEqual(list(augmented.X.columns), ["time_s", "descriptor"])
        self.assertEqual(augmented.X.shape, (6, 2))
        self.assertTrue(np.allclose(augmented.y, [1.0, 2.0, 0.0, 0.0, 1.5, 2.5]))
        self.assertTrue(np.allclose(augmented.alpha[:2], 1e-6))
        self.assertTrue(np.allclose(augmented.alpha[2:4], 0.05**2))
        self.assertTrue(np.allclose(augmented.alpha[4:], 0.2**2))
        self.assertTrue(np.array_equal(augmented.labels[:2], np.array(["observed", "observed"], dtype=object)))

    def test_validation_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            KnownLimitConstraint(feature="time_s", limit_value=0.0, target_value=0.0).generate(
                np.array([[1.0], [2.0]])
            )
        with self.assertRaises(ValueError):
            MonotonicTrendConstraint(feature=0, direction="sideways").generate(
                np.array([[1.0], [2.0]]),
                np.array([1.0, 2.0]),
            )
        with self.assertRaises(ValueError):
            VirtualObservationSet(X=np.array([[1.0]]), y=np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            append_virtual_observations(
                np.array([[1.0, 2.0]]),
                np.array([1.0]),
                VirtualObservationSet(X=np.array([[1.0]]), y=np.array([1.0])),
            )


if __name__ == "__main__":
    unittest.main()
