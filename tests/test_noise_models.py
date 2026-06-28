from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    FeatureNoiseModel,
    ObservationNoiseProfile,
    ReplicateNoiseModel,
    SourceNoiseModel,
    combine_noise_profiles,
    constant_noise_profile,
)


class NoiseModelTests(unittest.TestCase):
    def test_observation_noise_profile_returns_alpha_summary_and_frame(self):
        profile = ObservationNoiseProfile(
            noise_std=[0.1, 0.2, 0.3],
            labels=["a", "a", "b"],
            component_names=["source", "source", "feature"],
        )

        frame = profile.to_frame()
        summary = profile.summary()

        self.assertEqual(profile.n_observations, 3)
        self.assertTrue(np.allclose(profile.alpha, [0.01, 0.04, 0.09]))
        self.assertEqual(list(frame.columns), ["noise_std", "variance", "label", "component"])
        self.assertEqual(summary.loc[summary["label"] == "a", "count"].iloc[0], 2)

    def test_source_noise_model_assigns_known_and_default_sources(self):
        model = SourceNoiseModel(
            source_noise_std={
                "experiment": 0.05,
                "simulation": 0.20,
            },
            default_noise_std=0.50,
            unknown="default",
        )

        profile = model.profile(["experiment", "simulation", "literature"])

        self.assertTrue(np.allclose(profile.noise_std, [0.05, 0.20, 0.50]))
        self.assertTrue(np.allclose(profile.alpha, [0.0025, 0.04, 0.25]))
        self.assertTrue(
            np.array_equal(
                profile.labels,
                np.array(["source:experiment", "source:simulation", "source:literature"], dtype=object),
            )
        )

    def test_replicate_noise_model_estimates_group_noise_and_singleton_fallback(self):
        y = np.array([1.0, 1.2, 2.0, 2.4, 10.0])
        groups = np.array(["sample_a", "sample_a", "sample_b", "sample_b", "sample_c"], dtype=object)
        model = ReplicateNoiseModel(min_noise_std=0.05)

        profile = model.fit_profile(y, groups)

        expected_a = np.std([1.0, 1.2], ddof=1)
        expected_b = np.std([2.0, 2.4], ddof=1)
        pooled_fallback = np.sqrt(np.mean([expected_a**2, expected_b**2]))

        self.assertTrue(np.allclose(profile.noise_std[:2], expected_a))
        self.assertTrue(np.allclose(profile.noise_std[2:4], expected_b))
        self.assertTrue(np.allclose(profile.noise_std[4], pooled_fallback))
        self.assertAlmostEqual(model.group_noise_std_["sample_c"], pooled_fallback)

    def test_feature_noise_model_uses_user_equation(self):
        X = pd.DataFrame(
            {
                "temperature_k": [300.0, 400.0, 500.0],
                "loading": [0.1, 0.5, 1.0],
            }
        )
        model = FeatureNoiseModel(
            noise_std_function=lambda values: 0.01 + 1e-4 * (values[:, 0] - 300.0) + 0.02 * values[:, 1],
            label="temperature_loading_noise",
        )

        profile = model.profile(X)

        self.assertTrue(np.allclose(profile.noise_std, [0.012, 0.03, 0.05]))
        self.assertTrue(
            np.array_equal(
                profile.labels,
                np.array(["temperature_loading_noise"] * 3, dtype=object),
            )
        )

    def test_combines_noise_profiles_in_quadrature_max_and_sum(self):
        base = constant_noise_profile(3, 0.1, label="base")
        source = ObservationNoiseProfile([0.0, 0.2, 0.3], labels=["s1", "s2", "s3"])

        quadrature = combine_noise_profiles(base, source)
        maximum = combine_noise_profiles(base, source, mode="max")
        summed = combine_noise_profiles(base, source, mode="sum")

        self.assertTrue(np.allclose(quadrature.noise_std, np.sqrt([0.01, 0.05, 0.10])))
        self.assertTrue(np.allclose(maximum.noise_std, [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(summed.noise_std, [0.1, 0.3, 0.4]))
        self.assertTrue(np.all(quadrature.labels == "combined"))

    def test_validation_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            ObservationNoiseProfile([-0.1])
        with self.assertRaises(ValueError):
            SourceNoiseModel({"experiment": 0.1}).profile(["unknown"])
        with self.assertRaises(ValueError):
            ReplicateNoiseModel().profile(["sample"])
        with self.assertRaises(ValueError):
            FeatureNoiseModel(lambda values: np.array([0.1])).profile(np.ones((2, 1)))
        with self.assertRaises(ValueError):
            combine_noise_profiles(
                ObservationNoiseProfile([0.1]),
                ObservationNoiseProfile([0.1, 0.2]),
            )


if __name__ == "__main__":
    unittest.main()
