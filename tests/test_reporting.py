from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    MultiFidelityGPRPrediction,
    decompose_multifidelity_prediction,
    summarize_multifidelity_components,
)


class MultiFidelityReportingTests(unittest.TestCase):
    def test_decompose_multifidelity_prediction_returns_component_table(self):
        low_fidelity = np.array([6.0, 8.0])
        low_std = np.array([0.2, 0.3])
        correction = np.array([0.6, -0.4])
        correction_std = np.array([0.4, 0.5])
        rho = 1.5
        intercept = 0.4
        mean = rho * low_fidelity + intercept + correction
        std = np.sqrt((rho * low_std) ** 2 + correction_std**2)
        prediction = MultiFidelityGPRPrediction(
            mean=mean,
            std=std,
            low_fidelity_mean=low_fidelity,
            low_fidelity_std=low_std,
            correction_mean=correction,
            correction_std=correction_std,
            rho=rho,
            intercept=intercept,
        )

        frame = decompose_multifidelity_prediction(
            prediction,
            y_true=np.array([9.5, 12.5]),
            sample_labels=["alloy_a", "alloy_b"],
            model_name="delta_mf",
            split="test",
        )

        self.assertEqual(frame["sample_label"].tolist(), ["alloy_a", "alloy_b"])
        self.assertEqual(set(frame["model"]), {"delta_mf"})
        self.assertEqual(set(frame["split"]), {"test"})
        np.testing.assert_allclose(frame["scaled_low_fidelity_pred"], rho * low_fidelity)
        np.testing.assert_allclose(frame["reconstructed_y_pred"], mean)
        np.testing.assert_allclose(frame["component_residual"], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(frame["signed_error"], np.array([0.5, -0.5]))
        np.testing.assert_allclose(frame["low_fidelity_variance_fraction"].iloc[0], 0.36)
        self.assertIn("correction_variance_fraction", frame.columns)

    def test_summarize_multifidelity_components_accepts_prediction_tables(self):
        frame = pd.DataFrame(
            {
                "model": ["delta_mf", "delta_mf", "delta_mf", "delta_mf"],
                "split": ["train", "train", "test", "test"],
                "y_true": [10.0, 11.0, 12.0, 13.0],
                "y_pred": [10.1, 10.8, 11.7, 13.4],
                "low_fidelity_pred": [6.0, 7.0, 8.0, 9.0],
                "correction_pred": [0.7, -0.1, -0.3, -0.2],
                "rho": [1.5, 1.5, 1.5, 1.5],
                "intercept": [0.4, 0.4, 0.4, 0.4],
                "y_std": [0.5, 0.6, 0.7, 0.8],
                "low_fidelity_std": [0.2, 0.2, 0.3, 0.3],
                "correction_std": [0.4, 0.5, 0.5, 0.6],
            }
        )

        summary = summarize_multifidelity_components(frame, group_by=("model", "split"))
        test_row = summary[summary["split"] == "test"].iloc[0]

        self.assertEqual(summary.shape[0], 2)
        self.assertEqual(test_row["n_samples"], 2)
        self.assertIn("scaled_low_fidelity_pred_mean", summary.columns)
        self.assertIn("mean_abs_correction_pred", summary.columns)
        self.assertIn("max_abs_component_residual", summary.columns)
        self.assertIn("low_fidelity_variance_fraction_mean", summary.columns)
        self.assertAlmostEqual(test_row["RMSE"], np.sqrt((0.3**2 + (-0.4) ** 2) / 2.0))
        self.assertAlmostEqual(test_row["MAE"], 0.35)

    def test_summarize_multifidelity_components_accepts_prediction_object(self):
        prediction = MultiFidelityGPRPrediction(
            mean=np.array([2.0, 3.0]),
            low_fidelity_mean=np.array([1.0, 1.5]),
            correction_mean=np.array([0.8, 0.9]),
            rho=1.2,
            intercept=0.0,
        )

        summary = summarize_multifidelity_components(prediction)

        self.assertEqual(summary.shape[0], 1)
        self.assertEqual(summary.loc[0, "n_samples"], 2)
        self.assertAlmostEqual(summary.loc[0, "low_fidelity_pred_mean"], 1.25)
        self.assertAlmostEqual(summary.loc[0, "correction_pred_mean"], 0.85)

    def test_reporting_errors_are_explicit(self):
        prediction = MultiFidelityGPRPrediction(mean=np.array([1.0, 2.0]))

        with self.assertRaisesRegex(ValueError, "sample_labels"):
            decompose_multifidelity_prediction(prediction, sample_labels=["only_one"])
        with self.assertRaisesRegex(KeyError, "Missing group_by"):
            summarize_multifidelity_components(pd.DataFrame({"y_pred": [1.0]}), group_by="split")


if __name__ == "__main__":
    unittest.main()
