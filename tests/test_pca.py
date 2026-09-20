from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import fit_pca, summarize_pca, transform_pca


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "band_gap": [1.0, 2.0, 3.0, 4.0],
            "density": [4.0, 3.0, 2.0, 1.0],
            "volume": [1.0, 3.0, 2.0, 4.0],
        }
    )


class FitPcaTests(unittest.TestCase):
    def test_fit_returns_scores_and_records_training_feature_names(self):
        df = sample_frame()

        scores, pca, scaler = fit_pca(df, n_components=2)

        self.assertEqual(list(scores.columns), ["PC1", "PC2"])
        self.assertEqual(len(scores), len(df))
        self.assertIsNone(scaler)
        self.assertEqual(list(pca.matgpr_feature_names_in_), list(df.columns))

    def test_summarize_reports_cumulative_variance(self):
        _, pca, _ = fit_pca(sample_frame(), n_components=2)

        summary = summarize_pca(pca)

        self.assertEqual(summary["component"].tolist(), ["PC1", "PC2"])
        self.assertLessEqual(summary["cumulative_variance_ratio"].iloc[-1], 1.0 + 1e-9)

    def test_fit_requires_a_numeric_column(self):
        with self.assertRaises(ValueError):
            fit_pca(pd.DataFrame({"formula": ["NaCl", "KCl"]}))


class TransformPcaFeatureAlignmentTests(unittest.TestCase):
    """New data must be aligned with the columns seen during ``fit_pca``."""

    def test_transform_reproduces_training_scores(self):
        df = sample_frame()
        scores, pca, _ = fit_pca(df, n_components=2)

        np.testing.assert_allclose(transform_pca(df, pca).to_numpy(), scores.to_numpy())

    def test_transform_preserves_the_dataframe_index(self):
        df = sample_frame().set_index(pd.Index(["a", "b", "c", "d"], name="material"))
        _, pca, _ = fit_pca(df, n_components=2)

        self.assertEqual(list(transform_pca(df, pca).index), ["a", "b", "c", "d"])

    def test_transform_rejects_reordered_columns(self):
        df = sample_frame()
        _, pca, _ = fit_pca(df, n_components=2)

        with self.assertRaisesRegex(ValueError, "do not match"):
            transform_pca(df[["volume", "density", "band_gap"]], pca)

    def test_transform_rejects_missing_and_renamed_columns(self):
        df = sample_frame()
        _, pca, _ = fit_pca(df, n_components=2)

        with self.assertRaisesRegex(ValueError, "do not match"):
            transform_pca(df[["band_gap", "density"]], pca)
        with self.assertRaisesRegex(ValueError, "do not match"):
            transform_pca(df.rename(columns={"volume": "cell_volume"}), pca)

    def test_transform_rejects_arrays_with_the_wrong_feature_count(self):
        df = sample_frame()
        _, pca, _ = fit_pca(df, n_components=2)

        with self.assertRaisesRegex(ValueError, "expects 3"):
            transform_pca(df.to_numpy()[:, :2], pca)

    def test_transform_accepts_arrays_with_the_training_feature_count(self):
        df = sample_frame()
        scores, pca, _ = fit_pca(df, n_components=2)

        np.testing.assert_allclose(transform_pca(df.to_numpy(), pca).to_numpy(), scores.to_numpy())

    def test_transform_applies_the_fitted_scaler(self):
        df = sample_frame()
        scores, pca, scaler = fit_pca(df, n_components=2, scale=True)

        np.testing.assert_allclose(
            transform_pca(df, pca, scaler=scaler).to_numpy(), scores.to_numpy()
        )


if __name__ == "__main__":
    unittest.main()
