from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    build_cartesian_candidate_grid,
    build_composition_candidate_grid,
    exclude_existing_candidates,
    split_candidate_features,
)


class CandidateGenerationTests(unittest.TestCase):
    def test_build_cartesian_candidate_grid_preserves_parameter_order(self):
        candidates = build_cartesian_candidate_grid(
            {
                "temperature_c": [60, 80],
                "solvent": ["water", "ethanol"],
                "catalyst": "none",
            },
            fixed_values={"campaign": "screen_1"},
        )

        self.assertEqual(
            candidates.columns.tolist(),
            ["candidate_id", "temperature_c", "solvent", "catalyst", "campaign"],
        )
        self.assertEqual(candidates.shape[0], 4)
        self.assertEqual(candidates["candidate_id"].iloc[0], "candidate_000001")
        self.assertEqual(candidates["catalyst"].unique().tolist(), ["none"])
        self.assertEqual(candidates["campaign"].unique().tolist(), ["screen_1"])

    def test_build_cartesian_candidate_grid_validates_size_and_columns(self):
        with self.assertRaises(ValueError):
            build_cartesian_candidate_grid({})
        with self.assertRaises(ValueError):
            build_cartesian_candidate_grid({"x": []})
        with self.assertRaises(ValueError):
            build_cartesian_candidate_grid(
                {"x": [1, 2]},
                fixed_values={"x": 0},
            )
        with self.assertRaises(ValueError):
            build_cartesian_candidate_grid({"x": [1, 2]}, max_candidates=1)

    def test_build_composition_candidate_grid_creates_reduced_formulas(self):
        candidates = build_composition_candidate_grid(
            ["Al", "Ni"],
            step=0.5,
            candidate_id_column=None,
        )

        self.assertEqual(candidates["formula"].tolist(), ["Al", "AlNi", "Ni"])
        self.assertEqual(candidates["n_components"].tolist(), [1, 2, 1])
        np.testing.assert_allclose(candidates["frac_Al"].to_numpy(), [1.0, 0.5, 0.0])
        np.testing.assert_allclose(candidates["frac_Ni"].to_numpy(), [0.0, 0.5, 1.0])

    def test_build_composition_candidate_grid_can_filter_composition_space(self):
        candidates = build_composition_candidate_grid(
            ["Al", "Co", "Ni"],
            step=0.5,
            min_components=2,
            max_components=2,
            min_fraction=0.5,
            max_fraction=0.5,
        )

        self.assertEqual(candidates["formula"].tolist(), ["AlCo", "AlNi", "CoNi"])
        self.assertTrue(candidates["candidate_id"].str.startswith("composition_").all())
        self.assertTrue((candidates["n_components"] == 2).all())

    def test_build_composition_candidate_grid_validates_inputs(self):
        with self.assertRaises(ValueError):
            build_composition_candidate_grid(["Al", "Al"])
        with self.assertRaises(ValueError):
            build_composition_candidate_grid(["Al", "Xx"])
        with self.assertRaises(ValueError):
            build_composition_candidate_grid(["Al", "Ni"], step=0.3)
        with self.assertRaises(ValueError):
            build_composition_candidate_grid(["Al", "Ni"], min_fraction=0.8, max_fraction=0.2)
        with self.assertRaises(ValueError):
            build_composition_candidate_grid(["Al", "Ni"], min_components=3)
        with self.assertRaises(ValueError):
            build_composition_candidate_grid(
                ["Al", "Ni"],
                candidate_id_column="formula",
            )

    def test_exclude_existing_candidates_removes_or_annotates_observed_rows(self):
        candidates = pd.DataFrame(
            {
                "formula": ["Al", "AlNi", "Ni"],
                "temperature_c": [80, 80, 100],
            }
        )
        observed = pd.DataFrame(
            {
                "formula": ["AlNi"],
                "temperature_c": [80],
            }
        )

        filtered = exclude_existing_candidates(
            candidates,
            observed,
            key_columns=("formula", "temperature_c"),
        )
        annotated = exclude_existing_candidates(
            candidates,
            observed,
            key_columns=("formula", "temperature_c"),
            keep_indicator=True,
        )

        self.assertEqual(filtered["formula"].tolist(), ["Al", "Ni"])
        self.assertEqual(annotated["matgpr_is_observed"].tolist(), [False, True, False])

    def test_split_candidate_features_returns_bo_inputs(self):
        candidates = pd.DataFrame(
            {
                "formula": ["Al", "AlNi"],
                "frac_Al": [1.0, 0.5],
                "frac_Ni": [0.0, 0.5],
                "temperature_c": [80, 100],
            }
        )

        X_candidates, candidate_data = split_candidate_features(
            candidates,
            feature_columns=("frac_Al", "frac_Ni", "temperature_c"),
        )

        self.assertEqual(X_candidates.columns.tolist(), ["frac_Al", "frac_Ni", "temperature_c"])
        self.assertEqual(candidate_data.columns.tolist(), ["formula"])
        self.assertEqual(candidate_data["formula"].tolist(), ["Al", "AlNi"])

    def test_split_candidate_features_validates_numeric_features(self):
        candidates = pd.DataFrame({"formula": ["Al"], "frac_Al": [1.0]})

        with self.assertRaises(ValueError):
            split_candidate_features(candidates, feature_columns=("formula",))
        with self.assertRaises(ValueError):
            split_candidate_features(candidates, feature_columns=("missing",))


if __name__ == "__main__":
    unittest.main()
