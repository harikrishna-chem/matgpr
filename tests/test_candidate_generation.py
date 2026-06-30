from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from matgpr import (
    build_cartesian_candidate_grid,
    build_composition_candidate_grid,
    exclude_existing_candidates,
    summarize_candidate_category_coverage,
    summarize_candidate_duplicates,
    summarize_candidate_feature_coverage,
    summarize_candidate_pool,
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

    def test_summarize_candidate_pool_reports_features_categories_and_duplicates(self):
        candidates = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c", "d"],
                "formula": ["Al", "AlNi", "AlNi", "Ni"],
                "frac_Al": [1.0, 0.5, 0.5, 0.0],
                "frac_Ni": [0.0, 0.5, 0.5, np.nan],
                "solvent": ["water", "ethanol", "ethanol", None],
            }
        )

        diagnostics = summarize_candidate_pool(
            candidates,
            feature_columns=("frac_Al", "frac_Ni"),
            categorical_columns=("formula", "solvent"),
            key_columns=("formula", "solvent"),
        )

        overview = diagnostics.overview_frame().iloc[0]
        self.assertEqual(overview["matgpr_n_candidates"], 4)
        self.assertEqual(overview["matgpr_n_numeric_features"], 2)
        self.assertEqual(overview["matgpr_feature_missing_rows"], 1)
        self.assertEqual(overview["matgpr_duplicate_key_groups"], 1)
        self.assertEqual(overview["matgpr_duplicate_candidate_rows"], 2)

        numeric = diagnostics.numeric_feature_frame()
        self.assertEqual(numeric["feature"].tolist(), ["frac_Al", "frac_Ni"])
        frac_ni = numeric.loc[numeric["feature"] == "frac_Ni"].iloc[0]
        self.assertEqual(frac_ni["missing_or_invalid_count"], 1)
        self.assertAlmostEqual(frac_ni["max"], 0.5)

        categorical = diagnostics.categorical_feature_frame()
        solvent = categorical.loc[categorical["column"] == "solvent"].iloc[0]
        self.assertEqual(solvent["unique_count"], 2)
        self.assertEqual(solvent["missing_count"], 1)

        duplicates = diagnostics.duplicate_key_frame()
        self.assertEqual(duplicates.shape[0], 1)
        self.assertEqual(duplicates["formula"].iloc[0], "AlNi")
        self.assertEqual(duplicates["matgpr_duplicate_count"].iloc[0], 2)

    def test_summarize_candidate_feature_coverage_compares_with_reference_ranges(self):
        candidates = pd.DataFrame(
            {
                "temperature_c": [70.0, 80.0, 100.0],
                "frac_Al": [0.0, 0.5, 1.0],
            }
        )
        reference = pd.DataFrame(
            {
                "temperature_c": [60.0, 80.0, 120.0],
                "frac_Al": [0.25, 0.75, 1.0],
            }
        )

        coverage = summarize_candidate_feature_coverage(
            candidates,
            reference,
            feature_columns=("temperature_c", "frac_Al"),
        )

        temperature = coverage.loc[coverage["feature"] == "temperature_c"].iloc[0]
        self.assertAlmostEqual(temperature["reference_range_covered_fraction"], 0.5)
        self.assertAlmostEqual(temperature["reference_outside_candidate_fraction"], 2 / 3)
        self.assertAlmostEqual(temperature["candidate_outside_reference_fraction"], 0.0)

        frac_al = coverage.loc[coverage["feature"] == "frac_Al"].iloc[0]
        self.assertAlmostEqual(frac_al["reference_range_covered_fraction"], 1.0)
        self.assertAlmostEqual(frac_al["candidate_below_reference_fraction"], 1 / 3)

    def test_summarize_candidate_category_coverage_reports_missing_levels(self):
        candidates = pd.DataFrame(
            {
                "solvent": ["water", "ethanol", "dmf"],
                "route": ["thermal", "thermal", "photo"],
            }
        )
        reference = pd.DataFrame(
            {
                "solvent": ["water", "ethanol", "acetone"],
                "route": ["thermal", "thermal", "thermal"],
            }
        )

        coverage = summarize_candidate_category_coverage(
            candidates,
            reference,
            categorical_columns=("solvent", "route"),
        )

        solvent = coverage.loc[coverage["column"] == "solvent"].iloc[0]
        self.assertAlmostEqual(solvent["reference_levels_covered_fraction"], 2 / 3)
        self.assertEqual(solvent["candidate_new_level_count"], 1)
        self.assertEqual(solvent["reference_missing_level_count"], 1)
        self.assertEqual(solvent["candidate_new_levels"], "dmf")
        self.assertEqual(solvent["reference_missing_levels"], "acetone")

    def test_summarize_candidate_duplicates_returns_empty_table_when_unique(self):
        candidates = pd.DataFrame({"formula": ["Al", "AlNi", "Ni"]})

        duplicates = summarize_candidate_duplicates(candidates, key_columns=("formula",))

        self.assertTrue(duplicates.empty)
        self.assertEqual(
            duplicates.columns.tolist(),
            ["formula", "matgpr_duplicate_count", "matgpr_duplicate_fraction"],
        )

    def test_candidate_diagnostics_validate_inputs(self):
        candidates = pd.DataFrame({"formula": ["Al"], "frac_Al": [1.0]})

        with self.assertRaises(ValueError):
            summarize_candidate_pool(candidates, feature_columns=("formula",))
        with self.assertRaises(ValueError):
            summarize_candidate_feature_coverage(
                candidates,
                pd.DataFrame({"other": [1.0]}),
                feature_columns=("frac_Al",),
            )
        with self.assertRaises(ValueError):
            summarize_candidate_category_coverage(
                candidates,
                pd.DataFrame({"other": ["Al"]}),
                categorical_columns=("formula",),
            )


if __name__ == "__main__":
    unittest.main()
