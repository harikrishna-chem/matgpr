from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np
import pandas as pd

from matgpr.inorganic_fingerprints import (
    DEFAULT_COMPOSITION_STATISTICS,
    DEFAULT_ELEMENTAL_PROPERTIES,
    append_composition_fingerprints,
    append_element_fractions,
    clean_formula,
    composition_fingerprint,
    default_element_symbols,
    element_fraction_fingerprint,
    featurize_compositions,
    featurize_element_fractions,
)


class InorganicFingerprintTests(unittest.TestCase):
    def test_composition_fingerprint_uses_fraction_weighted_statistics(self):
        features = composition_fingerprint(
            "B4C",
            properties=("atomic_number",),
            statistics=("min", "max", "range", "fwm", "ad", "std"),
        )

        expected_mean = (4 * 5 + 1 * 6) / 5
        expected_ad = (4 / 5) * abs(5 - expected_mean) + (1 / 5) * abs(6 - expected_mean)

        self.assertEqual(features["atomic_number_min"], 5.0)
        self.assertEqual(features["atomic_number_max"], 6.0)
        self.assertEqual(features["atomic_number_range"], 1.0)
        self.assertAlmostEqual(features["atomic_number_fwm"], expected_mean)
        self.assertAlmostEqual(features["atomic_number_ad"], expected_ad)
        self.assertGreater(features["atomic_number_std"], 0.0)

    def test_default_descriptor_count_matches_paper_scale(self):
        features = composition_fingerprint("Al1.5Si1.5N2.5O1.5")

        self.assertEqual(
            len(features),
            len(DEFAULT_ELEMENTAL_PROPERTIES) * len(DEFAULT_COMPOSITION_STATISTICS),
        )
        self.assertIn("X_ad", features)
        self.assertIn("d_val_fwm", features)
        self.assertIn("atomic_radius_max", features)

    def test_formula_cleaning_removes_non_breaking_spaces(self):
        self.assertEqual(clean_formula("Ag0.05Pd0.95\u00a0"), "Ag0.05Pd0.95")

    def test_featurize_compositions_can_report_failures(self):
        result = featurize_compositions(["B4C", "4-Feb"], errors="coerce")

        self.assertEqual(result.features.shape[0], 2)
        self.assertEqual(result.failed.shape[0], 1)
        self.assertTrue(np.isnan(result.features.loc[1, "atomic_number_fwm"]))

    def test_append_composition_fingerprints_preserves_index(self):
        data = pd.DataFrame(
            {
                "composition": ["B4C", "BN"],
                "load": [0.49, 0.98],
            },
            index=[10, 20],
        )

        result = append_composition_fingerprints(data)

        self.assertEqual(result.index.tolist(), [10, 20])
        self.assertIn("load", result.columns)
        self.assertIn("atomic_mass_fwm", result.columns)

    def test_element_fraction_fingerprint_uses_fixed_element_basis(self):
        features = element_fraction_fingerprint("B4C", elements=("B", "C", "N"))

        self.assertAlmostEqual(features["element_fraction_B"], 0.8)
        self.assertAlmostEqual(features["element_fraction_C"], 0.2)
        self.assertEqual(features["element_fraction_N"], 0.0)

    def test_featurize_element_fractions_and_failures(self):
        result = featurize_element_fractions(
            ["B4C", "4-Feb"],
            elements=("B", "C"),
            errors="coerce",
        )

        self.assertEqual(result.features.shape, (2, 2))
        self.assertEqual(result.failed.shape[0], 1)
        self.assertTrue(np.isnan(result.features.loc[1, "element_fraction_B"]))

    def test_append_element_fractions_preserves_index(self):
        data = pd.DataFrame(
            {
                "composition": ["B4C", "BN"],
                "load": [0.49, 0.98],
            },
            index=[10, 20],
        )

        result = append_element_fractions(data, elements=("B", "C", "N"))

        self.assertEqual(result.index.tolist(), [10, 20])
        self.assertIn("load", result.columns)
        self.assertAlmostEqual(result.loc[10, "element_fraction_B"], 0.8)
        self.assertAlmostEqual(result.loc[20, "element_fraction_N"], 0.5)

    def test_default_element_symbols_are_periodic_table_ordered(self):
        symbols = default_element_symbols()

        self.assertEqual(symbols[0], "H")
        self.assertEqual(symbols[-1], "Og")
        self.assertEqual(len(symbols), 118)


if __name__ == "__main__":
    unittest.main()
