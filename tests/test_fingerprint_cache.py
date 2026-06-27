from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np

from matgpr import fingerprint_cache_key
from matgpr.inorganic_fingerprints import featurize_compositions
from matgpr.organic_fingerprints import featurize_smiles


class FingerprintCacheTests(unittest.TestCase):
    def test_cache_key_is_deterministic_for_mapping_order(self):
        first = fingerprint_cache_key(
            namespace="composition",
            value="B4C",
            parameters={"properties": ["atomic_number"], "statistics": ["fwm"]},
        )
        second = fingerprint_cache_key(
            namespace="composition",
            value="B4C",
            parameters={"statistics": ["fwm"], "properties": ["atomic_number"]},
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_composition_featurization_uses_persistent_cache(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first = featurize_compositions(
                ["B4C"],
                properties=("atomic_number",),
                statistics=("fwm",),
                cache_dir=cache_dir,
            )
            second = featurize_compositions(
                ["B4C"],
                properties=("atomic_number",),
                statistics=("fwm",),
                cache_dir=cache_dir,
            )

        self.assertFalse(bool(first.cache_hit.iloc[0]))
        self.assertTrue(bool(second.cache_hit.iloc[0]))
        self.assertEqual(first.cache_keys.iloc[0], second.cache_keys.iloc[0])
        self.assertAlmostEqual(
            first.features.loc[0, "atomic_number_fwm"],
            second.features.loc[0, "atomic_number_fwm"],
        )

    def test_smiles_featurization_uses_persistent_cache(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first = featurize_smiles(
                ["OCC"],
                fingerprint_type="descriptors",
                descriptors=("MolWt", "TPSA"),
                cache_dir=cache_dir,
            )
            second = featurize_smiles(
                ["OCC"],
                fingerprint_type="descriptors",
                descriptors=("MolWt", "TPSA"),
                cache_dir=cache_dir,
            )

        self.assertFalse(bool(first.cache_hit.iloc[0]))
        self.assertTrue(bool(second.cache_hit.iloc[0]))
        self.assertEqual(first.cache_keys.iloc[0], second.cache_keys.iloc[0])
        self.assertEqual(first.canonical_smiles.iloc[0], "CCO")
        self.assertEqual(second.canonical_smiles.iloc[0], "CCO")
        self.assertTrue(np.allclose(first.features.to_numpy(), second.features.to_numpy()))

    def test_failed_rows_include_cache_keys(self):
        result = featurize_smiles(
            [None],
            fingerprint_type="morgan",
            n_bits=8,
            errors="coerce",
        )

        self.assertEqual(result.failed.shape[0], 1)
        self.assertIn("cache_key", result.failed.columns)
        self.assertEqual(result.failed.loc[0, "cache_key"], result.cache_keys.iloc[0])
        self.assertIn("SMILES is empty", result.failed.loc[0, "error"])


if __name__ == "__main__":
    unittest.main()
