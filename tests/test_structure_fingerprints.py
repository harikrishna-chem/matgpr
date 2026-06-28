from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from matgpr.structure_fingerprints import (
    DEFAULT_STRUCTURE_FEATURES,
    append_structure_fingerprints,
    featurize_structures,
    structure_feature_names,
    structure_fingerprint,
)


def _diamond_structure() -> Structure:
    return Structure(
        Lattice.cubic(3.57),
        ["C", "C"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _rocksalt_structure() -> Structure:
    return Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


class StructureFingerprintTests(unittest.TestCase):
    def test_structure_fingerprint_returns_lattice_and_density_features(self):
        features = structure_fingerprint(_diamond_structure())

        self.assertEqual(set(features), set(DEFAULT_STRUCTURE_FEATURES))
        self.assertAlmostEqual(features["log_lattice_length_min"], np.log(3.57))
        self.assertAlmostEqual(features["cos_lattice_angle_mid"], 0.0, places=12)
        self.assertGreater(features["log_volume_per_atom"], 0.0)
        self.assertGreater(features["density"], 0.0)

    def test_structure_fingerprint_supports_selected_features(self):
        features = structure_fingerprint(
            _rocksalt_structure(),
            features=("volume_per_atom", "num_species"),
        )

        self.assertEqual(list(features), ["volume_per_atom", "num_species"])
        self.assertAlmostEqual(features["num_species"], 2.0)
        self.assertGreater(features["volume_per_atom"], 0.0)

    def test_featurize_structures_can_report_failures(self):
        result = featurize_structures(
            [_diamond_structure(), "not a cif"],
            features=("density",),
            errors="coerce",
        )

        self.assertEqual(result.features.shape, (2, 1))
        self.assertEqual(result.failed.shape[0], 1)
        self.assertTrue(np.isnan(result.features.loc[1, "density"]))

    def test_append_structure_fingerprints_preserves_index(self):
        data = pd.DataFrame(
            {
                "structure": [_diamond_structure(), _rocksalt_structure()],
                "hardness": [80.0, 25.0],
            },
            index=[10, 20],
        )

        result = append_structure_fingerprints(data, features=("density", "log_volume_per_atom"))

        self.assertEqual(result.index.tolist(), [10, 20])
        self.assertIn("hardness", result.columns)
        self.assertIn("density", result.columns)
        self.assertGreater(result.loc[10, "density"], 0.0)

    def test_structure_cache_reports_hits(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first = featurize_structures([_diamond_structure()], features=("density",), cache_dir=cache_dir)
            second = featurize_structures([_diamond_structure()], features=("density",), cache_dir=cache_dir)

        self.assertFalse(bool(first.cache_hit.iloc[0]))
        self.assertTrue(bool(second.cache_hit.iloc[0]))

    def test_structure_feature_names_validates_and_prefixes(self):
        names = structure_feature_names(("density",), column_prefix="struct")

        self.assertEqual(names, ["struct_density"])
        with self.assertRaises(ValueError):
            structure_feature_names(("not_a_descriptor",))


if __name__ == "__main__":
    unittest.main()
