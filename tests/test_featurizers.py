from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matgpr import CompositionFeaturizer, PolymerSmilesFeaturizer, SmilesFeaturizer, StructureFeaturizer


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


class CompositionFeaturizerTests(unittest.TestCase):
    def test_transform_dataframe_formula_column(self):
        data = pd.DataFrame({"formula": ["B4C", "BN"], "label": [1, 2]}, index=[10, 20])
        featurizer = CompositionFeaturizer(
            formula_column="formula",
            properties=("atomic_number",),
            statistics=("min", "max", "fwm"),
            column_prefix="comp",
        )

        features = featurizer.fit_transform(data)

        self.assertEqual(features.index.tolist(), [10, 20])
        self.assertEqual(
            features.columns.tolist(),
            [
                "comp_atomic_number_min",
                "comp_atomic_number_max",
                "comp_atomic_number_fwm",
            ],
        )
        self.assertEqual(featurizer.get_feature_names_out().tolist(), features.columns.tolist())
        self.assertEqual(featurizer.failed_.shape[0], 0)

    def test_transform_array_and_report_failures(self):
        featurizer = CompositionFeaturizer(
            properties=("atomic_number",),
            statistics=("fwm",),
            errors="coerce",
            return_dataframe=False,
        )

        features = featurizer.fit_transform(["B4C", "4-Feb"])

        self.assertEqual(features.shape, (2, 1))
        self.assertTrue(np.isnan(features[1, 0]))
        self.assertEqual(featurizer.failed_.shape[0], 1)

    def test_transformer_exposes_cache_metadata(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            featurizer = CompositionFeaturizer(
                properties=("atomic_number",),
                statistics=("fwm",),
                cache_dir=cache_dir,
            )
            featurizer.fit_transform(["B4C"])
            featurizer.transform(["B4C"])

        self.assertEqual(featurizer.cache_keys_.shape[0], 1)
        self.assertTrue(bool(featurizer.cache_hit_.iloc[0]))

    def test_composition_featurizer_is_cloneable(self):
        featurizer = CompositionFeaturizer(
            formula_column="formula",
            properties=("atomic_number",),
            statistics=("fwm",),
        )
        cloned = clone(featurizer)

        self.assertEqual(cloned.formula_column, "formula")
        self.assertEqual(cloned.properties, ("atomic_number",))
        self.assertEqual(cloned.statistics, ("fwm",))


class StructureFeaturizerTests(unittest.TestCase):
    def test_transform_dataframe_structure_column(self):
        data = pd.DataFrame(
            {
                "structure": [_diamond_structure(), _rocksalt_structure()],
                "label": [1, 2],
            },
            index=[10, 20],
        )
        featurizer = StructureFeaturizer(
            structure_column="structure",
            features=("density", "log_volume_per_atom"),
            column_prefix="struct",
        )

        features = featurizer.fit_transform(data)

        self.assertEqual(features.index.tolist(), [10, 20])
        self.assertEqual(features.columns.tolist(), ["struct_density", "struct_log_volume_per_atom"])
        self.assertEqual(featurizer.get_feature_names_out().tolist(), features.columns.tolist())
        self.assertEqual(featurizer.failed_.shape[0], 0)

    def test_structure_featurizer_works_in_simple_pipeline(self):
        pipeline = Pipeline(
            steps=[
                (
                    "structure",
                    StructureFeaturizer(
                        features=("density", "log_volume_per_atom"),
                        return_dataframe=False,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )

        transformed = pipeline.fit_transform([_diamond_structure(), _rocksalt_structure()])

        self.assertEqual(transformed.shape, (2, 2))
        self.assertTrue(np.isfinite(transformed).all())

    def test_structure_featurizer_is_cloneable(self):
        featurizer = StructureFeaturizer(
            structure_column="structure",
            features=("density",),
        )
        cloned = clone(featurizer)

        self.assertEqual(cloned.structure_column, "structure")
        self.assertEqual(cloned.features, ("density",))


class SmilesFeaturizerTests(unittest.TestCase):
    def test_molecule_smiles_featurizer_returns_descriptors(self):
        data = pd.DataFrame({"smiles": ["CCO", "CCN"]}, index=[3, 4])
        featurizer = SmilesFeaturizer(
            smiles_column="smiles",
            fingerprint_type="descriptors",
            descriptors=("MolWt", "TPSA"),
            column_prefix="mol",
        )

        features = featurizer.fit_transform(data)

        self.assertEqual(features.index.tolist(), [3, 4])
        self.assertEqual(features.columns.tolist(), ["mol_desc_MolWt", "mol_desc_TPSA"])
        self.assertEqual(featurizer.canonical_smiles_.tolist(), ["CCO", "CCN"])
        self.assertEqual(featurizer.failed_.shape[0], 0)

    def test_molecule_smiles_featurizer_can_include_canonical_smiles(self):
        featurizer = SmilesFeaturizer(
            fingerprint_type="morgan",
            n_bits=8,
            include_canonical_smiles=True,
        )

        features = featurizer.fit_transform(pd.Series(["OCC"], index=[9]))

        self.assertEqual(features.index.tolist(), [9])
        self.assertEqual(features.iloc[0, 0], "CCO")
        self.assertEqual(features.shape, (1, 9))
        self.assertIn("canonical_smiles", featurizer.get_feature_names_out()[0])

    def test_molecule_smiles_featurizer_works_in_simple_pipeline(self):
        pipeline = Pipeline(
            steps=[
                (
                    "smiles",
                    SmilesFeaturizer(
                        fingerprint_type="descriptors",
                        descriptors=("MolWt", "TPSA"),
                        return_dataframe=False,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )

        transformed = pipeline.fit_transform(["CCO", "CCN"])

        self.assertEqual(transformed.shape, (2, 2))
        self.assertTrue(np.isfinite(transformed).all())

    def test_polymer_smiles_featurizer_builds_cyclic_trimer(self):
        data = pd.DataFrame({"polymer": ["[*]CC[*]", "CCO"]})
        featurizer = PolymerSmilesFeaturizer(
            smiles_column="polymer",
            fingerprint_type="morgan",
            n_bits=16,
            errors="coerce",
        )

        features = featurizer.fit_transform(data)

        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(featurizer.canonical_smiles_.iloc[0], "C1CCCCC1")
        self.assertTrue(pd.isna(featurizer.canonical_smiles_.iloc[1]))
        self.assertEqual(featurizer.failed_.shape[0], 1)

    def test_polymer_featurizer_is_cloneable(self):
        featurizer = PolymerSmilesFeaturizer(
            smiles_column="polymer",
            fingerprint_type="morgan+descriptors",
            n_bits=32,
        )
        cloned = clone(featurizer)

        self.assertEqual(cloned.smiles_column, "polymer")
        self.assertEqual(cloned.fingerprint_type, "morgan+descriptors")
        self.assertEqual(cloned.n_bits, 32)


if __name__ == "__main__":
    unittest.main()
