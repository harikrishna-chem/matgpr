from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import pandas as pd
from rdkit import Chem

from matgpr.organic_fingerprints import (
    append_smiles_features,
    canonicalize_molecule_smiles,
    canonicalize_polymer_smiles,
    featurize_smiles,
    fingerprint_smiles,
)


class OrganicFingerprintTests(unittest.TestCase):
    def test_canonicalize_molecule_smiles(self):
        self.assertEqual(canonicalize_molecule_smiles("OCC"), "CCO")

    def test_canonicalize_polymer_builds_cyclic_trimer(self):
        canonical = canonicalize_polymer_smiles("[*]CC[*]")
        molecule = Chem.MolFromSmiles(canonical)

        self.assertEqual(canonical, "C1CCCCC1")
        self.assertEqual(molecule.GetNumAtoms(), 6)
        self.assertEqual(molecule.GetRingInfo().NumRings(), 1)

    def test_canonicalize_polymer_supports_one_atom_repeat_units(self):
        canonical = canonicalize_polymer_smiles("[*]C[*]")
        molecule = Chem.MolFromSmiles(canonical)

        self.assertEqual(canonical, "C1CC1")
        self.assertEqual(molecule.GetNumAtoms(), 3)
        self.assertEqual(molecule.GetRingInfo().NumRings(), 1)

    def test_canonicalize_polymer_removes_dummy_atoms(self):
        canonical = canonicalize_polymer_smiles("[*]CC=C(C)C[*]")

        self.assertNotIn("*", canonical)
        self.assertEqual(canonicalize_molecule_smiles(canonical), canonical)

    def test_canonicalize_polymer_preserves_dummy_bond_order(self):
        canonical = canonicalize_polymer_smiles("[*]=CCCC(=[*])C")
        molecule = Chem.MolFromSmiles(canonical)
        double_bonds = sum(
            1 for bond in molecule.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE
        )

        self.assertNotIn("*", canonical)
        self.assertEqual(double_bonds, 3)

    def test_canonicalize_polymer_rejects_missing_or_extra_dummy_atoms(self):
        with self.assertRaises(ValueError):
            canonicalize_polymer_smiles("CCO")
        with self.assertRaises(ValueError):
            canonicalize_polymer_smiles("[*]CC([*])C[*]")

    def test_fingerprint_smiles_supports_morgan_and_descriptors(self):
        array, canonical, names = fingerprint_smiles(
            "CCO",
            fingerprint_type="morgan+descriptors",
            n_bits=32,
        )

        self.assertEqual(canonical, "CCO")
        self.assertEqual(len(array), 42)
        self.assertEqual(len(names), 42)
        self.assertIn("desc_MolWt", names)

    def test_featurize_smiles_reports_polymer_failures(self):
        result = featurize_smiles(
            ["[*]CC[*]", "CCO"],
            smiles_type="polymer",
            fingerprint_type="morgan",
            n_bits=16,
            errors="coerce",
        )

        self.assertEqual(result.features.shape, (2, 16))
        self.assertEqual(result.failed.shape[0], 1)
        self.assertTrue(result.canonical_smiles.iloc[0])

    def test_featurize_smiles_supports_rdkit_and_maccs(self):
        rdkit_result = featurize_smiles(["CCO"], fingerprint_type="rdkit", n_bits=64)
        maccs_result = featurize_smiles(["CCO"], fingerprint_type="maccs")

        self.assertEqual(rdkit_result.features.shape, (1, 64))
        self.assertEqual(maccs_result.features.shape, (1, 167))

    def test_append_smiles_features_preserves_index(self):
        data = pd.DataFrame({"smiles": ["CCO", "CCN"]}, index=[10, 20])

        result = append_smiles_features(
            data,
            smiles_column="smiles",
            fingerprint_type="descriptors",
        )

        self.assertEqual(result.index.tolist(), [10, 20])
        self.assertIn("molecule_descriptors_canonical_smiles", result.columns)
        self.assertIn("molecule_descriptors_desc_MolWt", result.columns)


if __name__ == "__main__":
    unittest.main()
