"""Scientific invariants and failure contracts for materials featurization."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from matgpr import (
    CompositionFeaturizer,
    canonicalize_polymer_smiles,
    composition_fingerprint,
    default_element_symbols,
    element_fraction_fingerprint,
    featurize_compositions,
    featurize_smiles,
    featurize_structures,
    fingerprint_smiles,
    structure_fingerprint,
)

pytestmark = pytest.mark.scientific


def _diamond_structure() -> Structure:
    return Structure(
        Lattice.cubic(3.57),
        ["C", "C"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def test_equivalent_molecule_smiles_have_identical_canonical_fingerprints() -> None:
    first, first_canonical, first_names = fingerprint_smiles("OCC", n_bits=64)
    second, second_canonical, second_names = fingerprint_smiles("C(C)O", n_bits=64)

    assert first_canonical == second_canonical == "CCO"
    assert first_names == second_names
    np.testing.assert_array_equal(first, second)
    assert set(np.unique(first)).issubset({0.0, 1.0})


def test_reversed_polymer_endpoints_have_identical_cyclic_trimer_fingerprints() -> None:
    first, first_canonical, first_names = fingerprint_smiles(
        "[*]CCO[*]",
        smiles_type="polymer",
        n_bits=64,
    )
    second, second_canonical, second_names = fingerprint_smiles(
        "[*]OCC[*]",
        smiles_type="polymer",
        n_bits=64,
    )

    assert first_canonical == second_canonical == "C1COCCOCCO1"
    assert "*" not in first_canonical
    assert first_names == second_names
    np.testing.assert_array_equal(first, second)


def test_polymer_repeat_unit_rejects_disconnected_fragments() -> None:
    with pytest.raises(ValueError, match="single connected fragment"):
        canonicalize_polymer_smiles("[*]CC[*].C")


@pytest.mark.parametrize("repeat_units", [True, 1.5])
def test_polymer_repeat_units_require_an_integer(repeat_units: object) -> None:
    with pytest.raises(TypeError, match="repeat_units must be an integer"):
        canonicalize_polymer_smiles(
            "[*]CC[*]",
            repeat_units=repeat_units,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("repeat_units", [0, -1])
def test_polymer_repeat_units_must_be_positive(repeat_units: int) -> None:
    with pytest.raises(ValueError, match="repeat_units must be at least 1"):
        canonicalize_polymer_smiles("[*]CC[*]", repeat_units=repeat_units)


@pytest.mark.parametrize("n_bits", [True, 1.5])
def test_fingerprint_bit_count_requires_an_integer(n_bits: object) -> None:
    with pytest.raises(TypeError, match="n_bits must be an integer"):
        fingerprint_smiles("CCO", n_bits=n_bits)  # type: ignore[arg-type]


def test_fingerprint_bit_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_bits must be at least 1"):
        fingerprint_smiles("CCO", n_bits=0)


@pytest.mark.parametrize("radius", [True, 1.5])
def test_morgan_radius_requires_an_integer(radius: object) -> None:
    with pytest.raises(TypeError, match="radius must be an integer"):
        fingerprint_smiles("CCO", radius=radius)  # type: ignore[arg-type]


def test_morgan_radius_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="radius must be non-negative"):
        fingerprint_smiles("CCO", radius=-1)


@pytest.mark.parametrize(
    ("fingerprint_type", "descriptors"),
    [
        ("morgan+morgan", ("MolWt",)),
        ("descriptors", ("MolWt", "MolWt")),
        ("descriptors", ()),
    ],
)
def test_smiles_feature_definitions_must_be_supported_and_unique(
    fingerprint_type: str,
    descriptors: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        fingerprint_smiles(
            "CCO",
            fingerprint_type=fingerprint_type,
            descriptors=descriptors,
        )


def test_unique_fingerprint_components_preserve_requested_order() -> None:
    values, canonical, names = fingerprint_smiles(
        "CCO",
        fingerprint_type="descriptors+morgan",
        n_bits=16,
        descriptors=("MolWt",),
    )

    assert canonical == "CCO"
    assert names == ["desc_MolWt", *[f"morgan_{index}" for index in range(16)]]
    assert values.shape == (17,)
    assert values[0] > 0.0
    assert set(np.unique(values[1:])).issubset({0.0, 1.0})


def test_composition_descriptors_are_invariant_to_formula_order_and_scale() -> None:
    settings = {
        "properties": ("atomic_number", "atomic_mass"),
        "statistics": ("fwm", "std"),
    }
    first = composition_fingerprint("NaCl", **settings)
    reordered = composition_fingerprint("ClNa", **settings)
    scaled = composition_fingerprint("Na2Cl2", **settings)

    assert first.keys() == reordered.keys() == scaled.keys()
    np.testing.assert_allclose(list(first.values()), list(reordered.values()), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(list(first.values()), list(scaled.values()), rtol=0.0, atol=0.0)


def test_element_fractions_use_declared_basis_and_sum_to_one() -> None:
    fractions = element_fraction_fingerprint("LiFePO4", elements=("Li", "Fe", "P", "O", "Na"))

    assert list(fractions) == [
        "element_fraction_Li",
        "element_fraction_Fe",
        "element_fraction_P",
        "element_fraction_O",
        "element_fraction_Na",
    ]
    assert fractions["element_fraction_Na"] == 0.0
    assert sum(fractions.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("properties", "statistics"),
    [
        ((), ("fwm",)),
        (("atomic_number", "atomic_number"), ("fwm",)),
        (("atomic_number",), ()),
        (("atomic_number",), ("fwm", "fwm")),
    ],
)
def test_composition_feature_definitions_must_be_nonempty_and_unique(
    properties: tuple[str, ...],
    statistics: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        composition_fingerprint(
            "NaCl",
            properties=properties,
            statistics=statistics,
        )


def test_default_element_symbols_accepts_numpy_integer_but_not_boolean() -> None:
    assert default_element_symbols(np.int64(2)) == ("H", "He")
    with pytest.raises(TypeError, match="max_atomic_number must be an integer"):
        default_element_symbols(True)


def test_structure_descriptors_are_invariant_to_site_order() -> None:
    structure = _diamond_structure()
    reordered = Structure(
        structure.lattice,
        list(reversed(structure.species)),
        list(reversed(structure.frac_coords)),
    )

    first = structure_fingerprint(structure)
    second = structure_fingerprint(reordered)

    assert first.keys() == second.keys()
    np.testing.assert_allclose(list(first.values()), list(second.values()), rtol=0.0, atol=1e-14)


def test_intensive_structure_descriptors_are_supercell_invariant() -> None:
    structure = _diamond_structure()
    supercell = structure.copy()
    supercell.make_supercell([2, 2, 2])
    features = ("volume_per_atom", "log_volume_per_atom", "density", "num_species")

    first = structure_fingerprint(structure, features=features)
    second = structure_fingerprint(supercell, features=features)

    np.testing.assert_allclose(list(first.values()), list(second.values()), rtol=1e-12, atol=1e-12)


def test_structure_object_cif_text_and_file_have_identical_features(tmp_path) -> None:
    structure = _diamond_structure()
    cif_text = structure.to(fmt="cif")
    cif_path = tmp_path / "diamond.cif"
    cif_path.write_text(cif_text, encoding="utf-8")

    object_features = structure_fingerprint(structure)
    text_features = structure_fingerprint(cif_text)
    path_features = structure_fingerprint(cif_path)

    np.testing.assert_allclose(
        list(object_features.values()),
        list(text_features.values()),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        list(object_features.values()),
        list(path_features.values()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_structure_feature_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        structure_fingerprint(_diamond_structure(), features=("density", "density"))


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (featurize_smiles, "CCO"),
        (featurize_compositions, "NaCl"),
        (featurize_structures, "not a batch of structures"),
    ],
)
def test_batch_featurizers_reject_scalar_strings(function, value: str) -> None:
    with pytest.raises(TypeError, match="non-string iterable"):
        function(value)


def test_composition_featurizer_validates_descriptor_configuration_during_fit() -> None:
    with pytest.raises(ValueError, match="properties must contain at least one"):
        CompositionFeaturizer(properties=()).fit(["NaCl"])


def test_featurizer_accepts_numpy_integer_column_indices() -> None:
    data = np.asarray([["sample-a", "NaCl"], ["sample-b", "LiF"]], dtype=object)
    featurizer = CompositionFeaturizer(
        formula_column=np.int64(1),
        properties=("atomic_number",),
        statistics=("fwm",),
    )

    features = featurizer.fit_transform(data)

    assert features.shape == (2, 1)
    assert np.isfinite(features.to_numpy()).all()


def test_featurizer_rejects_boolean_column_indices_during_fit() -> None:
    data = np.asarray([["sample-a", "NaCl"], ["sample-b", "LiF"]], dtype=object)
    featurizer = CompositionFeaturizer(
        formula_column=True,
        properties=("atomic_number",),
        statistics=("fwm",),
    )

    with pytest.raises(TypeError, match="column index must be an integer"):
        featurizer.fit(data)


def test_multicolumn_featurizer_input_requires_an_explicit_column() -> None:
    data = np.asarray([["sample-a", "NaCl"], ["sample-b", "LiF"]], dtype=object)

    with pytest.raises(ValueError, match="provide an integer column"):
        CompositionFeaturizer().fit(data)
