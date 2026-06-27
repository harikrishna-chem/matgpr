from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_RDKIT_DESCRIPTORS: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "MolMR",
)


@dataclass(frozen=True)
class SmilesFingerprintResult:
    """Result returned by ``featurize_smiles``.

    Attributes
    ----------
    features
        Numeric fingerprint or descriptor features.
    canonical_smiles
        Canonical SMILES used for fingerprinting. For polymers, this is the
        canonicalized cyclic trimer surrogate built from the two ``[*]`` ends.
    failed
        Rows that could not be parsed or featurized. Empty when
        ``errors="raise"`` and no exception was raised.
    """

    features: pd.DataFrame
    canonical_smiles: pd.Series
    failed: pd.DataFrame


def canonicalize_molecule_smiles(smiles: object) -> str:
    """Canonicalize a molecule SMILES string with RDKit."""
    molecule = _mol_from_smiles(smiles)
    return _chem().MolToSmiles(molecule, canonical=True)


def canonicalize_polymer_smiles(
    smiles: object,
    *,
    repeat_units: int = 3,
    join_ends: bool = True,
) -> str:
    """Canonicalize a polymer repeat-unit SMILES string.

    By default, polymer repeat units must contain exactly two dummy atoms
    represented as ``[*]``. The repeat unit is expanded to a trimer, neighboring
    atoms at adjacent ``[*]`` ends are connected using the dummy-end bond order,
    the trimer is closed into a loop, and the dummy atoms are removed before
    RDKit canonicalization.
    """
    molecule = _mol_from_smiles(smiles)
    endpoint_info = _polymer_endpoint_info(molecule)

    if join_ends:
        molecule = _make_cyclic_polymer_oligomer(
            molecule,
            endpoint_info=endpoint_info,
            repeat_units=repeat_units,
        )
    return _chem().MolToSmiles(molecule, canonical=True)


def fingerprint_smiles(
    smiles: object,
    *,
    smiles_type: str = "molecule",
    fingerprint_type: str = "morgan",
    n_bits: int = 1024,
    radius: int = 2,
    descriptors: Sequence[str] = DEFAULT_RDKIT_DESCRIPTORS,
) -> tuple[np.ndarray, str, list[str]]:
    """Fingerprint one molecule or polymer SMILES string.

    Parameters
    ----------
    smiles
        Molecule SMILES, or polymer repeat-unit SMILES containing exactly two
        ``[*]`` dummy atoms when ``smiles_type="polymer"``.
    smiles_type
        ``"molecule"`` or ``"polymer"``.
    fingerprint_type
        ``"morgan"``, ``"rdkit"``, ``"maccs"``, ``"descriptors"``, or one of
        ``"morgan+descriptors"``, ``"rdkit+descriptors"``, and
        ``"maccs+descriptors"``.
    n_bits
        Fingerprint length for Morgan and RDKit fingerprints.
    radius
        Morgan fingerprint radius.
    descriptors
        RDKit descriptor names to include for descriptor-based features.
    """
    canonical_smiles = _canonicalize_by_type(smiles, smiles_type)
    molecule = _mol_from_smiles(canonical_smiles)
    fingerprint_type = fingerprint_type.lower()

    if "+" in fingerprint_type:
        parts = [part.strip() for part in fingerprint_type.split("+")]
        arrays: list[np.ndarray] = []
        names: list[str] = []
        for part in parts:
            array, part_names = _fingerprint_array(
                molecule,
                fingerprint_type=part,
                n_bits=n_bits,
                radius=radius,
                descriptors=descriptors,
            )
            arrays.append(array)
            names.extend(part_names)
        return np.concatenate(arrays).astype(float), canonical_smiles, names

    array, names = _fingerprint_array(
        molecule,
        fingerprint_type=fingerprint_type,
        n_bits=n_bits,
        radius=radius,
        descriptors=descriptors,
    )
    return array.astype(float), canonical_smiles, names


def featurize_smiles(
    smiles_values: Iterable[object],
    *,
    smiles_type: str = "molecule",
    fingerprint_type: str = "morgan",
    n_bits: int = 1024,
    radius: int = 2,
    descriptors: Sequence[str] = DEFAULT_RDKIT_DESCRIPTORS,
    column_prefix: str | None = None,
    errors: str = "raise",
) -> SmilesFingerprintResult:
    """Featurize a sequence of molecule or polymer SMILES strings."""
    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be either 'raise' or 'coerce'")

    prefix = column_prefix or _default_prefix(smiles_type, fingerprint_type)
    feature_names: list[str] | None = None
    rows: list[np.ndarray] = []
    canonical_smiles: list[str | float] = []
    failures: list[dict[str, object]] = []

    for index, smiles in enumerate(smiles_values):
        try:
            array, canonical, names = fingerprint_smiles(
                smiles,
                smiles_type=smiles_type,
                fingerprint_type=fingerprint_type,
                n_bits=n_bits,
                radius=radius,
                descriptors=descriptors,
            )
            if feature_names is None:
                feature_names = [f"{prefix}_{name}" for name in names]
            rows.append(array)
            canonical_smiles.append(canonical)
        except Exception as exc:
            if errors == "raise":
                raise ValueError(f"Could not featurize SMILES at position {index}: {smiles!r}") from exc
            if feature_names is None:
                feature_names = _feature_names(
                    fingerprint_type=fingerprint_type,
                    n_bits=n_bits,
                    descriptors=descriptors,
                    prefix=prefix,
                )
            rows.append(np.full(len(feature_names), np.nan, dtype=float))
            canonical_smiles.append(np.nan)
            failures.append({"index": index, "smiles": smiles, "error": str(exc)})

    if feature_names is None:
        feature_names = _feature_names(
            fingerprint_type=fingerprint_type,
            n_bits=n_bits,
            descriptors=descriptors,
            prefix=prefix,
        )

    return SmilesFingerprintResult(
        features=pd.DataFrame(rows, columns=feature_names),
        canonical_smiles=pd.Series(canonical_smiles, name=f"{prefix}_canonical_smiles"),
        failed=pd.DataFrame(failures, columns=["index", "smiles", "error"]),
    )


def append_smiles_features(
    data: pd.DataFrame,
    *,
    smiles_column: str,
    smiles_type: str = "molecule",
    fingerprint_type: str = "morgan",
    n_bits: int = 1024,
    radius: int = 2,
    descriptors: Sequence[str] = DEFAULT_RDKIT_DESCRIPTORS,
    column_prefix: str | None = None,
    errors: str = "raise",
    include_canonical_smiles: bool = True,
) -> pd.DataFrame:
    """Append RDKit features for a SMILES column to a dataframe."""
    if smiles_column not in data.columns:
        raise KeyError(f"SMILES column '{smiles_column}' not found")

    result = featurize_smiles(
        data[smiles_column],
        smiles_type=smiles_type,
        fingerprint_type=fingerprint_type,
        n_bits=n_bits,
        radius=radius,
        descriptors=descriptors,
        column_prefix=column_prefix,
        errors=errors,
    )
    features = result.features.set_index(data.index)
    output = data.copy()
    if include_canonical_smiles:
        output[result.canonical_smiles.name] = result.canonical_smiles.set_axis(data.index)
    return pd.concat([output, features], axis=1)


def _canonicalize_by_type(smiles: object, smiles_type: str) -> str:
    normalized = smiles_type.lower()
    if normalized == "molecule":
        return canonicalize_molecule_smiles(smiles)
    if normalized == "polymer":
        return canonicalize_polymer_smiles(smiles)
    raise ValueError("smiles_type must be either 'molecule' or 'polymer'")


def _polymer_endpoint_info(molecule) -> dict[str, tuple[int, int, object]]:
    dummy_atoms = sorted(
        [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0],
        key=lambda atom: atom.GetIdx(),
    )
    if len(dummy_atoms) != 2:
        raise ValueError(
            "Polymer SMILES must contain exactly two [*] dummy atoms; "
            f"found {len(dummy_atoms)}"
        )

    endpoints = []
    for atom in dummy_atoms:
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError("[*] dummy atoms must each have exactly one neighboring atom")
        endpoint_bonds = list(atom.GetBonds())
        endpoints.append((atom.GetIdx(), neighbors[0].GetIdx(), endpoint_bonds[0].GetBondType()))

    if endpoints[0][2] != endpoints[1][2]:
        raise ValueError("Polymer [*] dummy bonds must use the same bond order")

    return {"head": endpoints[0], "tail": endpoints[1]}


def _make_cyclic_polymer_oligomer(
    molecule,
    *,
    endpoint_info: dict[str, tuple[int, int, object]],
    repeat_units: int,
):
    if repeat_units < 1:
        raise ValueError("repeat_units must be at least 1")

    chem = _chem()
    monomer_atom_count = molecule.GetNumAtoms()
    combined = molecule
    for _ in range(repeat_units - 1):
        combined = chem.CombineMols(combined, molecule)

    head_dummy_index, head_neighbor_index, connection_bond_type = endpoint_info["head"]
    tail_dummy_index, tail_neighbor_index, _ = endpoint_info["tail"]
    editable = chem.RWMol(combined)

    for unit_index in range(repeat_units - 1):
        tail_neighbor = unit_index * monomer_atom_count + tail_neighbor_index
        next_head_neighbor = (unit_index + 1) * monomer_atom_count + head_neighbor_index
        if editable.GetBondBetweenAtoms(tail_neighbor, next_head_neighbor) is None:
            editable.AddBond(tail_neighbor, next_head_neighbor, connection_bond_type)

    last_tail_neighbor = (repeat_units - 1) * monomer_atom_count + tail_neighbor_index
    first_head_neighbor = head_neighbor_index
    if editable.GetBondBetweenAtoms(last_tail_neighbor, first_head_neighbor) is None:
        editable.AddBond(last_tail_neighbor, first_head_neighbor, connection_bond_type)

    dummy_indices = []
    for unit_index in range(repeat_units):
        offset = unit_index * monomer_atom_count
        dummy_indices.extend([offset + head_dummy_index, offset + tail_dummy_index])

    for atom_index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(atom_index)

    joined = editable.GetMol()
    chem.SanitizeMol(joined)
    return joined


def _fingerprint_array(
    molecule,
    *,
    fingerprint_type: str,
    n_bits: int,
    radius: int,
    descriptors: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    if fingerprint_type == "morgan":
        generator = _morgan_generator(radius=radius, n_bits=n_bits)
        bit_vector = generator.GetFingerprint(molecule)
        return _bit_vector_to_array(bit_vector), [f"morgan_{i}" for i in range(n_bits)]

    if fingerprint_type == "rdkit":
        bit_vector = _chem().RDKFingerprint(molecule, fpSize=n_bits)
        return _bit_vector_to_array(bit_vector), [f"rdkit_{i}" for i in range(n_bits)]

    if fingerprint_type == "maccs":
        bit_vector = _maccs_keys().GenMACCSKeys(molecule)
        n_maccs_bits = bit_vector.GetNumBits()
        return _bit_vector_to_array(bit_vector), [f"maccs_{i}" for i in range(n_maccs_bits)]

    if fingerprint_type == "descriptors":
        return _descriptor_array(molecule, descriptors)

    raise ValueError(
        "fingerprint_type must be one of: morgan, rdkit, maccs, descriptors, "
        "morgan+descriptors, rdkit+descriptors, maccs+descriptors"
    )


def _descriptor_array(molecule, descriptors: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    descriptor_lookup = dict(_descriptors().descList)
    values = []
    names = []
    for name in descriptors:
        if name not in descriptor_lookup:
            raise ValueError(f"Unsupported RDKit descriptor '{name}'")
        values.append(float(descriptor_lookup[name](molecule)))
        names.append(f"desc_{name}")
    return np.asarray(values, dtype=float), names


def _feature_names(
    *,
    fingerprint_type: str,
    n_bits: int,
    descriptors: Sequence[str],
    prefix: str,
) -> list[str]:
    names: list[str] = []
    for part in fingerprint_type.lower().split("+"):
        part = part.strip()
        if part in {"morgan", "rdkit"}:
            names.extend(f"{part}_{i}" for i in range(n_bits))
        elif part == "maccs":
            names.extend(f"maccs_{i}" for i in range(167))
        elif part == "descriptors":
            names.extend(f"desc_{name}" for name in descriptors)
        else:
            raise ValueError(f"Unsupported fingerprint type '{fingerprint_type}'")
    return [f"{prefix}_{name}" for name in names]


def _default_prefix(smiles_type: str, fingerprint_type: str) -> str:
    return f"{smiles_type.lower()}_{fingerprint_type.lower().replace('+', '_')}"


def _mol_from_smiles(smiles: object):
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        raise ValueError("SMILES is empty")
    molecule = _chem().MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError(f"Invalid SMILES '{smiles}'")
    return molecule


def _bit_vector_to_array(bit_vector) -> np.ndarray:
    array = np.zeros((bit_vector.GetNumBits(),), dtype=int)
    _data_structs().ConvertToNumpyArray(bit_vector, array)
    return array.astype(float)


def _chem():
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for organic and polymer fingerprints. "
            "Install it with `conda install -c conda-forge rdkit`."
        ) from exc
    return Chem


def _data_structs():
    try:
        from rdkit import DataStructs
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for organic and polymer fingerprints. "
            "Install it with `conda install -c conda-forge rdkit`."
        ) from exc
    return DataStructs


def _maccs_keys():
    try:
        from rdkit.Chem import MACCSkeys
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for organic and polymer fingerprints. "
            "Install it with `conda install -c conda-forge rdkit`."
        ) from exc
    return MACCSkeys


def _descriptors():
    try:
        from rdkit.Chem import Descriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for organic and polymer fingerprints. "
            "Install it with `conda install -c conda-forge rdkit`."
        ) from exc
    return Descriptors


def _morgan_generator(*, radius: int, n_bits: int):
    try:
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for organic and polymer fingerprints. "
            "Install it with `conda install -c conda-forge rdkit`."
        ) from exc
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
