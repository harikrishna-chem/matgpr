# Fingerprinting Options in matgpr

This document explains which fingerprinting and descriptor options belong in
`matgpr`, how to choose them for different materials problems, and how users
can implement them in notebooks or scripts.

The package should keep broadly useful, lightweight tools in the core
installation and keep heavier atomistic, graph, or deep-learning tools as
optional extras.

## Dependency Policy

### Core Dependencies

These packages are installed with `matgpr` because they support common
materials-informatics workflows and are useful across many examples.

| Package | Role in matgpr | Typical use |
| --- | --- | --- |
| `pymatgen` | Composition parsing and crystal-structure objects | Inorganic formulas, structure files, elemental properties |
| `rdkit` | Molecule and polymer SMILES fingerprints | Organic molecules, solvents, polymer repeat units |
| `matminer` | Standard materials featurizers | Composition, structure, site, band-structure, and DOS descriptors |
| `mendeleev` | Lightweight elemental-property table | Custom composition descriptors and element metadata |

### Optional Dependencies

These tools are valuable, but they should not be required for every `matgpr`
user because they are heavier, more specialized, or more likely to introduce
platform-specific dependency constraints.

| Extra | Packages | Use when |
| --- | --- | --- |
| `structures` | `ase`, `dscribe` | SOAP, ACSF, MBTR, Coulomb/Sine/Ewald matrix descriptors |
| `molecular-extra` | `mordredcommunity` | Large molecular descriptor sets beyond RDKit defaults |
| `jarvis` | `jarvis-tools` | JARVIS datasets and JARVIS-style materials descriptors |
| `deep` | `deepchem` | Graph/deep-learning featurizers and neural workflows |
| `all-fingerprints` | all optional fingerprint packages | You want one broad local environment for experimentation |

Install examples:

```bash
python -m pip install -e .
python -m pip install -e ".[structures]"
python -m pip install -e ".[molecular-extra]"
python -m pip install -e ".[all-fingerprints]"
```

## How To Choose Fingerprints

### 1. Inorganic Composition Only

Use this when the dataset has formulas such as `B4C`, `Al2O3`, or
`Ag0.05Gd0.048Pd0.902`, but no crystal structure.

Recommended options:

| Option | Package | Use case |
| --- | --- | --- |
| matgpr composition descriptors | `pymatgen` | Fast baseline descriptors from elemental-property statistics |
| matgpr element-fraction vectors | `pymatgen` | Direct inputs for composition-aware kernels |
| matminer composition featurizers | `matminer` | Stronger published baseline descriptors |
| custom elemental descriptors | `mendeleev` | When a domain-specific property table is needed |

Current `matgpr` implementation:

```python
from matgpr import CompositionFeaturizer, append_composition_fingerprints, append_element_fractions

model_data = append_composition_fingerprints(
    data,
    formula_column="composition",
    errors="coerce",
).dropna().reset_index(drop=True)

composition_features = CompositionFeaturizer(
    formula_column="composition",
    errors="coerce",
    cache_dir="fingerprint_cache",
).fit_transform(data)

element_fraction_features = append_element_fractions(
    data,
    formula_column="composition",
    elements=("B", "C", "N", "O", "Al", "Si"),
    errors="coerce",
)
```

Use `cache_dir` for repeated notebooks or cross-validation workflows. Cache
keys are deterministic hashes of the input value and featurization settings.
Failed-row reports include those keys so problematic rows can be traced across
runs.

Best default for early GPR examples:

1. Start with `append_composition_fingerprints`.
2. Add process variables, measurement conditions, or physical descriptors.
3. Compare against `matminer` composition featurizers when the baseline is
   stable.
4. Use `append_element_fractions` when the model will use
   `ElementFractionKernel`.

### 2. Inorganic Crystal Structures

Use this when CIF files, POSCAR files, Materials Project structures, or
`pymatgen.Structure` objects are available.

Recommended options:

| Option | Package | Use case |
| --- | --- | --- |
| matgpr structure descriptors | `pymatgen` | Lightweight lattice, volume, and density descriptors |
| matminer structure featurizers | `matminer` | Global crystal-structure descriptors for tabular ML |
| SOAP/ACSF/MBTR | `dscribe` | Atomistic local-environment and many-body descriptors |
| ASE object conversion | `ase` | Bridge between structure formats and atomistic descriptors |
| JARVIS descriptors | `jarvis-tools` | JARVIS-compatible structure workflows |

Current `matgpr` implementation:

```python
from pymatgen.core import Structure

from matgpr import StructureFeaturizer, append_structure_fingerprints

structure = Structure.from_file("material.cif")

structure_features = append_structure_fingerprints(
    data,
    structure_column="structure",
    errors="coerce",
)

structure_featurizer = StructureFeaturizer(
    structure_column="structure",
    errors="coerce",
    cache_dir="fingerprint_cache",
).fit_transform(data)
```

Practical guidance:

- For small tabular GPR, start with `append_structure_fingerprints` and
  `StructureFeatureKernel`.
- Compare against `matminer` structure featurizers when structure information
  appears important.
- For local atomic environments, defects, surfaces, or force-field-like
  descriptors, use `DScribe`.
- For graph neural networks, keep the graph featurizer optional and separate
  from the core GPR workflow.

Example pattern with optional `matminer` structure featurizers:

```python
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from pymatgen.core import Structure

structure = Structure.from_file("material.cif")
featurizers = [DensityFeatures(), GlobalSymmetryFeatures()]

features = {}
for featurizer in featurizers:
    labels = featurizer.feature_labels()
    values = featurizer.featurize(structure)
    features.update(dict(zip(labels, values)))
```

### 3. Organic Molecules and Solvents

Use this when inputs are ordinary molecule SMILES.

Recommended options:

| Option | Package | Use case |
| --- | --- | --- |
| RDKit Morgan fingerprints | `rdkit` | Strong default for molecular similarity and GPR |
| RDKit descriptors | `rdkit` | Interpretable molecular properties |
| MACCS keys | `rdkit` | Compact chemistry key baseline |
| Mordred descriptors | `mordredcommunity` | Large descriptor pool for feature selection |
| DeepChem featurizers | `deepchem` | Graph/deep-learning workflows |

Current `matgpr` implementation:

```python
from matgpr import SmilesFeaturizer, featurize_smiles

result = featurize_smiles(
    data["solvent_smiles"],
    smiles_type="molecule",
    fingerprint_type="morgan+descriptors",
    n_bits=256,
    column_prefix="solvent",
    errors="coerce",
)

solvent_features = SmilesFeaturizer(
    smiles_column="solvent_smiles",
    fingerprint_type="morgan+descriptors",
    n_bits=256,
    column_prefix="solvent",
    errors="coerce",
    cache_dir="fingerprint_cache",
).fit_transform(data)
```

Best default:

- Use `morgan+descriptors` for first GPR models.
- Use `descriptors` when interpretability matters more than raw performance.
- Add Mordred only when feature selection or SHAP analysis can control the
  larger descriptor space.

### 4. Polymer Repeat Units

Use this when polymer repeat-unit SMILES contain exactly two `[*]` dummy atoms.

Current `matgpr` polymer handling:

1. Identify the two `[*]` end atoms.
2. Build a trimer from the repeat unit.
3. Connect adjacent repeat units using the dummy-end bond order.
4. Close the trimer into a loop.
5. Remove all `[*]` atoms.
6. RDKit-canonicalize the cyclic trimer surrogate.
7. Fingerprint the canonicalized surrogate.

Example:

```python
from matgpr import PolymerSmilesFeaturizer, canonicalize_polymer_smiles, featurize_smiles

canonical = canonicalize_polymer_smiles("[*]CC[*]")
print(canonical)  # C1CCCCC1

polymer_result = featurize_smiles(
    data["polymer_smiles"],
    smiles_type="polymer",
    fingerprint_type="morgan+descriptors",
    n_bits=256,
    column_prefix="polymer",
    errors="coerce",
)

polymer_features = PolymerSmilesFeaturizer(
    smiles_column="polymer_smiles",
    fingerprint_type="morgan+descriptors",
    n_bits=256,
    column_prefix="polymer",
    errors="coerce",
    cache_dir="fingerprint_cache",
).fit_transform(data)
```

Recommended polymer workflow:

- Start with cyclic-trimer RDKit Morgan fingerprints.
- Add RDKit descriptors for interpretability.
- Add polymer-specific physics or condition features separately.
- Use SHAP on a compact selected feature set, not every fingerprint bit.

### 5. Experimental Conditions and Physics Features

Materials fingerprints should usually be combined with experimental conditions
or physics-derived features. Examples:

| Domain | Useful non-fingerprint features |
| --- | --- |
| hardness | load, `log1p(load)`, test method |
| solvent diffusivity | temperature, solvent concentration, solvent molecular weight |
| gas transport | diffusivity, solubility, gas identity, temperature |
| spall strength | yield strength, bulk modulus, fracture toughness, density |
| OPV | degeneracy, binding energy, donor/acceptor descriptors |

These columns should remain explicit. Do not hide them inside a generic
fingerprint block if they are part of the physical interpretation.

## Recommended Defaults by Dataset Type

| Dataset type | First model | Stronger next model | Optional advanced model |
| --- | --- | --- | --- |
| inorganic formula only | `pymatgen` composition descriptors | `matminer` composition featurizers | element embeddings or learned composition models |
| inorganic crystal structures | `matminer` structure featurizers | `DScribe` SOAP/MBTR | graph neural network descriptors |
| molecule SMILES | RDKit Morgan + descriptors | Mordred-selected descriptors | DeepChem graph featurizers |
| polymer repeat-unit SMILES | cyclic-trimer RDKit Morgan + descriptors | add Mordred or polymer physics features | polymer graph/sequence model |
| mixed materials and conditions | material fingerprint + condition columns | physics-informed mean features | multitask/BO workflow |

## Implementation Plan for matgpr

### Current Public API

Already implemented:

- `composition_fingerprint`
- `featurize_compositions`
- `append_composition_fingerprints`
- `default_element_symbols`
- `element_fraction_fingerprint`
- `featurize_element_fractions`
- `append_element_fractions`
- `structure_fingerprint`
- `featurize_structures`
- `append_structure_fingerprints`
- `canonicalize_molecule_smiles`
- `canonicalize_polymer_smiles`
- `fingerprint_smiles`
- `featurize_smiles`
- `append_smiles_features`
- `CompositionFeaturizer`
- `StructureFeaturizer`
- `SmilesFeaturizer`
- `PolymerSmilesFeaturizer`
- deterministic fingerprint cache keys through `fingerprint_cache_key`

### Near-Term Additions

Recommended next wrappers:

| Wrapper | Backend | Purpose |
| --- | --- | --- |
| `featurize_matminer_compositions` | `matminer` | Standard composition featurizer set |
| `append_matminer_composition_features` | `matminer` | Add matminer descriptors to a dataframe |
| `featurize_mendeleev_compositions` | `mendeleev` | Custom lightweight elemental-property descriptors |
| `featurize_matminer_structures` | `matminer` | Larger published structure featurizer set |

Keep optional wrappers import-safe. Use `require_optional_dependency(...)` at
the point of use so users get a clear install message without importing heavy
packages during `import matgpr`:

```python
from matgpr import require_optional_dependency

dscribe = require_optional_dependency("dscribe")
```

## Practical Selection Rules

1. If only formulas are available, use composition descriptors first.
2. If SMILES are available, use RDKit Morgan fingerprints plus descriptors.
3. If polymer repeat units are available, use the cyclic-trimer polymer
   canonicalization before fingerprinting.
4. If structures are available, start with matminer structure descriptors before
   moving to SOAP/MBTR.
5. If the dataset is small, prefer interpretable descriptors and simple physics
   features over very large fingerprint spaces.
6. If the dataset is large and diverse, consider optional deep/graph
   featurizers, but keep them out of the default dependency path.
7. Always report which fingerprint backend, fingerprint length, descriptor set,
   and preprocessing choices were used.

## Reporting Checklist

Each example notebook or report should state:

- input representation: formula, molecule SMILES, polymer SMILES, CIF,
  tabular property columns, or mixed inputs,
- fingerprint backend: `pymatgen`, `rdkit`, `matminer`, `mendeleev`,
  `dscribe`, etc.,
- fingerprint settings: bit length, radius, descriptor set, repeat-unit
  handling, or structure featurizer names,
- failed-row handling: `errors="raise"` or `errors="coerce"`,
- whether descriptors were scaled,
- whether physics/condition features were included separately,
- which feature set was used for SHAP or other interpretation.
