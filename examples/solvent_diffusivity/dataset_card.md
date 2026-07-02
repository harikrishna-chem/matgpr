# Dataset Card: Solvent Diffusivity

## Summary

This dataset supports the solvent diffusivity physics-informed GPR example in
`solvent_diffusivity_gpr_modeling.ipynb`. The task is to predict experimental
solvent diffusivity in polymers from polymer structures, solvent structures,
and measurement conditions.

## Dataset Location

- File: `examples/solvent_diffusivity/dataset.pkl`
- Notebook: `examples/solvent_diffusivity/solvent_diffusivity_gpr_modeling.ipynb`
- Report: `examples/solvent_diffusivity/solvent_diffusivity_report.md`
- Raw diffusivity rows: 3,044
- Experimental diffusivity rows: 2,421
- Rows retained after strict polymer filter: 2,339
- Rows retained after RDKit featurization: 2,339
- Retained polymers: 81 unique polymer SMILES strings
- Retained solvents: 151 unique solvent SMILES strings

## Reference

Nistane, J.; Datta, R.; Lee, Y. J.; Sahu, H.; Jang, S. S.; Lively, R.;
Ramprasad, R. "Polymer design for solvent separations by integrating
simulations, experiments and known physics via machine learning." npj
Computational Materials 11, 187 (2025).

Users should cite the original paper and `matgpr` when using this example in
public work.

## Data Type And Provenance

- Data source type: published polymer-solvent diffusivity dataset.
- First public `matgpr` scope: experimental diffusivity rows only.
- Excluded from this first single-task example: simulated diffusivity rows and
  sorption multitask targets.
- The notebook is a first single-task GPR demonstration for a paper that also
  motivates richer multitask and physics-enforced modeling.

## Target

The target is stored as:

```text
Target Diffusivity Value(log10)
```

The notebook renames it to `log10_diffusivity` and models it directly in
log10 diffusivity units. It also creates a raw positive `diffusivity` column to
demonstrate the `make_materials_target_transform("diffusivity")` preset.

## Inputs

The notebook starts from:

- `Polymer Canonical SMILES`
- `Solvent Canonical SMILES`
- `Temperature`
- `Weight Fraction of Solvent`
- `Target Diffusivity Value(log10)`
- `Experimental Selector`

After cleaning, the main model uses:

- 128-bit Morgan polymer fingerprint columns,
- 10 polymer RDKit descriptor columns,
- 128-bit Morgan solvent fingerprint columns,
- 10 solvent RDKit descriptor columns,
- `temperature_k`,
- `log_weight_fraction`,
- `solvent_molwt`.

This gives 276 RDKit-derived fingerprint/descriptor features plus 3 physics
features.

## Polymer And Molecule Preparation

The polymer preparation rule is specific and intentional:

- polymer repeat-unit SMILES must contain exactly two `[*]` dummy atoms,
- repeat units with fewer or more dummy atoms are rejected,
- the repeat unit is expanded into a trimer through the two dummy-end
  neighbors,
- the trimer is closed into a loop,
- all `[*]` atoms are removed,
- the cyclic trimer surrogate is RDKit-canonicalized,
- the canonicalized structure is fingerprinted.

Solvent molecules are RDKit-canonicalized before fingerprinting.

## Cleaning And Filtering

The notebook:

- keeps rows where `Experimental Selector == 1`,
- drops rows missing polymer SMILES, solvent SMILES, temperature, solvent
  weight fraction, or target diffusivity,
- keeps only polymer SMILES with exactly two `[*]` atoms,
- requires positive solvent weight fraction,
- drops rows with failed RDKit features.

No RDKit failures are observed after the strict polymer filter in the current
public dataset.

## Validation Protocol

The public notebook uses:

- repeated 80/20 train-test splits,
- training percentages of 10, 20, 30, 50, 70, 90, and 100 percent of the
  training pool,
- 10 random splits per learning-curve point to keep exact-GPR runtime
  practical,
- RMSE, R2, MAE, and Pearson r summaries,
- a 20 percent low-data release gate for retaining the PI-GPR example,
- a 90/10 train-test validation split with 10-fold cross-validation inside the
  90 percent training partition,
- a final production fit on 100 percent of the filtered data.

## Intended Use

This dataset card is intended for users who want to:

- reproduce the solvent diffusivity PI-GPR notebook,
- understand how polymer SMILES with `[*]` end groups are converted into
  fingerprints,
- compare simple temperature, concentration, and solvent-size physics means,
- adapt the workflow to another polymer-solvent transport dataset.

## Out-Of-Scope Use

The dataset should not be used as:

- a final production model for all polymer-solvent separations,
- evidence that a single-task exact GPR model is sufficient for the full
  original paper problem,
- a substitute for validating candidate membranes under the intended
  experimental conditions.

## Known Limitations

- The cyclic trimer is a practical fingerprint surrogate, not a full polymer
  physics representation.
- Exact GPR can become expensive as the retained dataset grows.
- The first public notebook excludes simulated rows and sorption targets.
- Applicability is limited to polymers, solvents, temperatures, and
  concentrations close to those represented in the filtered dataset.

