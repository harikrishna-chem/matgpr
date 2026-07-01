# Solvent Diffusivity Physics-Informed GPR Report

## Reference

Nistane, J.; Datta, R.; Lee, Y. J.; Sahu, H.; Jang, S. S.; Lively, R.; Ramprasad, R. **Polymer design for solvent separations by integrating simulations, experiments and known physics via machine learning.** npj Computational Materials 11, 187 (2025).

## Dataset

- File: `dataset.pkl`.
- Target: `Target Diffusivity Value(log10)`.
- First-version scope: experimental rows only.
- Polymer input: `Polymer Canonical SMILES`.
- Solvent input: `Solvent Canonical SMILES`.
- Condition inputs: temperature and solvent weight fraction.

## Polymer And Molecule Featurization

- Polymer repeat units must contain exactly two `[*]` atoms.
- The repeat unit is expanded into a trimer using the two `[*]` end atoms.
- Adjacent repeat units are connected through the end-atom neighbors using the dummy-end bond order, the trimer is closed into a loop, and all `[*]` atoms are removed.
- The resulting cyclic trimer surrogate is RDKit-canonicalized before fingerprinting.
- Solvent molecules are RDKit-canonicalized before fingerprinting.
- Initial fingerprints use Morgan bits plus selected RDKit descriptors.

## Models Tested

- `Standard GPR`: RDKit polymer and solvent features plus condition variables.
- `PI-GPR: Arrhenius`: adds an Arrhenius prior mean for temperature dependence.
- `PI-GPR: concentration`: adds a concentration/plasticization prior mean.
- `PI-GPR: solvent size`: adds a solvent-size penalty prior mean.
- `PI-GPR: combined`: combines Arrhenius, concentration, and solvent-size terms.

## Physics Mean Functions

### Arrhenius

```text
log10(D) = log10(D0) - Ea / (ln(10) R T)
```

- Features: temperature `T`.
- Learned parameters: `log_d0`, `activation_energy_kj_mol`.
- Constraint: activation energy is positive.

### Concentration

```text
log10(D) = baseline + concentration_slope * log10(weight_fraction)
```

- Features: solvent weight fraction.
- Learned parameters: `baseline`, `concentration_slope`.
- Constraint: concentration slope is positive.

### Solvent Size

```text
log10(D) = baseline - size_penalty * log10(MW_solvent)
```

- Features: RDKit solvent molecular weight.
- Learned parameters: `baseline`, `size_penalty`.
- Constraint: size penalty is positive.

### Combined

```text
log10(D) = log10(D0)
           - Ea / (ln(10) R T)
           + concentration_slope * log10(weight_fraction)
           - size_penalty * log10(MW_solvent)
```

- Features: temperature, solvent weight fraction, solvent molecular weight.
- Learned parameters: `log_d0`, `activation_energy_kj_mol`, `concentration_slope`, `size_penalty`.

## Learning Curve Protocol

- Training pool/test split is repeated 10 times to keep the exact-GPR benchmark practical.
- Training pool usage is varied from 10 percent to 100 percent using `[10, 20, 30, 50, 70, 90, 100]`.
- Each point reports mean and standard deviation across the 10 splits.
- Metrics: RMSE, R2, MAE, and Pearson r.

## 20 Percent Release Gate

- Standard GPR RMSE: 1.384.
- Retained PI-GPR model: `PI-GPR: concentration`.
- Retained PI-GPR RMSE: 1.333.
- RMSE advantage: 0.051.

This example is retained because the concentration prior mean gives a modest
but positive low-data improvement over Standard GPR.

## 90/10 Cross-Validation And Production Model

- After low-data model selection, the selected PI-GPR model is evaluated on a
  90/10 train-test split only if it beats Standard GPR at the 20 percent
  training-data RMSE comparison.
- The 90 percent training partition is evaluated with 10-fold cross-validation.
- The validation figure reports cross-validation statistics for the 90 percent training partition.
- The production model is then refit on 100 percent of the filtered data.

## Notes

- This is a single-task GPR version of a paper that emphasizes multitask and physics-enforced models.
- The reduced 10-split benchmark is still computationally expensive because exact GPR is cubic in training-set size.
- After initial runs, keep only the strongest and easiest-to-explain physics models.
