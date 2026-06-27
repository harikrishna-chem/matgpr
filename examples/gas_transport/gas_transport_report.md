# Gas Transport Physics-Informed GPR Report

## Reference

Phan, B. K.; Shen, K.-H.; Gurnani, R.; Tran, H.; Lively, R.; Ramprasad, R. **Gas permeability, diffusivity, and solubility in polymers: Simulation-experiment data fusion and multi-task machine learning.** npj Computational Materials 10, 185 (2024).

## Dataset

- File: `gas_transport_wide.csv`.
- First-version target: experimental CO2 permeability, `p_exp_CO2`.
- Physics features: experimental CO2 diffusivity `d_exp_CO2` and solubility `s_exp_CO2`.
- Rows are restricted to polymers with all three experimental quantities available.

## Polymer Featurization

- Polymer repeat units must contain exactly two `[*]` atoms.
- The repeat unit is expanded into a cyclic trimer before RDKit canonicalization.
- The two `[*]` ends define how adjacent repeat units are connected, including the dummy-end bond order, and all `[*]` atoms are removed before fingerprinting.
- Morgan fingerprints plus selected RDKit descriptors are used as the chemical feature space.

## Physics Basis

The solution-diffusion model relates permeability, diffusivity, and solubility:

```text
P = D * S
```

In log10 units:

```text
log10(P) = log10(D) + log10(S)
```

## Models Tested

- `Standard GPR`: polymer fingerprint baseline.
- `PI-GPR: logD + logS`: fixed solution-diffusion slope with learned bias.
- `PI-GPR: learned D/S weights`: learned positive weights on `logD` and `logS`.
- `PI-GPR: diffusivity`: uses only diffusivity in the mean.
- `PI-GPR: solubility`: uses only solubility in the mean.

## Learned Parameters

- `bias`: systematic offset between measured log permeability and the ideal logD + logS relation.
- `w_d`: learned positive weight on log diffusivity.
- `w_s`: learned positive weight on log solubility.

## Learning Curve Protocol

- Training pool/test split is repeated 10 times to keep the exact-GPR benchmark practical.
- Training pool usage is varied from 10 percent to 100 percent using `[10, 30, 50, 70, 90, 100]`.
- Each point reports mean and standard deviation across the 10 splits.
- Metrics: RMSE, R2, MAE, and Pearson r.

## 90/10 Validation And SHAP

- After low-data model selection, the selected model is evaluated on a 90/10 train-test split.
- The 90 percent training partition is evaluated with 10-fold cross-validation.
- The validation figure reports cross-validation statistics on the left and train/test parity with uncertainty bars on the right.
- The production model is then refit on 100 percent of the filtered data.
- Permutation SHAP is applied to the production model using the solution-diffusion physics features and the most target-correlated fingerprint/descriptor columns.

## Notes

- This example intentionally starts with a clean single-task subset.
- The full paper uses multitask fusion across permeability, diffusivity, solubility, gas type, and data fidelity.
- Later versions can extend this notebook to multitask GPR.
