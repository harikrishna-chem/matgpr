# Hardness Single-Task GPR Example

## Reference

Mukherjee, M.; Ramprasad, R.; Sahu, H. **Load-dependent Hardness Prediction for Materials using Machine Learning.**

## Scope

- Build the first hardness example using only experimental Vickers hardness data.
- Use `only_exp_data.csv` from the source folder as `examples/hardness/dataset.csv`.
- Reserve the combined DFT/experimental dataset for the later multitask model.

## What the Paper Shows

- Experimental hardness is strongly load dependent.
- DFT-derived hardness proxies based on bulk and shear moduli have only moderate correlation with experiment.
- The single-task experimental GPR model performs better than multitask models that mix experimental data with semiempirical DFT-derived hardness values.
- The key descriptor decision is to include indentation load alongside inorganic composition descriptors.

## Models in the Notebook

- `Composition-only GPR`: baseline model using only composition fingerprints.
- `Load-aware GPR`: single-task GPR using composition fingerprints plus applied load and `log1p(load)`.
- `PI-GPR: load mean`: physics-informed GPR with a load-dependent prior mean:

```text
H(load) = H_floor + A / sqrt(load + P0)
```

## Learned Physics Parameters

- `H_floor`: asymptotic hardness baseline at larger loads.
- `A`: strength of the low-load hardness enhancement.
- `P0`: positive load offset that keeps the equation stable near zero load.
- These parameters are learned during GPR training together with kernel hyperparameters.

## Inorganic Fingerprints Added to matgpr

- New module: `matgpr.inorganic_fingerprints`.
- Uses `pymatgen` to parse inorganic formulas and retrieve elemental properties.
- Generates 60 descriptors from 10 elemental properties and 6 statistics.
- Statistics include minimum, maximum, range, fraction-weighted mean, absolute deviation, and weighted standard deviation.

## Validation Plan

- Hold out 30 percent of the dataset for learning-curve evaluation.
- Plot RMSE and R2 from 10 percent to 70 percent training data.
- Select the best low-data model at 20 percent training data.
- Run a 90/10 validation split with 10-fold cross-validation on the 90 percent training partition.
- Show cross-validation statistics on the left and 90/10 train/test parity with uncertainty bars on the right.
- Fit a production model on 100 percent of experimental data after validation.
- Run permutation SHAP on the production model to identify the load and composition descriptors driving hardness predictions.

## Next Step

- Run the notebook with the full settings and inspect whether `Load-aware GPR` or `PI-GPR: load mean` gives the strongest low-data performance.
- If the physics-informed mean is not consistently better, tune the mean equation before moving to multitask learning.
