# Dataset Card: OPV PCE

## Summary

This dataset supports the OPV physics-informed GPR example in
`opv_gpr_modeling.ipynb`. The task is to predict organic photovoltaic power
conversion efficiency (PCE) from molecular and electronic descriptors.

## Dataset Location

- File: `examples/opv/dataset.pkl`
- Notebook: `examples/opv/opv_gpr_modeling.ipynb`
- Report: `examples/opv/physics_informed_gpr_report.md`
- Rows: 280 OPV donor-acceptor records
- Target column: `PCE`
- Descriptor columns used by the notebook: 13 numeric OPV descriptors after
  removing the record identifier and target

## Reference

H. Sahu, W. Rao, A. Troisi, and H. Ma, "Toward Predicting Efficiency of
Organic Solar Cells via Machine Learning and Improved Descriptors," Advanced
Energy Materials, 8, 1801032, 2018. DOI:
[10.1002/aenm.201801032](https://doi.org/10.1002/aenm.201801032).

Users should cite the original paper and `matgpr` when using this example in
public work.

## Data Type And Provenance

- Data source type: published-paper materials-informatics dataset.
- Measurement target: OPV power conversion efficiency in percent.
- Inputs: descriptor table derived from donor, acceptor, and OPV-relevant
  electronic structure quantities.
- This example should be treated as a tutorial dataset for reproducible GPR
  workflows, not as a complete OPV benchmarking resource.

## Target

`PCE` is a bounded efficiency percentage. The notebook keeps PCE in percent
units so the physics-informed mean functions return interpretable PCE values.
The notebook also demonstrates the `make_materials_target_transform("pce")`
preset for workflows where users want an explicit bounded target transform.

## Features Used

The notebook normalizes feature names before modeling. The core descriptor set
includes:

- `polarizability`
- `delLA`
- `delLD`
- `N_atom`
- `Eg`
- `lamda_h`
- `DIP`
- `AL-DH`
- `delHD`
- `E_bind`
- `DL-AL`
- `delGE`
- `E_T1`

The physics-informed mean functions use only a subset of these descriptors:

- `delHD`, `delLD`, and `delLA` for frontier-orbital near-degeneracy.
- `E_bind` for hole-electron binding energy.

The remaining descriptors are still available to the GP kernel, which learns
residual structure around the physics-informed mean.

## Cleaning And Preparation

The notebook:

- loads `dataset.pkl`,
- normalizes column names,
- separates `PCE` from descriptor columns,
- uses only numeric descriptor columns,
- computes all physics-score normalization statistics from the active training
  subset only,
- keeps external test data out of preprocessing and physics-score fitting.

## Validation Protocol

The public notebook uses:

- a fixed 30 percent external test set for the main learning curves,
- a 70 percent training pool,
- learning curves from 10 to 70 percent of the full dataset,
- 20 random stratified subsets at each learning-curve point,
- RMSE and R2 with standard-deviation error bars,
- a low-data parity comparison using 10 percent training data,
- a 90/10 train-test validation split with 10-fold cross-validation inside the
  90 percent training partition,
- a final production fit on 100 percent of the data after model selection.

## Intended Use

This dataset card is intended for users who want to:

- reproduce the OPV PI-GPR notebook,
- understand the descriptors used by the physics mean functions,
- compare standard GPR and physics-informed GPR in a low-data setting,
- adapt the workflow to another molecular materials dataset.

## Out-Of-Scope Use

The dataset should not be used as:

- a standalone claim of state-of-the-art OPV prediction,
- a substitute for device-level experimental validation,
- a universal OPV design benchmark without checking descriptor consistency and
  applicability domain.

## Known Limitations

- The dataset is small relative to the diversity of OPV chemistry and device
  fabrication conditions.
- PCE depends on processing, morphology, interfaces, and measurement details
  that are not fully represented by the descriptor table.
- Physics scores are simple tutorial priors, not a complete OPV device model.
- Predictions outside the descriptor range of the dataset should be treated as
  extrapolations.

