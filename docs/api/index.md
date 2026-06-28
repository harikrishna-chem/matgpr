# API Reference

The API reference is generated from the package docstrings with `mkdocstrings`.
Use this section when you need exact function signatures, parameters, return
types, and class methods.

## Public Workflow Modules

- [Estimators](estimators.md): scikit-learn-style GPR estimator classes.
- [Featurizers](featurizers.md): scikit-learn-style materials transformers.
- [Target Transforms](target_transforms.md): log targets, standardization, and
  physics-residual modeling helpers.
- [Physics-Informed GPR](gpytorch_gpr.md): GPyTorch training, prediction, and
  mean-function utilities.
- [Uncertainty](uncertainty.md): coverage, calibration, NLPD, and uncertainty
  diagnostics.
- [Fingerprints](fingerprints.md): lower-level composition, SMILES, polymer, and
  cache helpers.
- [Data And Metrics](data.md): cleaning, splitting, preprocessing, metrics, PCA,
  reporting, and artifact utilities.
- [Visualization](visualization.md): plotting helpers for model analysis.

Prefer the high-level estimator and featurizer APIs for reusable scripts.
Use the lower-level functions when building custom research workflows.
