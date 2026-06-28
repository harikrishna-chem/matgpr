# Changelog

All notable changes to `matgpr` will be documented in this file.

The format follows the spirit of Keep a Changelog, and versioning will follow
semantic-versioning conventions once the first public release is tagged.

## Unreleased

### Added

- Physics-informed GPR mean-function API through `PhysicsInformedMean`.
- GPyTorch exact GPR training and prediction helpers with uncertainty support.
- Scikit-learn GPR helper functions.
- Data cleaning, preprocessing, metrics, PCA, visualization, reporting, and
  artifact utilities.
- Inorganic composition fingerprints based on `pymatgen`.
- Organic molecule and polymer fingerprints based on RDKit.
- Cyclic-trimer polymer repeat-unit canonicalization for two-ended `[*]`
  polymer SMILES.
- Scikit-learn-style `MatGPRRegressor` and `PhysicsInformedGPRRegressor`
  estimator classes.
- Scikit-learn-style `CompositionFeaturizer`, `SmilesFeaturizer`, and
  `PolymerSmilesFeaturizer` transformer classes.
- Deterministic row-level fingerprint caching with cache keys, cache-hit
  reports, and cache keys in failed-row reports.
- Predictive uncertainty diagnostics for interval coverage, calibration curves,
  Gaussian NLPD, standardized residuals, and uncertainty-error plots.
- Published-paper example workflows for OPV, hardness, gas transport, solvent
  diffusivity, and spall strength.
- User guide and fingerprinting-options guide.
- MkDocs Material documentation site scaffold with docs-build workflow and
  generated API-reference pages.
- License-strategy note for source-available/academic-use licensing decision.
- Dual-license metadata and notices: AGPL-3.0 community license plus separate
  commercial license availability.
- CI/package-quality scaffolding.

### Changed

- Documentation workflow now builds docs only and does not deploy GitHub Pages
  while the repository remains private.
- Heavy fingerprinting tools are organized as optional dependency extras.
- New examples include 90/10 validation, 10-fold cross-validation, uncertainty
  parity plots, and production-model interpretation.

### Notes

- The community license is AGPL-3.0. Proprietary or closed-source commercial use
  requires a separate commercial license.
