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
- Physics-aware scikit-learn kernels, including Tanimoto fingerprint similarity
  and feature-subset additive/product kernel helpers.
- Element-fraction composition vectors and composition-aware GPR kernels for
  inorganic formula workflows.
- Lightweight crystal-structure descriptors, a `StructureFeaturizer`, and
  structure-aware GPR kernels for global lattice/packing similarity.
- Bounded target transforms for GPR workflows with finite physical output
  limits.
- Materials-property target-transform presets for efficiencies, fractions,
  transport coefficients, electronic gaps, stability energies, mechanical
  properties, transition temperatures, and signed energy targets.
- Soft physics-constraint utilities for known limits and monotonic trends using
  virtual observations.
- Exact RBF derivative-constrained GPR with function-value and derivative
  observations.
- Physics-aware observation-noise profiles for source-dependent, replicate, and
  feature-dependent uncertainty.
- Learned heteroscedastic GPR through a two-stage signal GP plus residual
  log-noise GP workflow.
- Reusable physics equation templates for Arrhenius, square-root-time,
  power-law, Hall-Petch, free-volume, and rule-of-mixtures mean functions.
- Reusable validation workflows for train/test evaluation, cross-validation,
  and configurable learning curves with named models, selectable train-size
  intervals, metric choices, train/test split summaries, and report-ready
  predictions.
- Target transforms for log-scale properties, explicit standardization, and
  physics-residual GPR workflows.
- Published-paper PI-GPR example workflows for OPV and solvent diffusivity.
- Quickstart documentation showing a standard GPR to physics-informed GPR
  workflow.
- Dataset cards and model cards for the OPV and solvent diffusivity public
  examples.
- User guide and fingerprinting-options guide.
- Documentation clarifying what PI-GPR does and does not guarantee.
- Initial exact multitask GPR utilities for complete multi-property target
  matrices.
- Scikit-learn-style multitask GPR estimator wrapper for complete
  multi-property target matrices.
- Versioning and API-stability guide for active-development `0.x` releases.
- Release checklist for `v0.1.0` and later `0.x` releases.
- PyPI readiness audit documenting package metadata, artifact contents,
  TestPyPI flow, and remaining live-upload blockers.
- MkDocs Material documentation site scaffold with docs-build workflow and
  generated API-reference pages.
- Apache-2.0 license metadata and notices.
- CI/package-quality scaffolding.
- Scikit-learn estimator compliance and pipeline tests covering official
  estimator checks, `GridSearchCV`, material featurizers, and GPR pipelines.
- Optional-dependency helper registry with clear install messages and tests for
  advanced fingerprinting backends.
- Package metadata URLs and keywords for documentation, issues, repository, and
  changelog discovery.
- Native estimator missing-value policies: `missing="error"`, `"drop"`, or
  `"impute"`, with fitted `MissingValueReport` summaries.
- Optional matminer Magpie composition descriptors through
  `MagpieCompositionFeaturizer`, `featurize_magpie_compositions`, and
  `append_magpie_composition_features`.
- Physics-equation registry/discovery metadata with feature specs, parameter
  specs, units, assumptions, search helpers, and dataframe summaries.
- Optional BoTorch Bayesian-optimization helpers for fitting `SingleTaskGP`
  surrogates, ranking finite candidate pools, and suggesting next experiments.
- Finite-pool Bayesian-optimization feasibility constraints for filtering or
  annotating candidate recommendations by synthesis, processing, safety, or
  other metadata limits.
- Diversity-aware finite-pool batch selection for next-experiment campaigns.
- Multi-fidelity observation data preparation with explicit fidelity order,
  target fidelity, sample IDs, feature names, and known noise variances.
- Initial two-level autoregressive co-kriging GPR with a learned constant
  `rho`, shared learned noise, lower-level function, estimator wrapper, tests,
  and documentation.

### Changed

- Documentation workflow now builds docs only and does not deploy GitHub Pages
  while the repository remains private.
- Heavy fingerprinting tools are organized as optional dependency extras.
- `matminer` and `mendeleev` are optional ecosystem dependencies, not required
  runtime dependencies.
- Public examples now use `dataset.pkl` instead of raw CSV datasets.
- Gas transport, hardness, and spall strength moved out of the first public
  PI-GPR example set for later review.

### Notes

- `matgpr` is released under the Apache License 2.0.
- Python 3.10 or newer is required.
