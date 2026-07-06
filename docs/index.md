# matgpr

`matgpr` is a Gaussian Process Regression toolkit for materials informatics.
It focuses on reproducible data preparation, materials featurization,
uncertainty-aware GPR, and physics-informed mean functions.
Python 3.10 or newer is required.

The package is designed for small-to-medium materials datasets where uncertainty,
validation protocol, and physical interpretation matter as much as point
prediction accuracy.

## What You Can Build

- Standard GPR baselines with scikit-learn and GPyTorch.
- Physics-informed GPR models where a mechanistic equation enters the GP mean.
- Reusable physics equation templates for common materials trends such as
  Arrhenius behavior, power laws, Hall-Petch strengthening, free-volume
  transport, and mixture rules.
- Physics-aware kernels such as Tanimoto similarity for molecular/polymer
  fingerprints, element-fraction kernels for inorganic compositions, and
  structure-feature kernels for crystal geometry.
- Target-transform and physics-residual workflows for properties that are
  positive, bounded, transformed, or baseline-corrected.
- Soft physics-constraint anchors for known limits and monotonic trends.
- Derivative-constrained GPR for slope-informed physics trends.
- Physics-aware observation-noise profiles for mixed-source and replicate data.
- Multitask GPR for complete and sparse multi-property materials datasets with
  learned inter-task covariance and scikit-learn-style estimator wrappers.
- Reusable validation workflows for train/test splits, cross-validation, and
  configurable learning curves.
- Candidate-generation helpers for finite chemistry, composition,
  formulation, and processing-condition pools.
- Optional BoTorch Bayesian optimization for ranking finite candidate pools,
  passing known observation noise, applying feasibility constraints, selecting
  diverse next-experiment batches, and auditing recommendation decisions.
- Multi-objective finite-pool selection with Pareto-front and weighted
  scalarization utilities.
- Composition, molecule, and polymer fingerprints for materials datasets.
- Learning curves, parity plots, uncertainty diagnostics, PCA plots, and SHAP
  analysis workflows.
- Published-paper PI-GPR examples for OPV and solvent diffusivity. Other
  candidate examples are being reviewed before inclusion in the public example
  set.

## Start Here

1. Try the [Quickstart](quickstart.md) for a compact standard GPR to PI-GPR
   workflow.
2. Review the [Example Cards](example_cards.md) for dataset provenance,
   validation protocols, and model limitations.
3. Review the [Benchmark Summary](benchmark_summary.md) for public example
   gate metrics, citation guidance, and contribution expectations.
4. Read the [User Guide](matgpr_user_guide.md) for the end-to-end workflow.
5. Read [Physics-Informed GPR](physics_informed_gpr.md) before defining custom
   equations.
6. Read [PI-GPR Guarantees](pi_gpr_guarantees.md) before interpreting a
   physics-informed result.
7. Read [Multitask GPR](multitask_gpr.md) for complete or sparse
   multi-property target matrices.
8. Read [Multi-Fidelity GPR](multifidelity_gpr.md) for simulation-plus-
   experiment workflows.
9. Follow the [Sparse Multitask Walkthrough](sparse_multitask_walkthrough.md)
   for incomplete multi-property datasets.
10. Review [Sparse Multitask Noise](sparse_multitask_noise_design.md)
   for shared, task-specific, and known sparse observation noise.
11. Use [Fingerprinting Options](fingerprinting_options.md) to choose descriptors.
12. Review [Versioning And Stability](versioning.md) before pinning a release
   for a paper, benchmark, or production workflow.
13. Use the [Release Checklist](release_checklist.md) before tagging a public
   package release.
14. Review [PyPI Readiness](pypi_readiness.md) before uploading a package
   distribution.
15. Check the [API Reference](api/index.md) when writing scripts or notebooks.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev,examples,bo]"
python -m ruff check matgpr tests scripts
python -m pytest
```

For documentation work:

```bash
python -m pip install -e ".[docs,examples]"
python -m mkdocs serve
```

## Citation

If you use `matgpr` in a publication, cite the archived release DOI:
<https://doi.org/10.5281/zenodo.21210387>. The same citation metadata is also
available in `CITATION.cff`. Individual example reports also list the original
papers and datasets that should be cited.

## License

`matgpr` is released under the Apache License 2.0.
