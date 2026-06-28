# matgpr

`matgpr` is a Gaussian Process Regression toolkit for materials informatics.
It focuses on reproducible data preparation, materials featurization,
uncertainty-aware GPR, and physics-informed mean functions.

The package is designed for small-to-medium materials datasets where uncertainty,
validation protocol, and physical interpretation matter as much as point
prediction accuracy.

## What You Can Build

- Standard GPR baselines with scikit-learn and GPyTorch.
- Physics-informed GPR models where a mechanistic equation enters the GP mean.
- Target-transform and physics-residual workflows for properties that are
  better modeled in transformed or baseline-corrected spaces.
- Composition, molecule, and polymer fingerprints for materials datasets.
- Learning curves, parity plots, uncertainty diagnostics, PCA plots, and SHAP
  analysis workflows.
- Published-paper examples for OPV, hardness, gas transport, solvent
  diffusivity, and spall strength.

## Start Here

1. Read the [User Guide](matgpr_user_guide.md) for the end-to-end workflow.
2. Read [Physics-Informed GPR](physics_informed_gpr.md) before defining custom
   equations.
3. Use [Fingerprinting Options](fingerprinting_options.md) to choose descriptors.
4. Check the [API Reference](api/index.md) when writing scripts or notebooks.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev,examples]"
python -m ruff check matgpr tests scripts
python -m pytest
```

For documentation work:

```bash
python -m pip install -e ".[docs,examples]"
python -m mkdocs serve
```

## Citation

If you use `matgpr` in a publication, cite the package using `CITATION.cff`.
Individual example reports also list the original papers and datasets that
should be cited.

## License

`matgpr` is dual-licensed: AGPL-3.0 for community use, with a separate
commercial license available for proprietary or closed-source commercial
applications.
