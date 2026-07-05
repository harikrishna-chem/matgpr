# Public Example Benchmark Summary

This page summarizes the current public `matgpr` example benchmarks. The goal
is transparency, not leaderboard-style claims. Each benchmark is a tutorial
workflow using a published materials dataset, a documented validation protocol,
and a clear comparison between standard GPR and physics-informed GPR.

## Current Public Benchmarks

| Example | Target | Public dataset size | Main comparison | Low-data retention gate | Retained PI-GPR model |
| --- | --- | ---: | --- | --- | --- |
| OPV PCE | Power conversion efficiency (%) | 280 records | Standard GPR vs OPV physics mean functions | 20 percent training-data RMSE: standard GPR 1.520, retained PI-GPR 1.278 | Degeneracy + binding |
| Solvent diffusivity | Log10 experimental diffusivity | 2,339 filtered experimental rows | Standard GPR vs transport-inspired mean functions | 20 percent training-data RMSE: standard GPR 1.384, retained PI-GPR 1.333 | Concentration |

The values above are copied from the public model cards. They are intended as
release-gate summaries for the example notebooks, not as universal claims of
state-of-the-art performance.

## Validation Standards

Public physics-informed examples should report:

- the exact dataset file and provenance,
- target definition and units,
- descriptors and physics features used by the GP,
- standard GPR baseline with the same train/test protocol,
- repeated learning curves with mean and standard deviation,
- held-out test or external split behavior with uncertainty where useful,
- the low-data gate used to decide whether the PI-GPR example stays public,
- the learned physics parameters and their physical interpretation,
- known limitations and likely applicability domain.

The current public examples use `dataset.pkl` files, dataset cards, model
cards, and notebooks so that users can audit the validation protocol before
adapting the workflow to a new materials problem.

## How To Cite

If a publication, report, or benchmark uses `matgpr`, cite:

- `matgpr` using `CITATION.cff`,
- the original paper or dataset for each example or dataset used,
- any external descriptor or chemistry package that materially affects the
  workflow, such as RDKit, pymatgen, or matminer when applicable.

For the public examples:

- OPV PCE users should cite the OPV source paper listed in
  `examples/opv/dataset_card.md`.
- Solvent diffusivity users should cite the polymer-solvent source paper
  listed in `examples/solvent_diffusivity/dataset_card.md`.

When publishing benchmark numbers, report the `matgpr` version or commit hash,
random seeds, train/test split protocol, and whether notebook smoke settings or
full benchmark settings were used.

## Adding New Physics-Informed Examples

New public examples should be held to the same standard as OPV and solvent
diffusivity. Before proposing a new example, prepare:

- a `dataset.pkl` or a documented data-preparation script,
- a notebook with standard GPR and physics-informed GPR comparisons,
- repeated learning curves with enough splits for stable conclusions,
- a dataset card with provenance, target units, cleaning rules, and limitations,
- a model card with equations, learned parameters, validation summary, and
  intended use,
- a short report explaining why the physics prior is reasonable,
- citation text for the source dataset and any non-standard dependencies.

Only keep a physics-informed public example when it demonstrates a meaningful,
well-documented improvement under the stated low-data protocol. If the result
is inconclusive, keep the workflow local or under review until the physics
model, descriptors, or validation design are stronger.
