# OPV Multi-Objective Bayesian Optimization Tradeoff Report

## Purpose

This example demonstrates a finite-pool multi-objective Bayesian-optimization
workflow with `matgpr`. It uses the OPV dataset retrospectively to show how a
materials discovery campaign can balance more than one design objective.

The demonstration objective is:

- maximize OPV power conversion efficiency,
- minimize exciton binding energy.

## Reference Dataset

H. Sahu, W. Rao, A. Troisi, and H. Ma, "Toward Predicting Efficiency of
Organic Solar Cells via Machine Learning and Improved Descriptors," Advanced
Energy Materials, 8, 1801032, 2018. DOI:
[10.1002/aenm.201801032](https://doi.org/10.1002/aenm.201801032).

## Workflow

- Load the OPV descriptor dataset from `examples/opv/dataset.pkl`.
- Treat a small OPV subset as measured training data.
- Treat the remaining OPV rows as a finite candidate pool.
- Build numeric candidate descriptors from OPV molecular descriptors.
- Add two simple physics scores:
  - frontier-orbital degeneracy score,
  - low exciton-binding-energy score.
- Standardize candidate features using only measured rows.
- Define two objective specifications:
  - PCE, maximized,
  - binding energy, minimized.
- Rank candidates with BoTorch expected hypervolume improvement.
- Summarize the recommendations with the BO audit utilities.
- Plot the candidate pool, retrospective Pareto front, and selected candidates.

## Why Multi-Objective BO

Single-objective BO is useful when the next experiment is judged by one
property. Many materials problems require tradeoffs. A high-performing material
may be unattractive if it has high cost, toxicity, instability, synthesis risk,
or an unfavorable secondary property.

Multi-objective BO keeps these tradeoffs explicit. Instead of collapsing the
problem into one target too early, it ranks candidates by expected improvement
to the Pareto frontier.

## Physics Features

The notebook adds lightweight OPV physics scores to the descriptor table:

- The degeneracy score favors smaller donor-acceptor orbital-offset terms.
- The binding score favors lower exciton binding energy.

These scores are not used as hidden targets. They are candidate descriptors
that help the surrogate model organize the finite pool around domain-relevant
OPV design ideas.

## Retrospective Target Use

Because this is a historical OPV dataset, PCE and binding-energy values are
already known. The notebook treats PCE values in the candidate pool as withheld
metadata for retrospective visualization and does not use them for ranking. A real BO campaign would not include future target measurements in
candidate metadata.

## Takeaway

The example shows how `matgpr` can support materials BO workflows beyond
single-property maximization. The same pattern can be reused for practical
design problems such as maximizing performance while minimizing degradation,
toxicity, synthesis difficulty, or cost.
