# OPV Bayesian Optimization Recommendation Audit Report

## Purpose

This example demonstrates how to use `matgpr` to audit Bayesian-optimization
recommendations for a finite materials candidate pool. It uses the OPV
descriptor dataset as a retrospective example: a small subset is treated as
already measured, and the remaining rows are treated as candidate OPV
materials.

The goal is to show how a recommended experimental batch can be explained with
report-ready tables, not just ranked by an acquisition function.

## Reference Dataset

H. Sahu, W. Rao, A. Troisi, and H. Ma, "Toward Predicting Efficiency of
Organic Solar Cells via Machine Learning and Improved Descriptors," Advanced
Energy Materials, 8, 1801032, 2018. DOI:
[10.1002/aenm.201801032](https://doi.org/10.1002/aenm.201801032).

## Workflow

- Load the OPV descriptor dataset from `examples/opv/dataset.pkl`.
- Treat 32 OPV rows as measured training data.
- Treat 96 remaining OPV rows as a finite candidate pool.
- Build numeric BO features from molecular descriptors and two OPV physics
  scores:
  - frontier-orbital degeneracy score,
  - low exciton-binding-energy score.
- Standardize candidate features using only measured training rows.
- Audit the candidate pool before ranking.
- Annotate finite-pool policies:
  - band-gap and binding-energy feasibility limits,
  - descriptor-space trust region,
  - duplicate or pending-candidate status.
- Rank candidates with BoTorch upper confidence bound.
- Select a diverse eligible batch from feasible, in-region, non-duplicate
  candidates.
- Summarize the final recommendations with `summarize_bo_recommendation_audit`.

## Why Annotate Before Selecting

The notebook intentionally annotates policies before selecting the final
batch. This keeps the full decision trail visible:

- how many candidates violated feasibility limits,
- how many candidates fell outside the trust region,
- whether any candidates were already pending or duplicates,
- which candidates were finally selected into the batch.

After annotation, the recommended batch is selected only from candidates that
are feasible, inside the trust region, and not duplicates.

## Audit Tables

`summarize_bo_recommendation_audit` returns four tables:

- `overview_frame()`: one-row summary of input, ranked, and recommended
  candidate counts plus acquisition and uncertainty summaries.
- `policy_summary_frame()`: feasibility, trust-region, duplicate, and
  batch-selection pass/fail counts.
- `score_summary_frame()`: recommended-versus-ranked ranges for acquisition,
  predicted mean, predicted standard deviation, and batch scores.
- `recommendation_frame()`: per-recommendation audit notes that combine rank,
  acquisition, predicted value, uncertainty, and policy status.

## Retrospective Target Use

Because this is a historical OPV dataset, candidate PCE values are already
known. The notebook renames them to `withheld_pce_for_retrospective` and uses
them only for visual inspection after ranking. A real closed-loop campaign
would not include target values in candidate metadata.

## Takeaway

The audit workflow makes BO recommendations more transparent for materials
informatics users. It helps separate model-driven reasons, such as acquisition
score and uncertainty, from practical experimental rules, such as feasibility,
trust-region, duplicate, and batch-diversity decisions.
