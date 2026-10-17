# Bayesian Optimization

The Bayesian-optimization API ranks finite candidate pools for next-experiment
selection. It also supports dataframe-level feasibility constraints for
synthesizability, processing windows, safety filters, equipment limits, and
diversity-aware experimental batch selection. Candidate trust-region policies
keep recommendations near known feasible descriptor space, while duplicate
policies remove or annotate materials that were already measured, selected, or
queued. Known observation noise can be passed directly or prepared from
reported uncertainty columns and replicate measurements before using noisy
expected-improvement acquisition functions.
Recommendation audit summaries convert ranked BO tables into compact
report-ready explanations of acquisition scores, posterior uncertainty,
feasibility, trust-region status, duplicate filtering, and batch-selection
decisions.
For multi-objective campaigns, independent BoTorch GP surrogates can be ranked
with expected hypervolume improvement while keeping objective directions in
original materials-property units. Greedy sequential batch selection can
recompute hypervolume improvement after each selected candidate is treated as a
pending experiment.
Use the experiment-logging API to record recommendations, selected batches,
measured outcomes, and compact campaign summaries across closed-loop
iterations.
BoTorch is a required dependency, so these helpers work in any `matgpr`
installation. It is imported lazily, only when a BoTorch-backed helper runs.

## Saved-Prediction Scoring

`rank_bo_candidate_predictions(...)` ranks a finite candidate table when a
saved estimator has already produced posterior means and standard deviations.
Set `acquisition_intent` explicitly for production workflows. Supported intents
are posterior-mean preference, UCB, LCB, EI, PI, and target probability. EI and
PI require an incumbent in the same target units and output space. All
uncertainty-aware intents require `predicted_std`; deterministic fallback is
available only by explicitly requesting `posterior_mean_preference`.

The result records the requested and effective acquisition, raw acquisition
base, feasibility-weighted component, uncertainty-use flag, and acquisition
rank. Same-target objective/constraint probabilities use their intersected
event. Sampling-based, noisy, cost-aware, multi-objective, and true joint-batch
acquisitions require richer model state and are rejected by this utility.

`CandidateDuplicatePolicy(existing_id_column=...)` preserves the matching
observed candidate ID when duplicate rows are annotated. `select_diverse_batch`
supports standardized post-ranking diversity over declared nonconstant numeric
features.

::: matgpr.bayesian_optimization

## Explanation And What-If Tables

`build_bo_explanation_tables(...)` converts ranked candidates into
plot-ready recommendation, acquisition, constraint, and feature-space tables.
`build_bo_what_if_slice(...)` creates strict one- or two-feature views over
evaluated candidates, including predictive uncertainty and ranking components,
without fitting a model or interpolating between candidates. These tables
preserve acquisition/eligibility/batch ranks, duplicate audit fields, exclusion
reasons, and nested applicability metadata when present.

::: matgpr.bo_explanations
