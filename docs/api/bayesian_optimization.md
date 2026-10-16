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

::: matgpr.bayesian_optimization

## Explanation And What-If Tables

`build_bo_explanation_tables(...)` converts ranked candidates into
plot-ready recommendation, acquisition, constraint, and feature-space tables.
`build_bo_what_if_slice(...)` creates strict one- or two-feature views over
evaluated candidates, including predictive uncertainty and ranking components,
without fitting a model or interpolating between candidates.

::: matgpr.bo_explanations
