# Bayesian Optimization Feature-Optimization Design

Date: 2026-10-14
Last updated: 2026-10-17

## Goal

Extend `matgpr` beyond finite-pool candidate ranking so users can ask:

- Which existing candidates should I test next?
- Which controllable input variables should I change to reach a target output?
- Which process window satisfies lower/upper output limits?
- Which next batch balances predicted performance, uncertainty, feasibility, and
  experimental cost?

This is especially important for process development and additive
manufacturing, where users often optimize laser power, scan speed, hatch
spacing, temperature, composition, or solvent/process conditions under physical
and equipment constraints.

## Package Boundary

`matgpr` should own reusable scientific BO building blocks:

- Search-space specifications for continuous, integer, categorical, and fixed
  variables.
- Objective specifications for maximize, minimize, target-value, and
  target-range workflows.
- Output constraints such as density above a threshold or porosity below a
  threshold.
- Acquisition-function intent recommendations.
- Future app-neutral optimization results, ask/tell loops, duplicate handling,
  and plot-ready audit payloads.

Application-specific policy belongs outside `matgpr`. A portal can decide which
widgets to show, where artifacts are saved, and how users approve experiments,
but it should not duplicate BO semantics.

## Supported User Intents

| Intent | Example | Acquisition families |
| --- | --- | --- |
| Rank finite candidates | Select 5 polymers from a generated pool | EI, NEI, UCB, PI, Thompson sampling |
| Maximize or minimize | Maximize conductivity, minimize diffusivity | EI/NEI, UCB/LCB |
| Hit a target | Find modulus near 2 GPa | target probability, target desirability |
| Satisfy output limits | Density > 99%, porosity < 1% | constrained EI/NEI, feasibility-weighted scores |
| Batch selection | Recommend 8 next experiments | qEI/qNEI/qUCB, Thompson sampling, diversity penalties |
| Multi-objective | Maximize strength and minimize roughness | scalarization first, EHVI/qEHVI later |
| Cost-aware BO | Prefer cheaper experiments when performance is similar | acquisition per cost |

## First Implementation Step

The first code step adds typed, JSON-friendly specification records in
`matgpr.bayesian_optimization_specs`:

- `BOFeatureSpec`
- `BOObjectivePreference`
- `BOOutputConstraintSpec`
- `BOOptimizationSpec`
- `recommend_bo_acquisition_intents`

These records do not fit a model. They define the contract that later optimizer
implementations can consume.

The second code step adds `build_bo_candidate_pool(...)` in
`matgpr.candidate_generation`. It converts `BOOptimizationSpec.features` into a
finite candidate table using either:

- `strategy="grid"` for small, interpretable process windows.
- `strategy="random"` for high-dimensional continuous, integer, and categorical
  spaces.

The third code step adds `rank_bo_candidate_pool(...)` in
`matgpr.bayesian_optimization`. It scores generated candidate pools for one
modeled target using maximize/minimize, target-value, or target-range objective
preferences. Output constraints on the same target are converted into
feasibility probabilities and multiplied into the acquisition score.
Target-value requests can provide the typed `target_tolerance` field in target
units; legacy metadata tolerance keys remain accepted for compatibility.

The fourth code step adds `rank_bo_candidate_predictions(...)`. It applies the
same objective and feasibility semantics to saved prediction rows with
`predicted_mean` and optional `predicted_std`, without refitting or calling a
surrogate. This is the reusable public utility for apps that load a persisted
model artifact, run predictions once, and then rank candidate recommendations
from those saved predictions. Its explicit saved-prediction acquisitions are:

- posterior-mean preference for deterministic ranking;
- UCB for maximization and LCB for minimization;
- expected improvement and probability of improvement with a compatible
  incumbent;
- target-value or target-range probability; and
- feasibility-weighted forms for same-target output constraints.

The saved-prediction path rejects NEI, Thompson sampling, cost-aware
acquisition, EHVI/qEHVI, and true joint-batch semantics because candidate means
and standard deviations alone cannot reproduce those algorithms. Same-target
probability events are intersected before Gaussian probability is evaluated;
they are not multiplied as if statistically independent. Signed deterministic,
UCB, and LCB utilities are converted to a nonnegative finite-pool component
before feasibility weighting so infeasible negative-utility points are not
accidentally favored.

The fifth code step adds `build_bo_explanation_tables(...)`. It converts a
ranked candidate table into four app-neutral DataFrames:

- Recommendation summary rows for the selected candidates.
- Acquisition decomposition rows showing the objective component,
  feasibility probability, and final acquisition score.
- Constraint rows with bounds, probability thresholds, and predicted
  feasibility for every candidate/constraint pair.
- Feature-space rows containing reference, candidate, and recommended points.

Feature-space rows use the original axis for one feature, the two original
axes for two features, and a standardized two-component PCA projection for
higher-dimensional numeric spaces. The package returns coordinates and
embedding metadata; applications remain responsible only for rendering.

The sixth code step adds `build_bo_what_if_slice(...)`. It constructs a strict
one- or two-feature view from an already ranked candidate table and returns:

- original feature-axis coordinates,
- predicted mean and standard deviation,
- configurable uncertainty bounds,
- objective, feasibility, and acquisition components,
- recommendation role, rank, and stable candidate/input ID,
- fixed and constant context values plus explicit
  `interpolation_method="none"` provenance.

The builder does not refit or call a model and does not interpolate between
candidates. If hidden varying features leave more than one candidate at an
axis coordinate, it rejects the slice until the caller supplies additional
fixed feature values.

## Explanation Payloads for Apps

Future BO result payloads should make recommendations explainable. For each
recommended point or candidate, include:

- Predicted mean and uncertainty.
- Acquisition score and acquisition family.
- Feasibility probability or constraint status.
- Applicability-domain status relative to training data.
- Duplicate or already-tested status.
- Rank and batch order.

The public explanation builders preserve package annotations for acquisition
rank, eligible rank, final batch order, duplicate matching, exclusion reasons,
and nested applicability records. What-if slices use the final package batch
annotation when it is present rather than inferring recommendations from a rank
cutoff.

Plot-ready payloads should support:

- Training data, candidate pool, and recommended points in a 2D embedding or
  selected feature plane.
- Acquisition score vs predicted mean/uncertainty.
- Feasibility and output-limit views.
- One-dimensional or two-dimensional what-if slices for controllable variables.

The app can render these plots, but the package should own the payload
semantics.

## Next Steps

1. Add optimizer-neutral result records for feature-optimization BO.
2. Add continuous/mixed search-space candidate generation from `BOFeatureSpec`.
3. Add constrained single-objective acquisition wrappers: first single-target
   preference scorer is implemented.
4. Add target-value and target-range acquisition utilities: first single-target
   scorer is implemented for fitted models and saved predictions.
5. Add app-neutral recommendation explanation payloads: done for single-target
   feature-optimization ranking.
6. Add strict one/two-variable what-if slices over evaluated candidates: done.
7. Add examples for process development and additive manufacturing: private
   GenMatics smoke example done; a public example can follow in a later release.
8. Saved-prediction acquisition, duplicate-ID audit, explanation propagation,
   and post-ranking batch-selection primitives: done.
