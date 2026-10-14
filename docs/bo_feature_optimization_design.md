# Bayesian Optimization Feature-Optimization Design

Date: 2026-10-14

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

## Explanation Payloads for Apps

Future BO result payloads should make recommendations explainable. For each
recommended point or candidate, include:

- Predicted mean and uncertainty.
- Acquisition score and acquisition family.
- Feasibility probability or constraint status.
- Applicability-domain status relative to training data.
- Duplicate or already-tested status.
- Rank and batch order.

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
3. Add constrained single-objective acquisition wrappers.
4. Add target-value and target-range acquisition utilities.
5. Add app-neutral recommendation explanation payloads.
6. Add examples for process development and additive manufacturing.
