# Public PI-GPR Scientific-Convergence Pilot

**Date:** 2026-11-05
**Plan step:** Step 9, Slice 5
**Status:** Completed

## Scope

Run a manual, fixed-seed physics-informed GPR convergence study for a
synthetic linear thermal-expansion relation:

```text
a(T) = a_ref + slope * ((T - 550 K) / 100 K)
```

The target lattice parameter `a` is measured in angstrom. The learned slope is
measured in angstrom per 100 K, and the fixed reference lattice parameter is
measured in angstrom.

Five noiseless training temperatures span 475-625 K. The dense evaluation grid
spans 250-850 K, so most evaluation points test extrapolation beyond the
training range. The RBF GP models any residual left by the physics mean.

## Learned and Fixed Quantities

- Learned on the fly: thermal slope, initialized at 0.02 angstrom per 100 K;
  true value 0.04 angstrom per 100 K.
- Fixed: reference lattice parameter, 3.6 angstrom at 550 K.
- GP-learned: RBF residual-kernel and Gaussian-likelihood parameters.
- Fixed optimization policy: learning rate 0.03, initial noise 0.01,
  standardized target, float64 CPU execution, and seed 2026.

## Budgets and Stable Contracts

The manual script fits independent fixed-seed models with 10, 25, 50, 100,
200, 400, 800, and 1,600 training iterations. It requires:

- one recorded loss per optimizer iteration;
- identical initial loss across fixed-seed runs;
- finite metrics and positive latent uncertainty at every budget;
- exact preservation of the fixed reference lattice parameter;
- material endpoint loss reduction;
- final slope error below `1e-5` angstrom per 100 K;
- final extrapolation RMSE below `1e-5` angstrom and at least two orders of
  magnitude below the 10-iteration result;
- contracting endpoint latent uncertainty; and
- at least 95% final interval coverage.

Intermediate RMSE is intentionally not required to improve monotonically.
Adam optimization can improve marginal likelihood while temporarily moving a
physical parameter or predictive error away from its best intermediate value.

## Results

From 10 to 1,600 iterations:

- final marginal loss changed from `-0.0467835` to `-3.6817288`;
- learned slope changed from `0.0269640` to `0.040000206` angstrom per 100 K;
- slope absolute error fell from `1.3036e-2` to `2.0625e-7` angstrom per
  100 K;
- extrapolation RMSE fell from `1.8932e-2` to `3.6006e-7` angstrom;
- mean latent standard deviation contracted from `8.6537e-3` to `1.5639e-5`
  angstrom;
- 95% interval coverage increased from 0.835 to 1.0; and
- the fixed 3.6-angstrom reference parameter remained exact.

The complete command took approximately 4.6 seconds including Python and
library startup; recorded model-fit time totaled about 2.1 seconds. Runtime is
audit evidence, not an assertion.

## Boundary

This is a noiseless synthetic case whose physics equation exactly matches the
data-generating relation. It demonstrates optimization convergence and unit
handling, not a universal PI-GPR advantage or expected real-data error. The
script makes no claim that every intermediate budget should outperform the
previous budget.

## Quality Gates

- `python scripts/check_pi_gpr_convergence.py` passes and emits JSON-safe
  records.
- The complete ordinary public suite passes 874 tests with 9 expected
  third-party warnings and 83.81% combined branch-aware coverage.
- Ruff lint and format, strict MkDocs, sdist/wheel builds, Twine checks, and
  diff hygiene pass.
- No production code, public export, function signature, dependency, or
  runtime behavior changed.

## Decision

The manual convergence check provides useful longer-budget scientific evidence
without increasing ordinary pull-request runtime. It remains a source-checkout
contributor command until Step 10 defines whether manual scientific checks
should receive scheduled CI ownership.

All five Step 9 pilots are complete. Step 10 begins with a public CI-tier audit
before changing workflow commands or coverage thresholds.

## Step 10 Follow-Up

Step 10, Slice 5 added `.github/workflows/scientific-convergence.yml`. The
pilot now runs weekly and by manual dispatch while remaining outside ordinary
push and pull-request CI. Its scientific assertions are unchanged.
