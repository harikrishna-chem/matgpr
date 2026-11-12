# Public Scheduled Scientific-Convergence CI

**Date:** 2026-11-06
**Repository:** `matgpr`
**Scope:** Step 10, Slice 5

## Purpose

Give the existing long-budget PI-GPR scientific-convergence pilot explicit CI
ownership without adding optimizer work to every pull request. This check
protects a known thermal-expansion relation, learned and fixed physics
parameters, extrapolation accuracy, and uncertainty behavior across training
budgets through 1,600 iterations.

## Workflow Contract

The `Scientific convergence` workflow:

- runs every Monday at 17:23 UTC from the default branch;
- supports manual `workflow_dispatch` execution;
- never runs automatically for pushes or pull requests;
- uses Python 3.11 and the normal package runtime;
- sets Python hash and numerical-library thread counts deterministically;
- runs with read-only repository permissions;
- permits only one convergence workflow run at a time without cancelling an
  active scientific check; and
- applies a 15-minute timeout to the convergence job.

The script emits JSON evidence directly to the workflow log. Runtime is
reported for diagnosis but is not an assertion because hosted-runner timing is
not a stable scientific contract.

## Scientific Contract

- Training temperatures: 475-625 K.
- Extrapolation temperatures: 250-850 K.
- Learned parameter: thermal slope in angstrom per 100 K.
- Fixed parameter: 3.6-angstrom reference lattice parameter.
- Optimizer budgets: 10, 25, 50, 100, 200, 400, 800, and 1,600 iterations.
- Final slope absolute error must be below `1e-5` angstrom per 100 K.
- Final extrapolation RMSE must be below `1e-5` angstrom.
- Final 95% interval coverage must be at least 0.95.
- Intermediate RMSE is not required to improve monotonically.

## Local Verification

The exact workflow environment and command passed locally:

- final slope absolute error: `2.0625e-7` angstrom per 100 K;
- final extrapolation RMSE: `3.6006e-7` angstrom;
- final mean latent standard deviation: `1.5639e-5` angstrom;
- final 95% interval coverage: `1.0`; and
- fixed reference lattice parameter: exactly `3.6` angstrom.

The command completed in approximately 3.7 seconds including process and
library startup. This observation is not a gate.

## Decision

The PI-GPR convergence pilot is now a scheduled and manually dispatchable
scientific CI check. Ordinary pull-request CI remains unchanged. Mutation
testing remains manual because its cost and interpretation do not suit an
automatic blocking workflow.
