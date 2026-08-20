# Meta-Comments Hardening Plan

Date: 2026-08-19

## Scope

Review `/Users/hari/Downloads/Meta-Comments.md` and apply useful, low-risk
hardening changes to public `matgpr`.

## Ownership Boundary

- Public `matgpr` owns reusable package robustness, validation, sklearn
  compatibility, and documentation.
- Private `genmatics-matgpr` is not changed by this pass.
- Untracked BO notebooks are not part of this task.

## Ordered Steps

1. Harden regression metrics shape validation, constant-target handling, and
   strict JSON safety.
2. Replace distance-kernel broadcasting with SciPy pairwise distance routines
   and expose safer kernel-builder bounds/defaults.
3. Improve sklearn GPR kernel factory normalization, defaults, and grid-search
   safety.
4. Add non-breaking data-cleaning improvements for vectorized placeholder
   replacement, reproducible dropped-column/imputer reporting, and IQR policy.
5. Add low-risk featurizer sklearn metadata and feature-name validation.
6. Add small GPyTorch guards for tensor conversion, feature-index validation,
   stable positive-parameter initialization, and logging with custom kernels.
7. Add focused tests for the changed behavior.

## Acceptance Checks

- `python -m ruff check matgpr tests`
- `python -m ruff format --check` on the files touched by this pass
- `python -m pytest`
- `python -m build`

## Deferred Items

- Changing featurizer `transform` to be fully stateless is behavior-changing
  because current public tests and examples expose last-transform diagnostics.
- Changing featurizer default output from pandas to NumPy is also
  behavior-changing and should be handled in a dedicated API-transition pass.
- Full GPyTorch positive-constraint refactor can be handled later if needed.
