# Testing Baseline: 2026-10-26

This report freezes the public-package quality baseline before TDD-driven
hardening or refactoring. It describes the current repository without changing
production behavior.

## Baseline Identity

- Commit: `5743e871f86c03967c08c6e9b1dd9c8d49cbd1f6`
- Audit branch: `agent/matgpr/tdd-baseline`
- Package version: `0.2.0`
- Python: CPython 3.11.15 on macOS arm64
- Machine-readable record:
  `docs/audits/testing_baseline_2026-10-26.json`

The CI matrix already covers Python 3.10, 3.11, and 3.12. This local baseline
uses Python 3.11 because it is the shared development and notebook environment.

## Gate Results

| Gate | Result |
| --- | --- |
| Tests | 414 passed; 0 failed; 0 skipped |
| Test runtime | 20.58 seconds with branch coverage |
| Warnings | 9 |
| Ruff lint | Passed |
| Ruff format check | Passed; 86 files unchanged |
| Source distribution and wheel | Passed; `matgpr-0.2.0` |
| Reduced notebook smokes | 5 passed |

The first local notebook attempt could not open Jupyter kernel ports inside the
filesystem/process sandbox. The same command passed with normal local process
permissions; this was an execution-environment restriction, not a package
failure.

## Reproduction Commands

```bash
python -m pytest -ra --durations=20 \
  --junitxml=/tmp/matgpr-tdd-junit.xml \
  --cov=matgpr --cov-branch \
  --cov-report=term \
  --cov-report=json:/tmp/matgpr-tdd-coverage.json
python -m ruff check matgpr tests scripts
python -m ruff format --check matgpr tests scripts
python -m build
MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/cache \
  python scripts/smoke_notebooks.py
```

## Coverage

| Measure | Result |
| --- | ---: |
| Combined line and branch coverage | 79.399% |
| Statement coverage | 84.254% |
| Branch coverage | 65.989% |
| Statements | 12,092 |
| Missing statements | 1,904 |
| Branches | 4,378 |
| Missing branches | 1,489 |
| Partial branches | 1,147 |

The combined percentage is useful as a gate, but scientific risk and branch
coverage should determine test priority. A small low-coverage helper and a
large moderately covered optimizer do not carry the same risk.

## Test Ownership Map

| Module or workflow | Coverage | Current direct owner | Baseline assessment |
| --- | ---: | --- | --- |
| `preprocessing.py` | 22.0% | No dedicated test module | Highest direct unit-test gap |
| `data_splitting.py` | 50.0% | No dedicated test module | Needs leakage, shape, and invalid-input tests |
| `mcp_smoke.py` | 43.7% | `test_mcp_smoke.py` | Core path covered; CLI/error branches remain |
| `visualization.py` | 66.9% | `test_visualization.py` | Needs edge, invalid-input, and optional-axes branches |
| `bayesian_optimization.py` | 73.0% | `test_bayesian_optimization.py` | Broad tests exist; 299 branches remain uncovered |
| `validation.py` | 73.7% | `test_validation.py` | Broad API coverage; split/error branches need expansion |
| `data_cleaning.py` | 73.6% | `test_data_cleaning.py` | Missing malformed and mixed-type cases |
| `sklearn_gpr.py` | 75.0% | Mostly estimator tests | Add direct wrapper contract coverage |
| `_validation.py` | 76.5% | Indirect across suite | Consolidate edge cases in focused unit tests |
| `safe_equations.py` | 82.4% | `test_safe_equations.py` | Strong base; security and numerical-limit branches remain |
| GPyTorch estimators | 79-89% | Dedicated estimator/model tests | Add scientific reference and convergence assertions |
| Multi-fidelity/multitask | 79-85% | Dedicated workflow tests | Add leakage and uncertainty reference checks |

The full per-module data is in the JSON record. Existing tests are primarily
organized by source module, but no explicit unit/contract/integration/scientific
marker taxonomy is registered yet.

## Public Contract Snapshot

- Package-level exports: 293
- Ordered export-name SHA-256:
  `b64f5af236df40169a73e93bb6031e3f7afb85ea866c8531746edd7c8eb46204`
- `tests/test_public_api.py` protects lazy imports and selected public behavior.
- The JSON baseline records the complete ordered export list.

This fingerprint detects accidental removals or reordering during internal
module splits. It is not permission to add exports without API review.

## Warning Inventory

| Warning | Count | Disposition |
| --- | ---: | --- |
| PyTorch `torch.jit.script` deprecation | 2 | Upstream; track and narrowly filter only if stable |
| BoTorch fused-extension fallback | 1 | Local cache/compiler environment; Python fallback worked |
| Linear-operator Cholesky jitter | 1 | Numerical path; retain visibility in scientific tests |
| GPyTorch prediction on training inputs | 5 | Expected in current train-prediction tests; characterize explicitly |

No warning should be globally hidden. Step 2 should document expected upstream
warnings and turn unexpected package warnings into failures.

## Duplicate-Code Classification

The audit compares function signature AST plus body AST. It found 11 strictly
identical groups containing 26 functions.

| Classification | Groups | Functions | Examples |
| --- | ---: | ---: | --- |
| Extraction candidate | 4 | 9 | one-dimensional numeric coercion, numeric matrices, slugging, dataframe validation |
| Contract-sensitive | 3 | 8 | validation-result prediction properties, target inverse prediction |
| Intentionally local | 4 | 9 | domain `n_features` properties, sklearn feature-name/kernel protocol methods |

Only the first category enters the initial cleanup queue. Contract-sensitive
groups need characterization tests first. Small protocol methods should remain
local unless sharing improves clarity.

## CI Assessment

Existing strengths:

- Python 3.10-3.12 test matrix;
- minimum-supported-dependency job;
- notebook smoke job;
- Ruff, tests, coverage reporting, and package build.

Step 2 gaps:

- coverage is reported but not enforced;
- CI does not currently request branch coverage;
- no explicit test-layer markers are registered;
- source tests run from an editable install, while the built wheel is not
  installed and imported in a clean smoke environment;
- expected versus unexpected warning policy is not encoded.

## Step 1 Conclusions

1. No baseline correctness failure was found by the existing suite.
2. Coverage is healthy overall but uneven around preprocessing, splitting,
   validation branches, visualization, and BO policy branches.
3. The released public surface is large, so API fingerprints and focused
   characterization tests must precede module splitting.
4. Public duplicate code is limited; aggressive deduplication would create more
   risk than value.
5. Step 2 should establish markers, branch-aware coverage gates, warning policy,
   reusable test fixtures, and a built-wheel smoke test before scientific or
   refactoring work begins.
