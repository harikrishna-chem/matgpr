# matgpr Next Steps

Use this file as the restart checklist for future work. Keep items short,
important, and actionable.

## Immediate Next Steps

- Run the OPV notebook with stronger settings: more repeats and more optimizer iterations.
- Save OPV learning-curve results to a CSV inside `examples/opv/results/`.
- Add rendered learning-curve and parity-plot images for the OPV example.
- Review OPV report numbers after the stronger run and update the report if metrics change.
- Decide whether to keep only `matgpr` imports or add a temporary migration note for old `genmatics_gpr` users.

## Documentation

- Add an API overview page for `PhysicsInformedMean`.
- Add a short installation and quick-start guide.
- Add a contribution note for adding new physics-informed examples.
- Add citation guidance for the OPV paper and for `matgpr`.

## Examples

- Add four more published-paper examples.
- For each example, include `dataset.csv`, notebook, and a short physics report.
- Avoid committing paper PDFs unless redistribution rights are clear.
- Keep each example focused on why physics-informed GPR helps.

## Modeling Roadmap

- Add multitask GPR support.
- Add other physics-informed GP mean functions and priors.
- Add Bayesian optimization workflows for selecting next experiments.
- Add benchmark utilities for repeated learning-curve comparisons.

## Future Codex Handoff

- Start by reading `docs/matgpr_log.md`.
- Then read this file for the active restart checklist.
- Check `git status -sb` before editing.
- Keep new updates short and append them to the dated log.
