# OPV Closed-Loop Bayesian Optimization Campaign Report

## Purpose

This example demonstrates a restartable ask-tell Bayesian-optimization
campaign using `matgpr` closed-loop logging utilities. It uses the OPV dataset
retrospectively so the notebook can simulate recommendations, selections,
measured outcomes, campaign restart, and the next ask step.

The companion notebook
`opv_bo_recommendation_audit.ipynb` focuses on why candidates were
recommended. This notebook focuses on what happens after recommendations are
made and how the campaign can be resumed safely.

## Reference Dataset

H. Sahu, W. Rao, A. Troisi, and H. Ma, "Toward Predicting Efficiency of
Organic Solar Cells via Machine Learning and Improved Descriptors," Advanced
Energy Materials, 8, 1801032, 2018. DOI:
[10.1002/aenm.201801032](https://doi.org/10.1002/aenm.201801032).

## Workflow

- Treat a subset of OPV rows as initially measured experiments.
- Treat the remaining rows as a finite candidate pool.
- Start with an empty temporary campaign log.
- Use `resume_bo_campaign` to recover the next iteration and available
  candidates.
- Run a BO ask step using BoTorch when installed, or a deterministic fallback
  ranking when BoTorch is unavailable.
- Log recommendations with `log_bo_recommendations`.
- Log the selected experimental batch with `log_selected_experiments`.
- Reveal withheld OPV PCE values as simulated observations and log them with
  `log_observations`.
- Resume the campaign and verify that completed candidates are unavailable for
  the next ask step.
- Repeat a second ask-tell iteration.
- Summarize and visualize the campaign log with `summarize_closed_loop_log`
  and `plot_bo_campaign_progress`.

## Why This Matters

Materials BO campaigns often span days or weeks. The model may recommend a
batch, only some experiments may be selected, and measurements may arrive
later. Without durable logging, it becomes easy to lose track of:

- what the model recommended,
- what was actually selected,
- what has already been measured,
- what is still pending,
- which candidates should be excluded from future recommendations.

The restart state returned by `resume_bo_campaign` solves this bookkeeping
problem for finite-pool BO workflows.

## Log Record Types

The notebook writes three main record types:

- `recommendation`: candidate rows returned by the BO ask step.
- `selection`: candidates actually selected for experimental execution.
- `observation`: measured outcomes returned from the lab or simulation.

Each row receives campaign metadata columns such as campaign ID, iteration,
record type, timestamp, model name, acquisition function, and selection policy.

## Retrospective Target Use

Because the OPV dataset is historical, target PCE values are known. The
notebook hides those values in candidate metadata until a simulated experiment
is selected, then reveals them as observations. A real closed-loop campaign
would replace this reveal step with new laboratory or simulation results.

## Takeaway

Closed-loop logging turns BO from a one-shot ranking script into a restartable
campaign. This pattern is important for future GenMatics workflows where
recommendations, selected experiments, observations, and candidate availability
will need to move through a portal, database, and cloud execution backend.
