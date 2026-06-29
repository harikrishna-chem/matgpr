# Bayesian Optimization

The Bayesian-optimization API ranks finite candidate pools for next-experiment
selection. It also supports dataframe-level feasibility constraints for
synthesizability, processing windows, safety filters, equipment limits, and
diversity-aware experimental batch selection. Known observation noise can be
passed directly or prepared from reported uncertainty columns and replicate
measurements before using noisy expected-improvement acquisition functions.
Use the experiment-logging API to record recommendations, selected batches,
measured outcomes, and compact campaign summaries across closed-loop
iterations.
BoTorch is optional; install `matgpr[bo]` before fitting BoTorch surrogates.

::: matgpr.bayesian_optimization
