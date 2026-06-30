# Experiment Logging

Closed-loop experiment logging helpers record Bayesian-optimization
recommendations, selected experiments, measured outcomes, and compact campaign
summaries. The functions are dataframe-based so they can be used with BoTorch
recommendations, manually selected batches, or external laboratory results.
Restart helpers infer the next BO ask iteration, recover pending selections,
separate completed observations, and filter finite candidate pools before a
new session starts.

::: matgpr.experiment_logging
