# Bayesian Optimization

The Bayesian-optimization API ranks finite candidate pools for next-experiment
selection. It also supports dataframe-level feasibility constraints for
synthesizability, processing windows, safety filters, equipment limits, and
diversity-aware experimental batch selection.
BoTorch is optional; install `matgpr[bo]` before fitting BoTorch surrogates.

::: matgpr.bayesian_optimization
