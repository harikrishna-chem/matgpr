# API Reference

The API reference is generated from the package docstrings with `mkdocstrings`.
Use this section when you need exact function signatures, parameters, return
types, and class methods.

## Public Workflow Modules

- [Estimators](estimators.md): scikit-learn-style single-task,
  physics-informed, and multitask GPR estimator classes.
- [Featurizers](featurizers.md): scikit-learn-style materials transformers.
- [Kernels](kernels.md): Tanimoto, element-fraction, structure-feature, and
  feature-subset kernel helpers.
- [Target Transforms](target_transforms.md): positive, bounded, standardized,
  physics-residual, and materials-property preset helpers.
- [Physics Constraints](physics_constraints.md): soft virtual observations for
  known limits and monotonic trends.
- [Derivative-Constrained GPR](derivative_gpr.md): exact RBF GPR with
  function-value and derivative observations.
- [Noise Models](noise_models.md): source-dependent, replicate-aware, and
  feature-dependent observation-noise profiles.
- [Heteroscedastic GPR](heteroscedastic_gpr.md): learned input-dependent
  observation noise with a residual-noise GP.
- [Multitask GPR](multitask_gpr.md): correlated multi-output GPR for complete
  materials-property target matrices.
- [Physics Equation Templates](physics_equations.md): reusable Arrhenius,
  power-law, Hall-Petch, free-volume, and mixture mean equations.
- [Physics-Informed GPR](gpytorch_gpr.md): GPyTorch training, prediction, and
  mean-function utilities.
- [Validation](validation.md): train/test evaluation, cross-validation, and
  configurable learning-curve utilities.
- [Candidate Generation](candidate_generation.md): finite chemistry,
  composition, formulation, and processing-condition candidate pools.
- [Bayesian Optimization](bayesian_optimization.md): optional BoTorch
  finite-pool candidate ranking for next-experiment selection.
- [BO Benchmarking](bo_benchmarking.md): offline finite-pool strategy replay
  against known outcomes.
- [Multi-Objective Selection](multi_objective.md): Pareto-front and weighted
  finite-pool ranking helpers.
- [Experiment Logging](experiment_logging.md): closed-loop recommendation,
  selection, observation, and campaign-summary logs.
- [Uncertainty](uncertainty.md): coverage, calibration, NLPD, and uncertainty
  diagnostics.
- [Fingerprints](fingerprints.md): lower-level composition, SMILES, polymer, and
  cache helpers.
- [Data And Metrics](data.md): cleaning, splitting, preprocessing, metrics, PCA,
  reporting, artifact utilities, and optional-dependency helpers.
- [Visualization](visualization.md): plotting helpers for model analysis.

Prefer the high-level estimator and featurizer APIs for reusable scripts.
Use the lower-level functions when building custom research workflows.
