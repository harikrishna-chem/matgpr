# What PI-GPR Does And Does Not Guarantee

Physics-informed Gaussian Process Regression can make a model more useful and
more interpretable, especially in low-data materials problems. It is still a
statistical model. A physics term in the mean function is an inductive bias,
not proof that the model is physically correct.

## What PI-GPR Does

In `matgpr`, a physics-informed model has the form:

$$
y(\mathbf{x}) = m_{physics}(\mathbf{x}_{phys}) + f_{residual}(\mathbf{x}) + \epsilon
$$

where:

- \(m_{physics}\) is a user-defined or template-based physics mean function,
- \(\mathbf{x}_{phys}\) are the features used by the physics equation,
- \(f_{residual}\) is the GP residual learned from data,
- \(\epsilon\) is observation noise.

This setup does provide a few clear guarantees about the model definition:

- The physics equation enters the GP mean function, not just a postprocessing
  plot.
- Learnable physics parameters are optimized during GP training together with
  kernel and likelihood parameters.
- The GP residual remains available to model mismatch between the physics trend
  and the observations.
- Predictive uncertainty is still computed from the fitted GP model.
- The physics equation, features, learned parameters, and validation protocol
  can be reported explicitly.

## What PI-GPR Does Not Guarantee

PI-GPR does not automatically guarantee:

- lower RMSE than standard GPR,
- better extrapolation outside the training domain,
- physically valid predictions for every candidate,
- calibrated uncertainty intervals,
- causal interpretation of learned parameters,
- correct physics if the equation is misspecified,
- correct units if feature scaling and unit handling are wrong,
- monotonicity, positivity, boundedness, or conservation laws unless those
  constraints are encoded separately,
- automatic discovery of the right physics equation from data.

A physics-informed mean can hurt performance when the equation has the wrong
trend, uses weak features, leaks information from test data, or is too complex
for the available dataset.

## What To Validate

Always compare PI-GPR against a standard GPR baseline using the same data split,
features, kernel class, optimizer settings, and evaluation metrics. For a
publishable result, report:

- repeated learning curves,
- held-out test metrics,
- parity plots with uncertainty bars,
- uncertainty diagnostics such as interval coverage and NLPD,
- learned physics parameters,
- the exact features used by the physics equation,
- the applicability domain of the dataset.

The strongest evidence for PI-GPR is not that it contains an equation. The
strongest evidence is that the equation improves or stabilizes validation
performance in the intended low-data regime while remaining physically
interpretable.

## When Extra Constraints Are Needed

Use additional `matgpr` tools when the property has known limits:

- Use target transforms for positive or bounded targets.
- Use virtual observations for soft known-limit or monotonicity anchors.
- Use derivative-constrained GPR when trusted slope information is available.
- Use physics-aware or learned heteroscedastic noise when uncertainty changes
  across the materials space.

These tools add more structure than a mean function alone. They still require
validation and should be reported separately from the PI-GPR mean equation.
