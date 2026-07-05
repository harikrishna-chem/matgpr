# Co-Kriging And Multi-Level Fidelity Design

This page records the design target for the next generation of `matgpr`
multi-fidelity Gaussian-process models. It is a design document, not a claim
that every model below is implemented today. The current implemented model is
the two-stage delta model documented in [Multi-Fidelity GPR](multifidelity_gpr.md).

## Goals

The design should support materials workflows where inexpensive sources are
used to improve predictions for scarce high-fidelity measurements:

- simulations plus experiments,
- low-accuracy plus high-accuracy calculations,
- screening measurements plus validated measurements,
- coarse processing-property estimates plus careful laboratory measurements,
- more than two ordered fidelity levels.

The user-facing API should make it clear which fidelity is the decision target,
which data source each observation comes from, how noise is handled, and how
uncertainty propagates from lower fidelities into higher-fidelity predictions.

## Model Family

### Current Delta Model

The implemented delta model uses:

$$
y_H(\mathbf{x}) = \rho y_L(\mathbf{x}) + b + \delta(\mathbf{x}) + \epsilon_H
$$

where \(y_L(\mathbf{x})\) is either supplied externally or predicted by an
internal low-fidelity surrogate. This is simple, stable for small datasets, and
easy to explain. It should remain the recommended first model.

### Two-Level Co-Kriging

The next target is a joint autoregressive co-kriging model:

$$
f_L(\mathbf{x}) \sim \mathcal{GP}(m_L(\mathbf{x}), k_L(\mathbf{x}, \mathbf{x}'))
$$

$$
f_H(\mathbf{x}) = \rho f_L(\mathbf{x}) + \delta_H(\mathbf{x})
$$

$$
\delta_H(\mathbf{x}) \sim
\mathcal{GP}(m_\delta(\mathbf{x}), k_\delta(\mathbf{x}, \mathbf{x}'))
$$

with observations:

$$
y_L = f_L(\mathbf{x}) + \epsilon_L,\qquad
y_H = f_H(\mathbf{x}) + \epsilon_H
$$

The resulting covariance blocks are:

$$
\operatorname{cov}[f_L(\mathbf{x}), f_L(\mathbf{x}')] = k_L(\mathbf{x}, \mathbf{x}')
$$

$$
\operatorname{cov}[f_H(\mathbf{x}), f_L(\mathbf{x}')] =
\rho k_L(\mathbf{x}, \mathbf{x}')
$$

$$
\operatorname{cov}[f_H(\mathbf{x}), f_H(\mathbf{x}')] =
\rho^2 k_L(\mathbf{x}, \mathbf{x}') + k_\delta(\mathbf{x}, \mathbf{x}')
$$

Unlike the current two-stage delta model, this model should fit all observed
low- and high-fidelity values jointly and infer posterior uncertainty using the
full cross-fidelity covariance.

### Multi-Level Autoregressive Fidelity

For ordered fidelity levels \(l = 0, 1, \dots, L\):

$$
f_0(\mathbf{x}) \sim \mathcal{GP}(m_0(\mathbf{x}), k_0(\mathbf{x}, \mathbf{x}'))
$$

$$
f_l(\mathbf{x}) =
\rho_l f_{l-1}(\mathbf{x}) + \delta_l(\mathbf{x}), \quad l \ge 1
$$

$$
\delta_l(\mathbf{x}) \sim
\mathcal{GP}(m_l(\mathbf{x}), k_l(\mathbf{x}, \mathbf{x}'))
$$

The highest fidelity is usually the target for prediction and Bayesian
optimization. Lower fidelities should be treated as information sources, not
as interchangeable targets.

## Data Representation

The implemented data container makes fidelity explicit:

```python
from matgpr import prepare_multifidelity_observations


observations = prepare_multifidelity_observations(
    X=X_all,
    y=y_all,
    fidelity=fidelity_labels,
    fidelity_order=["simulation_low", "simulation_high", "experiment"],
    sample_id=material_ids,
    noise_variance=known_variances,
)
```

Implemented container:

```python
@dataclass(frozen=True)
class MultiFidelityObservationData:
    X: np.ndarray
    y: np.ndarray
    fidelity_index: np.ndarray
    fidelity_names: tuple[str, ...]
    target_fidelity: str
    sample_id: np.ndarray | None
    noise_variance: np.ndarray | None
    feature_names: tuple[str, ...] | None
```

Design requirements:

- allow non-nested data, where high-fidelity samples are not necessarily a
  subset of low-fidelity samples,
- preserve sample identifiers for reporting,
- support known per-observation noise later,
- validate that every fidelity level has enough observations,
- keep fidelity order explicit instead of relying on alphabetical labels.

If `fidelity_order` is omitted, `matgpr` infers the order from first
appearance in `fidelity`; explicit order is still recommended for publication
workflows because labels such as `"low"`, `"high"`, `"DFT"`, and
`"experiment"` do not encode a universal ordering.

## Proposed Public API

### Estimator API

```python
from matgpr import CoKrigingGPRRegressor


model = CoKrigingGPRRegressor(
    fidelity_order=["simulation", "experiment"],
    target_fidelity="experiment",
    base_kernel="matern",
    delta_kernel="matern",
    rho_mode="constant",
    noise_mode="learned",
    training_iter=1000,
    random_state=7,
)

model.fit(X_all, y_all, fidelity=fidelity_labels)

prediction = model.predict_distribution(
    X_candidates,
    target_fidelity="experiment",
    return_fidelity_components=True,
)
```

Expected estimator attributes:

- `fidelity_names_`
- `target_fidelity_`
- `rho_` or `rho_by_level_`
- `fidelity_observation_counts_`
- `noise_mode_`
- `loss_history_`

### Lower-Level API

```python
result = fit_cokriging_gpr(
    observations,
    target_fidelity="experiment",
    base_kernel="matern",
    delta_kernel="matern",
    training_iter=1000,
)

prediction = result.predict(
    X_test,
    target_fidelity="experiment",
    return_std=True,
    return_fidelity_components=True,
)
```

Expected prediction object:

```python
@dataclass(frozen=True)
class CoKrigingGPRPrediction:
    mean: np.ndarray
    std: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None
    target_fidelity: str
    fidelity_component_means: dict[str, np.ndarray]
    fidelity_component_stds: dict[str, np.ndarray] | None
    rho_by_level: dict[str, float]
```

## Noise Modes

The first implementation should support:

- `noise_mode="learned"`: one learned Gaussian noise term per fidelity,
- `noise_mode="known"`: fixed per-observation noise variances,
- `noise_mode="shared"`: one shared noise term across all fidelities.

Later extensions can add source-specific, replicate-aware, and feature-dependent
noise profiles by reusing the existing `matgpr.noise_models` utilities.

## Kernel Choices

Default kernels should follow the current package style:

- Matern as a conservative default for continuous descriptors,
- RBF for smooth synthetic or simulation surfaces,
- ARD support for descriptor relevance,
- feature-subset kernels later for fidelity-specific descriptor groups,
- physics-informed mean functions later for any fidelity level.

The design should not assume all fidelities use the same kernel. A practical
first implementation can share kernel type across levels, but the result object
should be able to report one kernel per latent process.

## Validation Protocol

Validation should always report high-fidelity performance separately from
lower-fidelity fit quality. Useful comparisons:

- high-fidelity-only standard GPR,
- current two-stage delta multi-fidelity GPR,
- joint two-level co-kriging,
- multi-level co-kriging when three or more ordered fidelities are available.

Recommended outputs:

- learning curves where the x-axis is high-fidelity training count or percent,
- held-out high-fidelity parity plots with uncertainty,
- per-fidelity residual summaries,
- learned `rho` values and their stability across splits,
- uncertainty coverage at the target fidelity,
- component reports showing lower-fidelity and discrepancy contributions.

The existing `multifidelity_learning_curve` and
`summarize_multifidelity_components` APIs should be extended to accept
co-kriging predictions rather than replaced.

## Implementation Milestones

1. Implemented: add `MultiFidelityObservationData` and
   `prepare_multifidelity_observations`.
2. Implemented: add data-validation tests for ordered fidelity datasets.
3. Next: add a two-level `CoKrigingGPRRegressor` with learned constant `rho`.
4. Add prediction output with target-fidelity mean, uncertainty, and component
   summaries.
5. Add validation/reporting compatibility with existing learning-curve helpers.
6. Add multi-level autoregressive support for \(L > 1\).
7. Add known-noise and per-fidelity learned-noise modes.
8. Add BO integration targeting the highest fidelity.

## Next Coding Step

The next safest coding step is a minimal two-level co-kriging model skeleton:

- accept `MultiFidelityObservationData` as the fitting input,
- support exactly two ordered fidelities first,
- learn one constant \(\rho\) between lower and target fidelity,
- learn one low-fidelity latent kernel and one discrepancy kernel,
- report per-fidelity noise assumptions clearly,
- keep validation tests small before expanding to multi-level support.

The current data layer gives this model a stable input contract and keeps the
public API clean before heavier GPyTorch model code is added.
