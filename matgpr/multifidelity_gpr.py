from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import gpytorch
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, column_or_1d

from .gpytorch_gpr import (
    GPyTorchGPRResult,
    _make_gpytorch_base_kernel,
    _should_log_iteration,
    _to_tensor,
    _validate_training_options,
    fit_gpytorch_gpr,
)

__all__ = [
    "CoKrigingGPRPrediction",
    "CoKrigingGPRRegressor",
    "CoKrigingGPRResult",
    "DeltaMultiFidelityGPRResult",
    "ExactTwoLevelCoKrigingGPRModel",
    "MultiFidelityObservationData",
    "MultiFidelityGPRPrediction",
    "MultiFidelityGPRRegressor",
    "fit_cokriging_gpr",
    "fit_delta_multifidelity_gpr",
    "prepare_multifidelity_observations",
]


@dataclass(frozen=True)
class MultiFidelityObservationData:
    """Validated observation table for ordered multi-fidelity datasets.

    Attributes
    ----------
    X, y
        Feature matrix and scalar target values for every observed row.
    fidelity_index
        Integer fidelity level for each row. The order is defined by
        ``fidelity_names``.
    fidelity_names
        Ordered fidelity labels, usually from cheapest/lowest to target/highest
        fidelity.
    target_fidelity
        Fidelity label used as the default decision target.
    sample_id
        Optional material/sample identifiers aligned with rows of ``X``.
    noise_variance
        Optional known per-observation noise variances aligned with rows of
        ``X``.
    feature_names
        Optional descriptor names aligned with columns of ``X``.
    """

    X: np.ndarray
    y: np.ndarray
    fidelity_index: np.ndarray
    fidelity_names: tuple[str, ...]
    target_fidelity: str
    sample_id: np.ndarray | None = None
    noise_variance: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None

    @property
    def n_samples(self) -> int:
        """Number of observed fidelity rows."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.X.shape[1])

    @property
    def n_fidelities(self) -> int:
        """Number of ordered fidelity levels."""
        return len(self.fidelity_names)

    @property
    def target_fidelity_index(self) -> int:
        """Integer index of the target fidelity."""
        return self.fidelity_names.index(self.target_fidelity)

    @property
    def fidelity_labels(self) -> np.ndarray:
        """Row-aligned string fidelity labels."""
        return np.asarray(
            [self.fidelity_names[index] for index in self.fidelity_index],
            dtype=object,
        )

    @property
    def fidelity_observation_counts(self) -> dict[str, int]:
        """Observation counts keyed by fidelity label."""
        return {
            name: int(np.sum(self.fidelity_index == index))
            for index, name in enumerate(self.fidelity_names)
        }

    @property
    def target_rows(self) -> np.ndarray:
        """Row indices belonging to the target fidelity."""
        return self.rows_for_fidelity(self.target_fidelity)

    def rows_for_fidelity(self, fidelity: str | int) -> np.ndarray:
        """Return row indices for a fidelity label or integer index."""
        fidelity_index = self._resolve_fidelity_index(fidelity)
        return np.flatnonzero(self.fidelity_index == fidelity_index)

    def _resolve_fidelity_index(self, fidelity: str | int) -> int:
        if isinstance(fidelity, (int, np.integer)):
            index = int(fidelity)
            if 0 <= index < self.n_fidelities:
                return index
            raise ValueError(
                f"fidelity index must be between 0 and {self.n_fidelities - 1}; got {index}"
            )

        label = str(fidelity)
        if label not in self.fidelity_names:
            raise ValueError(f"Unknown fidelity {label!r}; expected one of {self.fidelity_names}")
        return self.fidelity_names.index(label)


@dataclass(frozen=True)
class MultiFidelityGPRPrediction:
    """Prediction from a delta multi-fidelity Gaussian-process model.

    Attributes
    ----------
    mean, std, lower, upper
        High-fidelity predictive mean, standard deviation, and optional
        confidence interval in original high-fidelity target units.
    low_fidelity_mean, low_fidelity_std
        Low-fidelity values or low-fidelity surrogate predictions used in the
        autoregressive correction.
    correction_mean, correction_std
        GP correction term ``delta(x)`` and its uncertainty.
    rho, intercept
        Fitted linear mapping from low fidelity to high fidelity.
    """

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    low_fidelity_mean: np.ndarray | None = None
    low_fidelity_std: np.ndarray | None = None
    correction_mean: np.ndarray | None = None
    correction_std: np.ndarray | None = None
    rho: float = 1.0
    intercept: float = 0.0


@dataclass(frozen=True)
class CoKrigingGPRPrediction:
    """Prediction from a two-level autoregressive co-kriging model."""

    mean: np.ndarray
    std: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    low_fidelity_mean: np.ndarray | None = None
    low_fidelity_std: np.ndarray | None = None
    scaled_low_fidelity_mean: np.ndarray | None = None
    scaled_low_fidelity_std: np.ndarray | None = None
    discrepancy_mean: np.ndarray | None = None
    discrepancy_std: np.ndarray | None = None
    fidelity: str = ""
    rho: float = 1.0


@dataclass(frozen=True)
class CoKrigingGPRResult:
    """Fitted two-level autoregressive co-kriging GPR result."""

    model: "ExactTwoLevelCoKrigingGPRModel"
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    observation_data: MultiFidelityObservationData
    loss_history: list[float]
    target_mean: float
    target_std: float
    standardize_y: bool
    low_fidelity: str
    target_fidelity: str
    low_fidelity_index: int
    target_fidelity_index: int
    fidelity_observation_counts: dict[str, int]
    low_fidelity_kernel: str
    discrepancy_kernel: str
    ard: bool
    noise_mode: str
    noise_variance: float
    rho: float
    device: str
    dtype: torch.dtype

    def predict(
        self,
        X,
        *,
        target_fidelity: str | int | None = None,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool = True,
    ) -> CoKrigingGPRPrediction:
        """Predict one modeled fidelity for new samples."""
        return _predict_cokriging_gpr(
            self,
            X,
            target_fidelity=target_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )


class ExactTwoLevelCoKrigingGPRModel(gpytorch.models.ExactGP):
    """Exact two-level autoregressive co-kriging model.

    The covariance follows

    ``f_high(x) = rho * f_low(x) + delta(x)``,

    with independent low-fidelity and discrepancy kernels. The first
    implementation supports exactly two fidelity labels and a shared learned
    Gaussian likelihood noise term.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_fidelity_indices: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        low_fidelity_index: int,
        target_fidelity_index: int,
        low_fidelity_kernel: str = "matern",
        discrepancy_kernel: str = "matern",
        ard_num_dims: int | None = None,
        initial_rho: float = 1.0,
    ):
        super().__init__((train_x, train_fidelity_indices), train_y, likelihood)
        self.low_fidelity_index = int(low_fidelity_index)
        self.target_fidelity_index = int(target_fidelity_index)
        self.register_parameter(
            name="rho",
            parameter=torch.nn.Parameter(
                torch.as_tensor(float(initial_rho), dtype=train_x.dtype, device=train_x.device)
            ),
        )
        self.register_parameter(
            name="low_mean",
            parameter=torch.nn.Parameter(
                torch.zeros((), dtype=train_x.dtype, device=train_x.device)
            ),
        )
        self.register_parameter(
            name="discrepancy_mean",
            parameter=torch.nn.Parameter(
                torch.zeros((), dtype=train_x.dtype, device=train_x.device)
            ),
        )
        self.low_covar_module = gpytorch.kernels.ScaleKernel(
            _make_gpytorch_base_kernel(low_fidelity_kernel, ard_num_dims=ard_num_dims)
        )
        self.discrepancy_covar_module = gpytorch.kernels.ScaleKernel(
            _make_gpytorch_base_kernel(discrepancy_kernel, ard_num_dims=ard_num_dims)
        )

    def forward(
        self,
        x: torch.Tensor,
        fidelity_indices: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        fidelity_indices = fidelity_indices.long().reshape(-1)
        if x.shape[0] != fidelity_indices.shape[0]:
            raise ValueError("x and fidelity_indices must contain the same number of rows")

        mean_x = self._mean_for_fidelity(fidelity_indices, reference=x)
        covar_x = self._covariance_for_fidelity(x, fidelity_indices)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _mean_for_fidelity(
        self,
        fidelity_indices: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rho = self.rho.to(dtype=reference.dtype, device=reference.device)
        low_mean = self.low_mean.to(dtype=reference.dtype, device=reference.device)
        discrepancy_mean = self.discrepancy_mean.to(dtype=reference.dtype, device=reference.device)
        target_mean = rho * low_mean + discrepancy_mean
        return torch.where(
            fidelity_indices == self.target_fidelity_index,
            target_mean.expand(fidelity_indices.shape[0]),
            low_mean.expand(fidelity_indices.shape[0]),
        )

    def _covariance_for_fidelity(
        self,
        x: torch.Tensor,
        fidelity_indices: torch.Tensor,
    ):
        rho = self.rho.to(dtype=x.dtype, device=x.device)
        is_target = fidelity_indices == self.target_fidelity_index
        low_weights = torch.where(is_target, rho.expand_as(x[:, 0]), torch.ones_like(x[:, 0]))
        discrepancy_weights = torch.where(
            is_target,
            torch.ones_like(x[:, 0]),
            torch.zeros_like(x[:, 0]),
        )
        low_weight_matrix = low_weights.unsqueeze(-1) * low_weights.unsqueeze(-2)
        discrepancy_weight_matrix = (
            discrepancy_weights.unsqueeze(-1) * discrepancy_weights.unsqueeze(-2)
        )

        low_covar = self.low_covar_module(x).mul(low_weight_matrix)
        discrepancy_covar = self.discrepancy_covar_module(x).mul(discrepancy_weight_matrix)
        return low_covar + discrepancy_covar


@dataclass(frozen=True)
class DeltaMultiFidelityGPRResult:
    """Fitted two-stage delta multi-fidelity GPR result."""

    correction_model: GPyTorchGPRResult
    low_fidelity_model: GPyTorchGPRResult | None
    rho: float
    intercept: float
    high_fidelity_target_mean: float
    high_fidelity_target_std: float
    correction_target: np.ndarray
    low_fidelity_at_high: np.ndarray
    fit_intercept: bool
    include_low_fidelity_uncertainty: bool
    device: str
    dtype: torch.dtype

    def predict(
        self,
        X,
        *,
        low_fidelity=None,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool = True,
        include_low_fidelity_uncertainty: bool | None = None,
    ) -> MultiFidelityGPRPrediction:
        """Predict high-fidelity values for new samples."""
        return _predict_delta_multifidelity_gpr(
            self,
            X,
            low_fidelity=low_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
            include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        )


def prepare_multifidelity_observations(
    X,
    y,
    fidelity,
    *,
    fidelity_order: Sequence[str] | None = None,
    target_fidelity: str | None = None,
    sample_id=None,
    noise_variance=None,
    feature_names: Sequence[str] | None = None,
    min_observations_per_fidelity: int = 1,
) -> MultiFidelityObservationData:
    """Validate and encode row-wise observations from ordered fidelities.

    ``fidelity_order`` should normally be supplied from low/cheap to
    high/target fidelity. If omitted, the order is inferred from first
    appearance in ``fidelity`` and the last level becomes the default target.
    """
    inferred_feature_names = _infer_feature_names(X)
    X_array = _to_2d_numpy(X, "X")
    y_array = _to_1d_numpy(y, "y")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    labels = _to_1d_fidelity_labels(fidelity, "fidelity")
    if X_array.shape[0] != labels.shape[0]:
        raise ValueError("X and fidelity must contain the same number of samples")

    min_count = _validate_min_observations_per_fidelity(min_observations_per_fidelity)
    fidelity_names = _resolve_fidelity_names(labels, fidelity_order, min_count)
    target_fidelity_resolved = _resolve_target_fidelity(target_fidelity, fidelity_names)
    index_by_name = {name: index for index, name in enumerate(fidelity_names)}
    fidelity_index = np.asarray([index_by_name[label] for label in labels], dtype=int)

    sample_id_array = _resolve_sample_id(sample_id, X_array.shape[0])
    noise_variance_array = _resolve_noise_variance(noise_variance, X_array.shape[0])
    feature_names_resolved = _resolve_feature_names(
        feature_names if feature_names is not None else inferred_feature_names,
        X_array.shape[1],
    )

    return MultiFidelityObservationData(
        X=X_array.copy(),
        y=y_array.astype(float, copy=True),
        fidelity_index=fidelity_index,
        fidelity_names=fidelity_names,
        target_fidelity=target_fidelity_resolved,
        sample_id=sample_id_array,
        noise_variance=noise_variance_array,
        feature_names=feature_names_resolved,
    )


def fit_cokriging_gpr(
    observation_data: MultiFidelityObservationData,
    *,
    low_fidelity: str | None = None,
    target_fidelity: str | None = None,
    low_fidelity_kernel: str = "matern",
    discrepancy_kernel: str = "matern",
    ard: bool = True,
    initial_rho: float | None = None,
    lr: float = 0.01,
    training_iter: int = 1000,
    initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    noise_mode: str = "shared",
    min_observations_per_fidelity: int = 2,
    device: str = "cpu",
    dtype: torch.dtype | str = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> CoKrigingGPRResult:
    """Fit a two-level autoregressive co-kriging GPR model.

    This first implementation supports exactly two fidelity levels with one
    shared learned Gaussian noise term. Use
    :func:`prepare_multifidelity_observations` to create ``observation_data``.
    """
    if not isinstance(observation_data, MultiFidelityObservationData):
        raise ValueError("observation_data must be a MultiFidelityObservationData instance")
    dtype = _resolve_torch_dtype(dtype)
    _validate_training_options(lr=lr, training_iter=training_iter, log_every=log_every)
    noise_mode = _validate_cokriging_noise_mode(noise_mode)
    _validate_cokriging_known_noise(observation_data, noise_mode=noise_mode)

    low_name, target_name, low_index, target_index = _resolve_two_level_fidelities(
        observation_data,
        low_fidelity=low_fidelity,
        target_fidelity=target_fidelity,
        min_observations_per_fidelity=min_observations_per_fidelity,
    )
    initial_rho_value = _resolve_initial_cokriging_rho(
        observation_data,
        low_fidelity_index=low_index,
        target_fidelity_index=target_index,
        initial_rho=initial_rho,
    )

    train_x = _to_tensor(observation_data.X, device=device, dtype=dtype)
    train_fidelity_indices = torch.as_tensor(
        observation_data.fidelity_index,
        dtype=torch.long,
        device=device,
    )
    train_y = _to_tensor(observation_data.y, device=device, dtype=dtype).reshape(-1)

    if standardize_y:
        target_mean = train_y.mean()
        target_std = train_y.std(unbiased=False)
        if target_std.item() <= 0:
            raise ValueError("y has zero standard deviation")
        train_y_model = (train_y - target_mean) / target_std
    else:
        target_mean = torch.tensor(0.0, dtype=dtype, device=device)
        target_std = torch.tensor(1.0, dtype=dtype, device=device)
        train_y_model = train_y

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
    if initial_noise is not None:
        if initial_noise <= 0:
            raise ValueError("initial_noise must be positive")
        likelihood.initialize(noise=float(initial_noise))

    model = ExactTwoLevelCoKrigingGPRModel(
        train_x,
        train_fidelity_indices,
        train_y_model,
        likelihood,
        low_fidelity_index=low_index,
        target_fidelity_index=target_index,
        low_fidelity_kernel=low_fidelity_kernel,
        discrepancy_kernel=discrepancy_kernel,
        ard_num_dims=train_x.shape[1] if ard else None,
        initial_rho=initial_rho_value,
    ).to(device=device, dtype=dtype)
    model.target_mean = target_mean.detach()
    model.target_std = target_std.detach()
    model.standardize_y = standardize_y
    model.fidelity_names = observation_data.fidelity_names
    model.low_fidelity = low_name
    model.target_fidelity = target_name

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_history: list[float] = []

    for iteration in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x, train_fidelity_indices)
        loss = -marginal_log_likelihood(output, train_y_model)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

        if verbose and _should_log_iteration(iteration, training_iter, log_every):
            _print_cokriging_training_status(iteration, training_iter, loss, likelihood, model)

    model.training_loss_history = loss_history
    rho = float(model.rho.detach().cpu().item())
    standardized_noise = float(likelihood.noise.detach().cpu().reshape(-1)[0])
    noise_variance = standardized_noise * float(target_std.detach().cpu().item()) ** 2

    return CoKrigingGPRResult(
        model=model,
        likelihood=likelihood,
        observation_data=observation_data,
        loss_history=loss_history,
        target_mean=float(target_mean.detach().cpu().item()),
        target_std=float(target_std.detach().cpu().item()),
        standardize_y=standardize_y,
        low_fidelity=low_name,
        target_fidelity=target_name,
        low_fidelity_index=low_index,
        target_fidelity_index=target_index,
        fidelity_observation_counts=observation_data.fidelity_observation_counts,
        low_fidelity_kernel=low_fidelity_kernel,
        discrepancy_kernel=discrepancy_kernel,
        ard=ard,
        noise_mode=noise_mode,
        noise_variance=noise_variance,
        rho=rho,
        device=device,
        dtype=dtype,
    )


def fit_delta_multifidelity_gpr(
    X_high,
    y_high,
    *,
    low_fidelity_high=None,
    X_low=None,
    y_low=None,
    fit_intercept: bool = True,
    correction_kernel: str = "matern",
    low_fidelity_kernel: str = "matern",
    ard: bool = True,
    lr: float = 0.01,
    training_iter: int = 1000,
    low_fidelity_training_iter: int | None = None,
    correction_initial_noise: float | None = 0.1,
    low_fidelity_initial_noise: float | None = 0.1,
    standardize_y: bool = True,
    include_low_fidelity_uncertainty: bool = True,
    device: str = "cpu",
    dtype: torch.dtype | str = torch.float64,
    verbose: bool = True,
    log_every: int = 100,
) -> DeltaMultiFidelityGPRResult:
    """Fit a delta multi-fidelity GPR model.

    The model uses the autoregressive correction

    ``y_high(x) = rho * y_low(x) + intercept + delta(x)``,

    where ``delta(x)`` is modeled by a standard exact GPR. Provide
    ``low_fidelity_high`` when low-fidelity values are already available at the
    high-fidelity training points. Alternatively, provide ``X_low`` and
    ``y_low`` to fit a low-fidelity GPR surrogate first.
    """
    dtype = _resolve_torch_dtype(dtype)
    X_high_array = _to_2d_numpy(X_high, "X_high")
    y_high_array = _to_1d_numpy(y_high, "y_high")
    check_consistent_length(X_high_array, y_high_array)
    _validate_feature_width_match(X_high_array, X_low, "X_high", "X_low")

    low_model: GPyTorchGPRResult | None = None
    if low_fidelity_high is None:
        if X_low is None or y_low is None:
            raise ValueError(
                "Provide low_fidelity_high, or provide both X_low and y_low to fit "
                "a low-fidelity surrogate"
            )
        low_model = fit_gpytorch_gpr(
            X_low,
            y_low,
            kernel=low_fidelity_kernel,
            ard=ard,
            lr=lr,
            training_iter=low_fidelity_training_iter or training_iter,
            initial_noise=low_fidelity_initial_noise,
            standardize_y=standardize_y,
            device=device,
            dtype=dtype,
            verbose=verbose,
            log_every=log_every,
        )
        low_prediction = low_model.predict(
            X_high_array,
            return_std=True,
            include_observation_noise=False,
        )
        low_fidelity_high_array = low_prediction.mean
    else:
        if X_low is not None or y_low is not None:
            raise ValueError("Use either low_fidelity_high or X_low/y_low, not both")
        low_fidelity_high_array = _to_1d_numpy(low_fidelity_high, "low_fidelity_high")

    check_consistent_length(X_high_array, low_fidelity_high_array)

    rho, intercept = _fit_linear_fidelity_map(
        low_fidelity_high_array,
        y_high_array,
        fit_intercept=fit_intercept,
    )
    correction_target = y_high_array - (rho * low_fidelity_high_array + intercept)
    standardize_correction = standardize_y and np.std(correction_target) > 0.0
    correction_model = fit_gpytorch_gpr(
        X_high_array,
        correction_target,
        kernel=correction_kernel,
        ard=ard,
        lr=lr,
        training_iter=training_iter,
        initial_noise=correction_initial_noise,
        standardize_y=standardize_correction,
        device=device,
        dtype=dtype,
        verbose=verbose,
        log_every=log_every,
    )

    return DeltaMultiFidelityGPRResult(
        correction_model=correction_model,
        low_fidelity_model=low_model,
        rho=float(rho),
        intercept=float(intercept),
        high_fidelity_target_mean=float(y_high_array.mean()),
        high_fidelity_target_std=float(y_high_array.std(ddof=0)),
        correction_target=correction_target.copy(),
        low_fidelity_at_high=low_fidelity_high_array.copy(),
        fit_intercept=fit_intercept,
        include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        device=device,
        dtype=dtype,
    )


class MultiFidelityGPRRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-style delta multi-fidelity GPR estimator.

    The estimator models scarce high-fidelity measurements by combining
    low-fidelity values with a GPR correction. Pass ``low_fidelity`` to
    :meth:`fit` and :meth:`predict` when low-fidelity values are externally
    available. If ``X_low`` and ``y_low`` are passed to :meth:`fit`, the
    estimator fits an internal low-fidelity surrogate and can predict without
    explicit low-fidelity values.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        correction_kernel: str = "matern",
        low_fidelity_kernel: str = "matern",
        ard: bool = True,
        lr: float = 0.01,
        training_iter: int = 1000,
        low_fidelity_training_iter: int | None = None,
        correction_initial_noise: float | None = 0.1,
        low_fidelity_initial_noise: float | None = 0.1,
        standardize_y: bool = True,
        include_low_fidelity_uncertainty: bool = True,
        device: str = "cpu",
        dtype: str | torch.dtype = "float64",
        verbose: bool = False,
        log_every: int = 100,
        include_observation_noise: bool = True,
        random_state: int | None = None,
    ):
        self.fit_intercept = fit_intercept
        self.correction_kernel = correction_kernel
        self.low_fidelity_kernel = low_fidelity_kernel
        self.ard = ard
        self.lr = lr
        self.training_iter = training_iter
        self.low_fidelity_training_iter = low_fidelity_training_iter
        self.correction_initial_noise = correction_initial_noise
        self.low_fidelity_initial_noise = low_fidelity_initial_noise
        self.standardize_y = standardize_y
        self.include_low_fidelity_uncertainty = include_low_fidelity_uncertainty
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.log_every = log_every
        self.include_observation_noise = include_observation_noise
        self.random_state = random_state

    def fit(self, X, y, *, low_fidelity=None, X_low=None, y_low=None):
        """Fit the multi-fidelity GPR model."""
        _seed_torch(self.random_state)
        X_checked = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        y_checked = column_or_1d(
            check_array(
                y,
                ensure_2d=False,
                dtype="numeric",
                ensure_all_finite=True,
                input_name="y",
            ),
            warn=True,
        )
        check_consistent_length(X_checked, y_checked)
        self.n_features_in_ = X_checked.shape[1]

        self.result_ = fit_delta_multifidelity_gpr(
            X_checked,
            y_checked,
            low_fidelity_high=low_fidelity,
            X_low=X_low,
            y_low=y_low,
            fit_intercept=self.fit_intercept,
            correction_kernel=self.correction_kernel,
            low_fidelity_kernel=self.low_fidelity_kernel,
            ard=self.ard,
            lr=self.lr,
            training_iter=self.training_iter,
            low_fidelity_training_iter=self.low_fidelity_training_iter,
            correction_initial_noise=self.correction_initial_noise,
            low_fidelity_initial_noise=self.low_fidelity_initial_noise,
            standardize_y=self.standardize_y,
            include_low_fidelity_uncertainty=self.include_low_fidelity_uncertainty,
            device=self.device,
            dtype=_resolve_torch_dtype(self.dtype),
            verbose=self.verbose,
            log_every=self.log_every,
        )
        self.correction_model_ = self.result_.correction_model
        self.low_fidelity_model_ = self.result_.low_fidelity_model
        self.rho_ = self.result_.rho
        self.intercept_ = self.result_.intercept
        self.correction_target_ = self.result_.correction_target.copy()
        self.low_fidelity_at_high_ = self.result_.low_fidelity_at_high.copy()
        return self

    def predict(
        self,
        X,
        *,
        low_fidelity=None,
        return_std: bool = False,
        include_observation_noise: bool | None = None,
        include_low_fidelity_uncertainty: bool | None = None,
    ):
        """Predict high-fidelity values."""
        prediction = self.predict_distribution(
            X,
            low_fidelity=low_fidelity,
            return_std=return_std,
            include_observation_noise=include_observation_noise,
            include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        )
        if return_std:
            return prediction.mean, prediction.std
        return prediction.mean

    def predict_distribution(
        self,
        X,
        *,
        low_fidelity=None,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool | None = None,
        include_low_fidelity_uncertainty: bool | None = None,
    ) -> MultiFidelityGPRPrediction:
        """Return high-fidelity predictive mean, uncertainty, and components."""
        check_is_fitted(self, "result_")
        X_checked = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_} features"
            )
        if include_observation_noise is None:
            include_observation_noise = self.include_observation_noise

        return self.result_.predict(
            X_checked,
            low_fidelity=low_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
            include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        )

    def score(self, X, y, sample_weight=None, *, low_fidelity=None) -> float:
        """Return R2 for high-fidelity predictions."""
        return r2_score(
            y,
            self.predict(X, low_fidelity=low_fidelity),
            sample_weight=sample_weight,
        )


class CoKrigingGPRRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-style two-level autoregressive co-kriging estimator."""

    def __init__(
        self,
        *,
        fidelity_order: Sequence[str] | None = None,
        low_fidelity: str | None = None,
        target_fidelity: str | None = None,
        low_fidelity_kernel: str = "matern",
        discrepancy_kernel: str = "matern",
        ard: bool = True,
        initial_rho: float | None = None,
        lr: float = 0.01,
        training_iter: int = 1000,
        initial_noise: float | None = 0.1,
        standardize_y: bool = True,
        noise_mode: str = "shared",
        min_observations_per_fidelity: int = 2,
        device: str = "cpu",
        dtype: str | torch.dtype = "float64",
        verbose: bool = False,
        log_every: int = 100,
        include_observation_noise: bool = True,
        random_state: int | None = None,
    ):
        self.fidelity_order = fidelity_order
        self.low_fidelity = low_fidelity
        self.target_fidelity = target_fidelity
        self.low_fidelity_kernel = low_fidelity_kernel
        self.discrepancy_kernel = discrepancy_kernel
        self.ard = ard
        self.initial_rho = initial_rho
        self.lr = lr
        self.training_iter = training_iter
        self.initial_noise = initial_noise
        self.standardize_y = standardize_y
        self.noise_mode = noise_mode
        self.min_observations_per_fidelity = min_observations_per_fidelity
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.log_every = log_every
        self.include_observation_noise = include_observation_noise
        self.random_state = random_state

    def fit(self, X, y, *, fidelity, sample_id=None):
        """Fit a two-level co-kriging GPR model."""
        _seed_torch(self.random_state)
        X_checked = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        y_checked = column_or_1d(
            check_array(
                y,
                ensure_2d=False,
                dtype="numeric",
                ensure_all_finite=True,
                input_name="y",
            ),
            warn=True,
        )
        check_consistent_length(X_checked, y_checked)
        self.n_features_in_ = X_checked.shape[1]

        observation_data = prepare_multifidelity_observations(
            X_checked,
            y_checked,
            fidelity,
            fidelity_order=self.fidelity_order,
            target_fidelity=self.target_fidelity,
            sample_id=sample_id,
            min_observations_per_fidelity=self.min_observations_per_fidelity,
        )
        self.result_ = fit_cokriging_gpr(
            observation_data,
            low_fidelity=self.low_fidelity,
            target_fidelity=self.target_fidelity,
            low_fidelity_kernel=self.low_fidelity_kernel,
            discrepancy_kernel=self.discrepancy_kernel,
            ard=self.ard,
            initial_rho=self.initial_rho,
            lr=self.lr,
            training_iter=self.training_iter,
            initial_noise=self.initial_noise,
            standardize_y=self.standardize_y,
            noise_mode=self.noise_mode,
            min_observations_per_fidelity=self.min_observations_per_fidelity,
            device=self.device,
            dtype=_resolve_torch_dtype(self.dtype),
            verbose=self.verbose,
            log_every=self.log_every,
        )
        self.model_ = self.result_.model
        self.likelihood_ = self.result_.likelihood
        self.observation_data_ = self.result_.observation_data
        self.fidelity_names_ = self.result_.observation_data.fidelity_names
        self.low_fidelity_ = self.result_.low_fidelity
        self.target_fidelity_ = self.result_.target_fidelity
        self.fidelity_observation_counts_ = self.result_.fidelity_observation_counts
        self.loss_history_ = list(self.result_.loss_history)
        self.rho_ = self.result_.rho
        self.noise_variance_ = self.result_.noise_variance
        return self

    def predict(
        self,
        X,
        *,
        target_fidelity: str | int | None = None,
        return_std: bool = False,
        include_observation_noise: bool | None = None,
    ):
        """Predict one modeled fidelity."""
        prediction = self.predict_distribution(
            X,
            target_fidelity=target_fidelity,
            return_std=return_std,
            include_observation_noise=include_observation_noise,
        )
        if return_std:
            return prediction.mean, prediction.std
        return prediction.mean

    def predict_distribution(
        self,
        X,
        *,
        target_fidelity: str | int | None = None,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool | None = None,
    ) -> CoKrigingGPRPrediction:
        """Return predictive mean, uncertainty, and interval for one fidelity."""
        check_is_fitted(self, "result_")
        X_checked = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_} features"
            )
        if include_observation_noise is None:
            include_observation_noise = self.include_observation_noise
        return self.result_.predict(
            X_checked,
            target_fidelity=target_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )

    def score(self, X, y, sample_weight=None, *, target_fidelity: str | int | None = None) -> float:
        """Return R2 for predictions at one fidelity."""
        return r2_score(
            y,
            self.predict(X, target_fidelity=target_fidelity),
            sample_weight=sample_weight,
        )


def _predict_delta_multifidelity_gpr(
    result: DeltaMultiFidelityGPRResult,
    X,
    *,
    low_fidelity,
    return_std: bool,
    confidence_level: float | None,
    include_observation_noise: bool,
    include_low_fidelity_uncertainty: bool | None,
) -> MultiFidelityGPRPrediction:
    _validate_confidence_level(confidence_level)
    X_array = _to_2d_numpy(X, "X")
    low_mean, low_std = _resolve_prediction_low_fidelity(
        result,
        X_array,
        low_fidelity=low_fidelity,
        return_std=return_std or confidence_level is not None,
    )
    correction_prediction = result.correction_model.predict(
        X_array,
        return_std=return_std or confidence_level is not None,
        include_observation_noise=include_observation_noise,
    )
    mean = result.rho * low_mean + result.intercept + correction_prediction.mean

    correction_std = correction_prediction.std
    std = None
    if return_std or confidence_level is not None:
        if correction_std is None:
            raise ValueError("Internal correction prediction did not return standard deviations")
        include_low = (
            result.include_low_fidelity_uncertainty
            if include_low_fidelity_uncertainty is None
            else bool(include_low_fidelity_uncertainty)
        )
        variance = correction_std**2
        if include_low and low_std is not None:
            variance = variance + (result.rho * low_std) ** 2
        std = np.sqrt(np.maximum(variance, 0.0))

    lower = None
    upper = None
    if confidence_level is not None:
        if std is None:
            raise ValueError("confidence_level requires return_std=True")
        z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        lower = mean - z_value * std
        upper = mean + z_value * std

    return MultiFidelityGPRPrediction(
        mean=mean,
        std=std,
        lower=lower,
        upper=upper,
        low_fidelity_mean=low_mean,
        low_fidelity_std=low_std,
        correction_mean=correction_prediction.mean,
        correction_std=correction_std,
        rho=result.rho,
        intercept=result.intercept,
    )


def _predict_cokriging_gpr(
    result: CoKrigingGPRResult,
    X,
    *,
    target_fidelity: str | int | None,
    return_std: bool,
    confidence_level: float | None,
    include_observation_noise: bool,
) -> CoKrigingGPRPrediction:
    _validate_confidence_level(confidence_level)
    result.model.eval()
    result.likelihood.eval()
    test_x = _to_tensor(X, device=result.device, dtype=result.dtype)
    _validate_cokriging_prediction_features(test_x, result.model)
    fidelity_label, fidelity_index = _resolve_cokriging_prediction_fidelity(
        result,
        target_fidelity=target_fidelity,
    )
    pred_fidelity_indices = torch.full(
        (test_x.shape[0],),
        int(fidelity_index),
        dtype=torch.long,
        device=result.device,
    )

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_distribution = result.model(test_x, pred_fidelity_indices)
        prediction_distribution = (
            result.likelihood(latent_distribution) if include_observation_noise else latent_distribution
        )
        mean = prediction_distribution.mean
        std = prediction_distribution.stddev if return_std or confidence_level is not None else None

    target_mean = result.model.target_mean.to(dtype=mean.dtype, device=mean.device)
    target_std = result.model.target_std.to(dtype=mean.dtype, device=mean.device)
    mean = mean * target_std + target_mean
    if std is not None:
        std = std * target_std

    mean_array = mean.detach().cpu().numpy()
    std_array = None if std is None else std.detach().cpu().numpy()
    lower = None
    upper = None
    if confidence_level is not None:
        if std_array is None:
            raise ValueError("confidence_level requires return_std=True")
        z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        lower = mean_array - z_value * std_array
        upper = mean_array + z_value * std_array

    components = _predict_cokriging_components(
        result,
        test_x,
        fidelity_index=fidelity_index,
        return_std=return_std or confidence_level is not None,
    )

    return CoKrigingGPRPrediction(
        mean=mean_array,
        std=std_array,
        lower=lower,
        upper=upper,
        low_fidelity_mean=components["low_fidelity_mean"],
        low_fidelity_std=components["low_fidelity_std"],
        scaled_low_fidelity_mean=components["scaled_low_fidelity_mean"],
        scaled_low_fidelity_std=components["scaled_low_fidelity_std"],
        discrepancy_mean=components["discrepancy_mean"],
        discrepancy_std=components["discrepancy_std"],
        fidelity=fidelity_label,
        rho=float(result.model.rho.detach().cpu().item()),
    )


def _predict_cokriging_components(
    result: CoKrigingGPRResult,
    test_x: torch.Tensor,
    *,
    fidelity_index: int,
    return_std: bool,
) -> dict[str, np.ndarray | None]:
    if fidelity_index == result.target_fidelity_index:
        return _predict_cokriging_target_components(result, test_x, return_std=return_std)
    if fidelity_index == result.low_fidelity_index:
        return _predict_cokriging_low_components(result, test_x, return_std=return_std)
    return _empty_cokriging_components()


def _predict_cokriging_target_components(
    result: CoKrigingGPRResult,
    test_x: torch.Tensor,
    *,
    return_std: bool,
) -> dict[str, np.ndarray | None]:
    n_samples = int(test_x.shape[0])
    low_indices = torch.full(
        (n_samples,),
        result.low_fidelity_index,
        dtype=torch.long,
        device=result.device,
    )
    target_indices = torch.full(
        (n_samples,),
        result.target_fidelity_index,
        dtype=torch.long,
        device=result.device,
    )
    component_x = torch.cat([test_x, test_x], dim=0)
    component_indices = torch.cat([low_indices, target_indices], dim=0)
    target_mean, target_std = _cokriging_target_transform(result)
    rho = result.model.rho.detach().to(dtype=test_x.dtype, device=test_x.device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_distribution = result.model(component_x, component_indices)
        latent_mean = latent_distribution.mean
        low_mean = latent_mean[:n_samples]
        high_mean = latent_mean[n_samples:]
        discrepancy_mean = high_mean - rho * low_mean

        low_std = None
        scaled_low_std = None
        discrepancy_std = None
        if return_std:
            covariance = latent_distribution.covariance_matrix
            low_variance = covariance.diag()[:n_samples].clamp_min(0.0)
            high_variance = covariance.diag()[n_samples:].clamp_min(0.0)
            paired_covariance = covariance[
                torch.arange(n_samples, device=test_x.device) + n_samples,
                torch.arange(n_samples, device=test_x.device),
            ]
            discrepancy_variance = (
                high_variance + rho.pow(2) * low_variance - 2.0 * rho * paired_covariance
            ).clamp_min(0.0)
            low_std = torch.sqrt(low_variance) * target_std
            scaled_low_std = torch.abs(rho) * torch.sqrt(low_variance) * target_std
            discrepancy_std = torch.sqrt(discrepancy_variance) * target_std

    low_mean_original = low_mean * target_std + target_mean
    scaled_low_mean = rho * low_mean * target_std
    discrepancy_mean_original = discrepancy_mean * target_std + target_mean

    return {
        "low_fidelity_mean": _tensor_to_numpy(low_mean_original),
        "low_fidelity_std": _optional_tensor_to_numpy(low_std),
        "scaled_low_fidelity_mean": _tensor_to_numpy(scaled_low_mean),
        "scaled_low_fidelity_std": _optional_tensor_to_numpy(scaled_low_std),
        "discrepancy_mean": _tensor_to_numpy(discrepancy_mean_original),
        "discrepancy_std": _optional_tensor_to_numpy(discrepancy_std),
    }


def _predict_cokriging_low_components(
    result: CoKrigingGPRResult,
    test_x: torch.Tensor,
    *,
    return_std: bool,
) -> dict[str, np.ndarray | None]:
    low_indices = torch.full(
        (test_x.shape[0],),
        result.low_fidelity_index,
        dtype=torch.long,
        device=result.device,
    )
    target_mean, target_std = _cokriging_target_transform(result)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_distribution = result.model(test_x, low_indices)
        low_mean = latent_distribution.mean
        low_std = latent_distribution.stddev * target_std if return_std else None

    components = _empty_cokriging_components()
    components["low_fidelity_mean"] = _tensor_to_numpy(low_mean * target_std + target_mean)
    components["low_fidelity_std"] = _optional_tensor_to_numpy(low_std)
    return components


def _empty_cokriging_components() -> dict[str, np.ndarray | None]:
    return {
        "low_fidelity_mean": None,
        "low_fidelity_std": None,
        "scaled_low_fidelity_mean": None,
        "scaled_low_fidelity_std": None,
        "discrepancy_mean": None,
        "discrepancy_std": None,
    }


def _cokriging_target_transform(result: CoKrigingGPRResult) -> tuple[torch.Tensor, torch.Tensor]:
    target_mean = result.model.target_mean.to(dtype=result.dtype, device=result.device)
    target_std = result.model.target_std.to(dtype=result.dtype, device=result.device)
    return target_mean, target_std


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _optional_tensor_to_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return _tensor_to_numpy(value)


def _resolve_prediction_low_fidelity(
    result: DeltaMultiFidelityGPRResult,
    X: np.ndarray,
    *,
    low_fidelity,
    return_std: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if low_fidelity is not None:
        values = _to_1d_numpy(low_fidelity, "low_fidelity")
        check_consistent_length(X, values)
        return values, None
    if result.low_fidelity_model is None:
        raise ValueError(
            "low_fidelity is required for prediction because no low-fidelity "
            "surrogate was fitted"
        )
    prediction = result.low_fidelity_model.predict(
        X,
        return_std=return_std,
        include_observation_noise=False,
    )
    return prediction.mean, prediction.std


def _fit_linear_fidelity_map(
    low_fidelity: np.ndarray,
    high_fidelity: np.ndarray,
    *,
    fit_intercept: bool,
) -> tuple[float, float]:
    if np.std(low_fidelity) <= 0:
        raise ValueError("low_fidelity values must have nonzero variance")
    if fit_intercept:
        design = np.column_stack([low_fidelity, np.ones_like(low_fidelity)])
        rho, intercept = np.linalg.lstsq(design, high_fidelity, rcond=None)[0]
    else:
        rho = float(np.dot(low_fidelity, high_fidelity) / np.dot(low_fidelity, low_fidelity))
        intercept = 0.0
    return float(rho), float(intercept)


def _resolve_two_level_fidelities(
    observation_data: MultiFidelityObservationData,
    *,
    low_fidelity: str | None,
    target_fidelity: str | None,
    min_observations_per_fidelity: int,
) -> tuple[str, str, int, int]:
    if observation_data.n_fidelities != 2:
        raise ValueError(
            "Two-level co-kriging currently requires exactly two fidelity levels; "
            f"got {observation_data.n_fidelities}"
        )
    min_count = _validate_min_observations_per_fidelity(min_observations_per_fidelity)
    low_count_levels = [
        name
        for name, count in observation_data.fidelity_observation_counts.items()
        if count < min_count
    ]
    if low_count_levels:
        raise ValueError(
            "Each fidelity must have at least "
            f"{min_count} observed value(s); low-count fidelity level(s): {low_count_levels}"
        )

    target_name = (
        observation_data.target_fidelity
        if target_fidelity is None
        else _normalize_fidelity_label(target_fidelity, "target_fidelity")
    )
    if target_name not in observation_data.fidelity_names:
        raise ValueError(
            f"target_fidelity must be one of {observation_data.fidelity_names}; got {target_name!r}"
        )
    low_candidates = [name for name in observation_data.fidelity_names if name != target_name]
    if low_fidelity is None:
        low_name = low_candidates[0]
    else:
        low_name = _normalize_fidelity_label(low_fidelity, "low_fidelity")
        if low_name not in observation_data.fidelity_names:
            raise ValueError(
                f"low_fidelity must be one of {observation_data.fidelity_names}; got {low_name!r}"
            )
        if low_name == target_name:
            raise ValueError("low_fidelity and target_fidelity must be different")

    low_index = observation_data.fidelity_names.index(low_name)
    target_index = observation_data.fidelity_names.index(target_name)
    return low_name, target_name, low_index, target_index


def _resolve_initial_cokriging_rho(
    observation_data: MultiFidelityObservationData,
    *,
    low_fidelity_index: int,
    target_fidelity_index: int,
    initial_rho: float | None,
) -> float:
    if initial_rho is not None:
        value = float(initial_rho)
        if not np.isfinite(value):
            raise ValueError("initial_rho must be finite")
        return value

    paired_rho = _paired_sample_initial_rho(
        observation_data,
        low_fidelity_index=low_fidelity_index,
        target_fidelity_index=target_fidelity_index,
    )
    return 1.0 if paired_rho is None else paired_rho


def _paired_sample_initial_rho(
    observation_data: MultiFidelityObservationData,
    *,
    low_fidelity_index: int,
    target_fidelity_index: int,
) -> float | None:
    if observation_data.sample_id is None:
        return None

    low_by_id: dict[object, float] = {}
    target_by_id: dict[object, float] = {}
    for row_index, sample_id in enumerate(observation_data.sample_id):
        fidelity_index = int(observation_data.fidelity_index[row_index])
        if fidelity_index == low_fidelity_index:
            low_by_id.setdefault(sample_id, float(observation_data.y[row_index]))
        elif fidelity_index == target_fidelity_index:
            target_by_id.setdefault(sample_id, float(observation_data.y[row_index]))

    paired_ids = [sample_id for sample_id in low_by_id if sample_id in target_by_id]
    if len(paired_ids) < 2:
        return None
    low_values = np.asarray([low_by_id[sample_id] for sample_id in paired_ids], dtype=float)
    target_values = np.asarray([target_by_id[sample_id] for sample_id in paired_ids], dtype=float)
    if np.std(low_values) <= 0:
        return None
    rho, _ = _fit_linear_fidelity_map(low_values, target_values, fit_intercept=True)
    return rho


def _validate_cokriging_noise_mode(noise_mode: str) -> str:
    normalized = str(noise_mode).strip().lower()
    aliases = {
        "shared": "shared",
        "learned": "shared",
        "global": "shared",
    }
    if normalized not in aliases:
        raise ValueError("noise_mode must be 'shared' for the first co-kriging implementation")
    return aliases[normalized]


def _validate_cokriging_known_noise(
    observation_data: MultiFidelityObservationData,
    *,
    noise_mode: str,
) -> None:
    if observation_data.noise_variance is None:
        return
    if noise_mode == "shared":
        raise ValueError(
            "Known per-observation noise is stored by MultiFidelityObservationData "
            "but is not supported by the first co-kriging implementation"
        )


def _resolve_cokriging_prediction_fidelity(
    result: CoKrigingGPRResult,
    *,
    target_fidelity: str | int | None,
) -> tuple[str, int]:
    if target_fidelity is None:
        label = result.target_fidelity
        index = result.target_fidelity_index
        return label, index
    index = result.observation_data._resolve_fidelity_index(target_fidelity)
    return result.observation_data.fidelity_names[index], index


def _validate_cokriging_prediction_features(
    test_x: torch.Tensor,
    model: ExactTwoLevelCoKrigingGPRModel,
) -> None:
    if test_x.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    expected_features = model.train_inputs[0].shape[1]
    if test_x.shape[1] != expected_features:
        raise ValueError(
            f"X has {test_x.shape[1]} features, but the model was fitted with "
            f"{expected_features} features"
        )


def _print_cokriging_training_status(
    iteration: int,
    training_iter: int,
    loss: torch.Tensor,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    model: ExactTwoLevelCoKrigingGPRModel,
) -> None:
    noise = likelihood.noise.item()
    low_outputscale = model.low_covar_module.outputscale.item()
    discrepancy_outputscale = model.discrepancy_covar_module.outputscale.item()
    print(
        f"Iter {iteration + 1:4d}/{training_iter} | "
        f"Loss: {loss.item():.4f} | "
        f"Noise: {noise:.4e} | "
        f"Rho: {model.rho.item():.4f} | "
        f"Low outputscale: {low_outputscale:.4e} | "
        f"Discrepancy outputscale: {discrepancy_outputscale:.4e}"
    )


def _infer_feature_names(X) -> tuple[str, ...] | None:
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    return tuple(str(column) for column in columns)


def _to_1d_fidelity_labels(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")

    labels: list[str] = []
    for raw_value in array:
        labels.append(_normalize_fidelity_label(raw_value, name))
    return np.asarray(labels, dtype=object)


def _resolve_fidelity_names(
    labels: np.ndarray,
    fidelity_order: Sequence[str] | None,
    min_observations_per_fidelity: int,
) -> tuple[str, ...]:
    if fidelity_order is None:
        fidelity_names = tuple(dict.fromkeys(labels.tolist()))
    else:
        fidelity_names = tuple(
            _normalize_fidelity_label(name, "fidelity_order") for name in fidelity_order
        )

    if not fidelity_names:
        raise ValueError("fidelity_order must contain at least one fidelity level")
    if len(set(fidelity_names)) != len(fidelity_names):
        raise ValueError("fidelity_order must contain unique fidelity labels")

    observed = set(labels.tolist())
    expected = set(fidelity_names)
    unknown = sorted(observed - expected)
    if unknown:
        raise ValueError(
            "fidelity contains labels not present in fidelity_order: "
            f"{unknown}; expected one of {fidelity_names}"
        )

    low_count_levels = [
        name
        for name in fidelity_names
        if int(np.sum(labels == name)) < min_observations_per_fidelity
    ]
    if low_count_levels:
        raise ValueError(
            "Each fidelity must have at least "
            f"{min_observations_per_fidelity} observed value(s); low-count "
            f"fidelity level(s): {low_count_levels}"
        )
    return fidelity_names


def _resolve_target_fidelity(
    target_fidelity: str | None,
    fidelity_names: tuple[str, ...],
) -> str:
    target = (
        fidelity_names[-1]
        if target_fidelity is None
        else _normalize_fidelity_label(target_fidelity, "target_fidelity")
    )
    if target not in fidelity_names:
        raise ValueError(f"target_fidelity must be one of {fidelity_names}; got {target!r}")
    return target


def _normalize_fidelity_label(value, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must not contain missing labels")
    label = str(value).strip()
    if label == "" or label.lower() == "nan":
        raise ValueError(f"{name} must not contain missing labels")
    return label


def _resolve_sample_id(sample_id, n_samples: int) -> np.ndarray | None:
    if sample_id is None:
        return None
    sample_id_array = np.asarray(sample_id, dtype=object).reshape(-1)
    if sample_id_array.shape[0] != n_samples:
        raise ValueError(
            f"sample_id must contain {n_samples} value(s); got {sample_id_array.shape[0]}"
        )
    return sample_id_array.copy()


def _resolve_noise_variance(noise_variance, n_samples: int) -> np.ndarray | None:
    if noise_variance is None:
        return None
    noise_array = _to_1d_numpy(noise_variance, "noise_variance")
    if noise_array.shape[0] != n_samples:
        raise ValueError(
            f"noise_variance must contain {n_samples} value(s); got {noise_array.shape[0]}"
        )
    if np.any(noise_array < 0.0):
        raise ValueError("noise_variance must contain non-negative values")
    return noise_array.astype(float, copy=True)


def _resolve_feature_names(
    feature_names: Sequence[str] | None,
    n_features: int,
) -> tuple[str, ...] | None:
    if feature_names is None:
        return None
    resolved = tuple(str(name) for name in feature_names)
    if len(resolved) != n_features:
        raise ValueError(f"feature_names must contain {n_features} value(s); got {len(resolved)}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("feature_names must be unique")
    return resolved


def _validate_min_observations_per_fidelity(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("min_observations_per_fidelity must be a positive integer")
    min_count = int(value)
    if min_count < 1:
        raise ValueError("min_observations_per_fidelity must be at least 1")
    return min_count


def _to_2d_numpy(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_1d_numpy(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_feature_width_match(X_high: np.ndarray, X_low, high_name: str, low_name: str) -> None:
    if X_low is None:
        return
    X_low_array = _to_2d_numpy(X_low, low_name)
    if X_high.shape[1] != X_low_array.shape[1]:
        raise ValueError(
            f"{high_name} has {X_high.shape[1]} features, but {low_name} has "
            f"{X_low_array.shape[1]} features"
        )


def _validate_confidence_level(confidence_level: float | None) -> None:
    if confidence_level is None:
        return
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower()
    if normalized in {"float64", "double", "torch.float64"}:
        return torch.float64
    if normalized in {"float32", "single", "torch.float32"}:
        return torch.float32
    raise ValueError("dtype must be 'float64', 'float32', or a torch dtype")


def _seed_torch(random_state: int | None) -> None:
    if random_state is not None:
        torch.manual_seed(int(random_state))
