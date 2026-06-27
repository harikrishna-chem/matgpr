from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .gpytorch_gpr import GPyTorchPrediction, PhysicsEquation, PhysicsInformedMean, fit_gpytorch_gpr

__all__ = [
    "MatGPRRegressor",
    "PhysicsInformedGPRRegressor",
]


class MatGPRRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-style wrapper around exact GPyTorch GPR.

    This estimator exposes the familiar ``fit``, ``predict``, ``score``,
    ``get_params``, and ``set_params`` interface while reusing the lower-level
    :func:`matgpr.fit_gpytorch_gpr` implementation. It is intended for
    notebook workflows, cross-validation utilities, and users who prefer a
    standard estimator object over separate train/predict helper functions.
    """

    def __init__(
        self,
        *,
        kernel: str = "matern",
        ard: bool = True,
        lr: float = 0.01,
        training_iter: int = 1000,
        initial_noise: float | None = 0.1,
        standardize_y: bool = True,
        device: str = "cpu",
        dtype: str | torch.dtype = "float64",
        verbose: bool = False,
        log_every: int = 100,
        include_observation_noise: bool = True,
        random_state: int | None = None,
    ):
        self.kernel = kernel
        self.ard = ard
        self.lr = lr
        self.training_iter = training_iter
        self.initial_noise = initial_noise
        self.standardize_y = standardize_y
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.log_every = log_every
        self.include_observation_noise = include_observation_noise
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the Gaussian Process Regressor.

        Parameters
        ----------
        X
            Numeric feature matrix. Dataframes are accepted; string column
            names are stored in ``feature_names_in_`` for prediction-time
            consistency checks.
        y
            One-dimensional target values.
        """
        feature_names = _feature_names_from_input(X)
        X_checked, y_checked = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype="numeric",
            y_numeric=True,
        )
        self.n_features_in_ = X_checked.shape[1]
        if feature_names is not None:
            if len(feature_names) != self.n_features_in_:
                raise ValueError("Number of dataframe columns does not match validated features")
            self.feature_names_in_ = feature_names
        elif hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")

        _seed_torch(self.random_state)
        dtype = _resolve_torch_dtype(self.dtype)
        mean_module = self._build_mean_module(X_checked=X_checked, y_checked=y_checked)

        self.result_ = fit_gpytorch_gpr(
            X_checked,
            y_checked,
            kernel=self.kernel,
            ard=self.ard,
            mean_module=mean_module,
            lr=self.lr,
            training_iter=self.training_iter,
            initial_noise=self.initial_noise,
            standardize_y=self.standardize_y,
            device=self.device,
            dtype=dtype,
            verbose=self.verbose,
            log_every=self.log_every,
        )
        self.model_ = self.result_.model
        self.likelihood_ = self.result_.likelihood
        self.loss_history_ = list(self.result_.loss_history)
        self.target_mean_ = self.result_.target_mean
        self.target_std_ = self.result_.target_std
        self.mean_module_ = self.result_.model.mean_module
        self._after_fit()
        return self

    def predict(
        self,
        X,
        *,
        return_std: bool = False,
        include_observation_noise: bool | None = None,
    ):
        """Predict target values for new samples.

        Parameters
        ----------
        X
            Numeric feature matrix with the same columns used during fitting.
        return_std
            If ``True``, return ``(mean, std)`` like scikit-learn's
            ``GaussianProcessRegressor``.
        include_observation_noise
            If ``True``, predictive uncertainty includes fitted observation
            noise. If omitted, the estimator-level setting is used.
        """
        prediction = self.predict_distribution(
            X,
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
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool | None = None,
    ) -> GPyTorchPrediction:
        """Return predictive mean, uncertainty, and optional confidence bounds."""
        check_is_fitted(self, "result_")
        X_checked = self._validate_prediction_input(X)
        if include_observation_noise is None:
            include_observation_noise = self.include_observation_noise

        return self.result_.predict(
            X_checked,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )

    def predict_interval(
        self,
        X,
        *,
        confidence_level: float = 0.95,
        include_observation_noise: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predictive mean and confidence interval bounds."""
        prediction = self.predict_distribution(
            X,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )
        return prediction.mean, prediction.lower, prediction.upper

    def score(self, X, y, sample_weight=None) -> float:
        """Return the coefficient of determination, R2."""
        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    def _build_mean_module(self, *, X_checked: np.ndarray, y_checked: np.ndarray):
        return None

    def _after_fit(self) -> None:
        return None

    def _validate_prediction_input(self, X) -> np.ndarray:
        if hasattr(self, "feature_names_in_"):
            input_feature_names = _feature_names_from_input(X)
            if input_feature_names is not None and not np.array_equal(
                input_feature_names,
                self.feature_names_in_,
            ):
                raise ValueError(
                    "Prediction features must match the fitted feature names and order"
                )

        X_checked = check_array(
            X,
            ensure_2d=True,
            dtype="numeric",
            ensure_min_samples=1,
        )
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted with "
                f"{self.n_features_in_} features"
            )
        return X_checked


class PhysicsInformedGPRRegressor(MatGPRRegressor):
    """Scikit-learn-style physics-informed Gaussian Process Regressor.

    Physics enters through a :class:`matgpr.PhysicsInformedMean` module. The GP
    then learns the residual between the physics equation and the data.
    """

    def __init__(
        self,
        *,
        equation: PhysicsEquation | None = None,
        feature_indices: Mapping[str, int] | None = None,
        physics_features: Sequence[str] | None = None,
        learnable_parameters: Mapping[str, float] | None = None,
        positive_parameters: Sequence[str] = (),
        fixed_parameters: Mapping[str, float] | None = None,
        feature_means: Mapping[str, float] | None = None,
        feature_stds: Mapping[str, float] | None = None,
        kernel: str = "matern",
        ard: bool = True,
        lr: float = 0.01,
        training_iter: int = 1000,
        initial_noise: float | None = 0.1,
        standardize_y: bool = True,
        device: str = "cpu",
        dtype: str | torch.dtype = "float64",
        verbose: bool = False,
        log_every: int = 100,
        include_observation_noise: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            kernel=kernel,
            ard=ard,
            lr=lr,
            training_iter=training_iter,
            initial_noise=initial_noise,
            standardize_y=standardize_y,
            device=device,
            dtype=dtype,
            verbose=verbose,
            log_every=log_every,
            include_observation_noise=include_observation_noise,
            random_state=random_state,
        )
        self.equation = equation
        self.feature_indices = feature_indices
        self.physics_features = physics_features
        self.learnable_parameters = learnable_parameters
        self.positive_parameters = positive_parameters
        self.fixed_parameters = fixed_parameters
        self.feature_means = feature_means
        self.feature_stds = feature_stds

    def _build_mean_module(self, *, X_checked: np.ndarray, y_checked: np.ndarray):
        if self.equation is None:
            raise ValueError("equation must be provided before fitting")

        feature_indices = self._resolve_physics_feature_indices()
        return PhysicsInformedMean(
            equation=self.equation,
            feature_indices=feature_indices,
            learnable_parameters=self.learnable_parameters,
            positive_parameters=self.positive_parameters,
            fixed_parameters=self.fixed_parameters,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds,
        )

    def _after_fit(self) -> None:
        if hasattr(self.mean_module_, "current_parameter_values"):
            self.learned_physics_parameters_ = self.mean_module_.current_parameter_values()

    def _resolve_physics_feature_indices(self) -> dict[str, int]:
        if self.feature_indices is not None and self.physics_features is not None:
            raise ValueError("Use either feature_indices or physics_features, not both")

        if self.feature_indices is not None:
            return dict(self.feature_indices)

        if self.physics_features is None:
            return {}

        if not hasattr(self, "feature_names_in_"):
            raise ValueError("physics_features requires fitting with a dataframe that has columns")

        fitted_names = list(self.feature_names_in_)
        missing = [name for name in self.physics_features if name not in fitted_names]
        if missing:
            raise ValueError(f"physics_features were not found in X columns: {missing}")
        return {name: fitted_names.index(name) for name in self.physics_features}


def _feature_names_from_input(X) -> np.ndarray | None:
    if not hasattr(X, "columns"):
        return None
    names = np.asarray(X.columns, dtype=object)
    if not all(isinstance(name, str) for name in names):
        return None
    return names


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower().replace("torch.", "")
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "single": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in aliases:
        raise ValueError("dtype must be one of: float32, float64, or a torch dtype")
    return aliases[normalized]


def _seed_torch(random_state: int | None) -> None:
    if random_state is None:
        return
    torch.manual_seed(int(random_state))
