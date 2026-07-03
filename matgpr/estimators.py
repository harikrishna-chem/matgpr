from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover - compatibility with older scikit-learn
    validate_data = None

from .gpytorch_gpr import GPyTorchPrediction, PhysicsEquation, PhysicsInformedMean, fit_gpytorch_gpr
from .multitask_gpr import MultitaskGPyTorchPrediction, fit_multitask_gpytorch_gpr

__all__ = [
    "MatGPRRegressor",
    "MissingValueReport",
    "MultitaskGPRRegressor",
    "PhysicsInformedGPRRegressor",
]


@dataclass(frozen=True)
class MissingValueReport:
    """Summary of estimator-level missing-value handling.

    The fit-time report is attached to fitted estimators as
    ``missing_report_``. Prediction-time reports are returned by
    ``summarize_prediction_missing_values`` without mutating estimator state.
    The report records the selected policy, row counts, feature-level missing
    counts, and any imputation settings used by the estimator.
    """

    stage: str
    policy: str
    input_rows: int
    output_rows: int
    dropped_rows: int
    rows_with_missing_features: int
    rows_with_missing_target: int
    feature_missing_counts: dict[str, int]
    imputed_features: tuple[str, ...] = ()
    imputation_strategy: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain dictionary for notebook display."""
        return asdict(self)


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
        missing: str = "error",
        imputation_strategy: str = "median",
        imputation_fill_value: object | None = None,
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
        self.missing = missing
        self.imputation_strategy = imputation_strategy
        self.imputation_fill_value = imputation_fill_value
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

        Notes
        -----
        Missing-value handling is controlled by ``missing``. ``"error"``
        rejects missing data, ``"drop"`` removes incomplete training rows, and
        ``"impute"`` learns a numeric feature imputer from the training data.
        Missing targets are never imputed; they are either rejected or dropped.
        """
        X_checked, y_checked = _validate_fit_input(self, X, y)

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

    def summarize_prediction_missing_values(self, X) -> MissingValueReport:
        """Summarize missing values in prediction features without predicting.

        This method is useful for auditing candidate pools before calling
        :meth:`predict`. It does not mutate the estimator, which keeps
        prediction calls compatible with scikit-learn conventions.
        """
        check_is_fitted(self, "result_")
        policy = _validate_missing_policy(self.missing)
        X_checked = _validate_prediction_features(self, X, allow_nan=True)
        return _make_missing_report(
            self,
            stage="predict",
            policy=policy,
            X_checked=X_checked,
            y_checked=None,
            output_rows=X_checked.shape[0],
            imputed_features=_imputed_feature_names(self, X_checked)
            if policy == "impute"
            else (),
            imputation_strategy=_validate_imputation_strategy(self.imputation_strategy)
            if policy == "impute"
            else None,
        )

    def score(self, X, y, sample_weight=None) -> float:
        """Return the coefficient of determination, R2."""
        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    def _build_mean_module(self, *, X_checked: np.ndarray, y_checked: np.ndarray):
        return None

    def _after_fit(self) -> None:
        return None

    def _validate_prediction_input(self, X) -> np.ndarray:
        return _validate_predict_input(self, X)


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
        missing: str = "error",
        imputation_strategy: str = "median",
        imputation_fill_value: object | None = None,
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
            missing=missing,
            imputation_strategy=imputation_strategy,
            imputation_fill_value=imputation_fill_value,
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


class MultitaskGPRRegressor(MatGPRRegressor):
    """Scikit-learn-style wrapper around exact multitask GPyTorch GPR.

    This estimator models complete multi-output target matrices with a shared
    input-space kernel and a learned task covariance. It is intended for
    materials datasets where each row has multiple related properties observed,
    such as strength and ductility for the same alloy or permeability and
    selectivity for the same polymer.
    """

    def __init__(
        self,
        *,
        task_names: Sequence[str] | None = None,
        task_covar_rank: int = 1,
        kernel: str = "matern",
        ard: bool = True,
        lr: float = 0.01,
        training_iter: int = 1000,
        initial_noise: float | None = 0.1,
        initial_task_noises: Sequence[float] | None = None,
        standardize_y: bool = True,
        missing: str = "error",
        imputation_strategy: str = "median",
        imputation_fill_value: object | None = None,
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
            missing=missing,
            imputation_strategy=imputation_strategy,
            imputation_fill_value=imputation_fill_value,
            device=device,
            dtype=dtype,
            verbose=verbose,
            log_every=log_every,
            include_observation_noise=include_observation_noise,
            random_state=random_state,
        )
        self.task_names = task_names
        self.task_covar_rank = task_covar_rank
        self.initial_task_noises = initial_task_noises

    def fit(self, X, y):
        """Fit the multitask Gaussian Process Regressor.

        Parameters
        ----------
        X
            Numeric feature matrix. Dataframes are accepted; string column
            names are stored in ``feature_names_in_`` for prediction-time
            consistency checks.
        y
            Two-dimensional numeric target matrix with shape
            ``(n_samples, n_tasks)``. Dataframes are accepted; string target
            columns are used as task names when ``task_names`` is not supplied.

        Notes
        -----
        This first multitask estimator expects complete target observations.
        Missing targets are rejected or row-dropped according to ``missing``;
        target values are never imputed.
        """
        X_checked, y_checked, task_names = _validate_multitask_fit_input(self, X, y)

        _seed_torch(self.random_state)
        dtype = _resolve_torch_dtype(self.dtype)

        self.result_ = fit_multitask_gpytorch_gpr(
            X_checked,
            y_checked,
            task_names=task_names,
            task_covar_rank=self.task_covar_rank,
            kernel=self.kernel,
            ard=self.ard,
            lr=self.lr,
            training_iter=self.training_iter,
            initial_noise=self.initial_noise,
            initial_task_noises=self.initial_task_noises,
            standardize_y=self.standardize_y,
            device=self.device,
            dtype=dtype,
            verbose=self.verbose,
            log_every=self.log_every,
        )
        self.model_ = self.result_.model
        self.likelihood_ = self.result_.likelihood
        self.loss_history_ = list(self.result_.loss_history)
        self.target_mean_ = self.result_.target_mean.copy()
        self.target_std_ = self.result_.target_std.copy()
        self.task_names_ = self.result_.task_names
        self.n_tasks_in_ = self.result_.num_tasks
        self.n_outputs_ = self.result_.num_tasks
        return self

    def predict_distribution(
        self,
        X,
        *,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool | None = None,
    ) -> MultitaskGPyTorchPrediction:
        """Return per-task predictive means, uncertainties, and intervals."""
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


def _feature_names_from_input(X) -> np.ndarray | None:
    if not hasattr(X, "columns"):
        return None
    names = np.asarray(X.columns, dtype=object)
    if not all(isinstance(name, str) for name in names):
        return None
    return names


def _validate_fit_input(estimator, X, y) -> tuple[np.ndarray, np.ndarray]:
    _clear_missing_state(estimator)
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None")
    policy = _validate_missing_policy(estimator.missing)
    _validate_imputation_strategy(estimator.imputation_strategy)

    if validate_data is not None:
        X_checked = validate_data(
            estimator,
            X,
            reset=True,
            ensure_2d=True,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
    else:
        feature_names = _feature_names_from_input(X)
        X_checked = check_array(
            X,
            ensure_2d=True,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        estimator.n_features_in_ = X_checked.shape[1]
        if feature_names is not None:
            if len(feature_names) != estimator.n_features_in_:
                raise ValueError("Number of dataframe columns does not match validated features")
            estimator.feature_names_in_ = feature_names
        elif hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")

    y_checked = check_array(
        y,
        ensure_2d=False,
        dtype="numeric",
        ensure_all_finite="allow-nan",
        input_name="y",
    )
    y_checked = column_or_1d(y_checked, warn=True)

    check_consistent_length(X_checked, y_checked)
    return _apply_fit_missing_policy(estimator, X_checked, y_checked, policy)


def _validate_multitask_fit_input(estimator, X, y) -> tuple[np.ndarray, np.ndarray, tuple[str, ...] | None]:
    _clear_missing_state(estimator)
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None")
    policy = _validate_missing_policy(estimator.missing)
    _validate_imputation_strategy(estimator.imputation_strategy)

    if validate_data is not None:
        X_checked = validate_data(
            estimator,
            X,
            reset=True,
            ensure_2d=True,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
    else:
        feature_names = _feature_names_from_input(X)
        X_checked = check_array(
            X,
            ensure_2d=True,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        estimator.n_features_in_ = X_checked.shape[1]
        if feature_names is not None:
            if len(feature_names) != estimator.n_features_in_:
                raise ValueError("Number of dataframe columns does not match validated features")
            estimator.feature_names_in_ = feature_names
        elif hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")

    inferred_task_names = _task_names_from_target_input(y)
    if inferred_task_names is not None:
        estimator.target_names_in_ = np.asarray(inferred_task_names, dtype=object)
    elif hasattr(estimator, "target_names_in_"):
        delattr(estimator, "target_names_in_")

    y_checked = check_array(
        y,
        ensure_2d=True,
        dtype="numeric",
        ensure_all_finite="allow-nan",
        input_name="y",
    )
    if y_checked.shape[1] < 2:
        raise ValueError(
            "MultitaskGPRRegressor requires at least two target columns; "
            f"got {y_checked.shape[1]}"
        )

    check_consistent_length(X_checked, y_checked)
    X_checked, y_checked = _apply_fit_missing_policy(estimator, X_checked, y_checked, policy)
    return X_checked, y_checked, _resolve_estimator_task_names(estimator, inferred_task_names)


def _validate_predict_input(estimator, X) -> np.ndarray:
    policy = _validate_missing_policy(estimator.missing)
    X_checked = _validate_prediction_features(
        estimator,
        X,
        allow_nan=policy == "impute",
    )
    return _apply_predict_missing_policy(estimator, X_checked, policy)


def _validate_prediction_features(estimator, X, *, allow_nan: bool) -> np.ndarray:
    if validate_data is not None:
        return validate_data(
            estimator,
            X,
            reset=False,
            ensure_2d=True,
            dtype="numeric",
            ensure_all_finite="allow-nan" if allow_nan else True,
            ensure_min_samples=1,
        )

    if hasattr(estimator, "feature_names_in_"):
        input_feature_names = _feature_names_from_input(X)
        if input_feature_names is not None and not np.array_equal(
            input_feature_names,
            estimator.feature_names_in_,
        ):
            raise ValueError("Prediction features must match the fitted feature names and order")

    X_checked = check_array(
        X,
        ensure_2d=True,
        dtype="numeric",
        ensure_min_samples=1,
        ensure_all_finite="allow-nan" if allow_nan else True,
    )
    if X_checked.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X_checked.shape[1]} features, but this estimator was fitted with "
            f"{estimator.n_features_in_} features"
        )
    return X_checked


def _clear_missing_state(estimator) -> None:
    for attr in (
        "feature_imputer_",
        "feature_imputation_statistics_",
        "missing_report_",
    ):
        if hasattr(estimator, attr):
            delattr(estimator, attr)


def _validate_missing_policy(policy: str) -> str:
    normalized = str(policy).lower()
    valid = {"error", "drop", "impute"}
    if normalized not in valid:
        raise ValueError("missing must be one of: 'error', 'drop', or 'impute'")
    return normalized


def _validate_imputation_strategy(strategy: str) -> str:
    normalized = str(strategy).lower()
    valid = {"mean", "median", "most_frequent", "constant"}
    if normalized not in valid:
        raise ValueError(
            "imputation_strategy must be one of: "
            "'mean', 'median', 'most_frequent', or 'constant'"
        )
    return normalized


def _apply_fit_missing_policy(
    estimator,
    X_checked: np.ndarray,
    y_checked: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    feature_missing_mask = np.isnan(X_checked).any(axis=1)
    target_missing_mask = _target_missing_row_mask(y_checked)
    any_missing_mask = feature_missing_mask | target_missing_mask

    if policy == "error" and np.any(any_missing_mask):
        report = _make_missing_report(
            estimator,
            stage="fit",
            policy=policy,
            X_checked=X_checked,
            y_checked=y_checked,
            output_rows=X_checked.shape[0],
        )
        estimator.missing_report_ = report
        raise ValueError(
            "Input contains NaN/missing values. Set missing='drop' to remove "
            "incomplete training rows or missing='impute' to impute missing "
            f"features. Missing-value report: {report.to_dict()}"
        )

    if policy == "drop":
        keep_mask = ~any_missing_mask
        X_out = X_checked[keep_mask]
        y_out = y_checked[keep_mask]
        if X_out.shape[0] == 0:
            raise ValueError("No training rows remain after missing='drop'")
        estimator.missing_report_ = _make_missing_report(
            estimator,
            stage="fit",
            policy=policy,
            X_checked=X_checked,
            y_checked=y_checked,
            output_rows=X_out.shape[0],
        )
        return X_out, y_out

    if policy == "impute":
        keep_mask = ~target_missing_mask
        X_to_impute = X_checked[keep_mask]
        y_out = y_checked[keep_mask]
        if X_to_impute.shape[0] == 0:
            raise ValueError("No training rows remain after dropping missing targets")

        strategy = _validate_imputation_strategy(estimator.imputation_strategy)
        _raise_for_unimputable_columns(estimator, X_to_impute, strategy)
        imputer = SimpleImputer(
            strategy=strategy,
            fill_value=estimator.imputation_fill_value,
            keep_empty_features=True,
        )
        X_out = imputer.fit_transform(X_to_impute)
        estimator.feature_imputer_ = imputer
        estimator.feature_imputation_statistics_ = np.asarray(imputer.statistics_).copy()
        estimator.missing_report_ = _make_missing_report(
            estimator,
            stage="fit",
            policy=policy,
            X_checked=X_checked,
            y_checked=y_checked,
            output_rows=X_out.shape[0],
            imputed_features=_imputed_feature_names(estimator, X_to_impute),
            imputation_strategy=strategy,
        )
        return X_out, y_out

    estimator.missing_report_ = _make_missing_report(
        estimator,
        stage="fit",
        policy=policy,
        X_checked=X_checked,
        y_checked=y_checked,
        output_rows=X_checked.shape[0],
    )
    return X_checked, y_checked


def _apply_predict_missing_policy(estimator, X_checked: np.ndarray, policy: str) -> np.ndarray:
    if policy != "impute":
        return X_checked

    check_is_fitted(estimator, "feature_imputer_")
    return estimator.feature_imputer_.transform(X_checked)


def _make_missing_report(
    estimator,
    *,
    stage: str,
    policy: str,
    X_checked: np.ndarray,
    y_checked: np.ndarray | None,
    output_rows: int,
    imputed_features: tuple[str, ...] = (),
    imputation_strategy: str | None = None,
) -> MissingValueReport:
    feature_missing_mask = np.isnan(X_checked).any(axis=1)
    target_missing_mask = np.zeros(X_checked.shape[0], dtype=bool)
    if y_checked is not None:
        target_missing_mask = _target_missing_row_mask(y_checked)

    return MissingValueReport(
        stage=stage,
        policy=policy,
        input_rows=int(X_checked.shape[0]),
        output_rows=int(output_rows),
        dropped_rows=int(X_checked.shape[0] - output_rows),
        rows_with_missing_features=int(feature_missing_mask.sum()),
        rows_with_missing_target=int(target_missing_mask.sum()),
        feature_missing_counts=_feature_missing_counts(estimator, X_checked),
        imputed_features=tuple(imputed_features),
        imputation_strategy=imputation_strategy,
    )


def _feature_missing_counts(estimator, X_checked: np.ndarray) -> dict[str, int]:
    labels = _feature_labels(estimator, X_checked.shape[1])
    counts = np.isnan(X_checked).sum(axis=0)
    return {
        label: int(count)
        for label, count in zip(labels, counts, strict=True)
        if int(count) > 0
    }


def _feature_labels(estimator, n_features: int) -> list[str]:
    if hasattr(estimator, "feature_names_in_"):
        return [str(name) for name in estimator.feature_names_in_]
    return [f"feature_{index}" for index in range(n_features)]


def _target_missing_row_mask(y_checked: np.ndarray) -> np.ndarray:
    missing = np.isnan(y_checked)
    if missing.ndim == 1:
        return missing
    return missing.any(axis=1)


def _task_names_from_target_input(y) -> tuple[str, ...] | None:
    if not hasattr(y, "columns"):
        return None
    if not all(isinstance(name, str) and name for name in y.columns):
        return None
    return tuple(y.columns)


def _resolve_estimator_task_names(
    estimator,
    inferred_task_names: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if estimator.task_names is not None:
        return tuple(str(name) for name in estimator.task_names)
    return inferred_task_names


def _imputed_feature_names(estimator, X_checked: np.ndarray) -> tuple[str, ...]:
    labels = _feature_labels(estimator, X_checked.shape[1])
    counts = np.isnan(X_checked).sum(axis=0)
    return tuple(
        label for label, count in zip(labels, counts, strict=True) if int(count) > 0
    )


def _raise_for_unimputable_columns(
    estimator,
    X_checked: np.ndarray,
    strategy: str,
) -> None:
    if strategy == "constant":
        return
    all_missing_mask = np.isnan(X_checked).all(axis=0)
    if not np.any(all_missing_mask):
        return
    labels = _feature_labels(estimator, X_checked.shape[1])
    missing_columns = [
        label
        for label, is_all_missing in zip(labels, all_missing_mask, strict=True)
        if is_all_missing
    ]
    raise ValueError(
        "Cannot impute feature columns with all values missing using "
        f"imputation_strategy={strategy!r}: {missing_columns}. Remove these "
        "columns or use imputation_strategy='constant'."
    )


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
