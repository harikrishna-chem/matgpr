from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split

from .metrics import regression_metrics, train_test_regression_metrics
from .reporting import decompose_multifidelity_prediction
from .uncertainty import prediction_interval_bounds, uncertainty_diagnostics

__all__ = [
    "CoKrigingTrainTestValidationResult",
    "CrossValidationResult",
    "LearningCurveResult",
    "SparseMultitaskTrainTestValidationResult",
    "MultitaskTrainTestValidationResult",
    "TrainTestValidationResult",
    "evaluate_cokriging_train_test_split",
    "cross_validate_regressor",
    "evaluate_multitask_train_test_split",
    "evaluate_sparse_multitask_train_test_split",
    "evaluate_train_test_split",
    "learning_curve",
    "multifidelity_learning_curve",
    "summarize_multitask_predictions",
    "summarize_sparse_multitask_predictions",
]


@dataclass(frozen=True)
class TrainTestValidationResult:
    """Result from a single train/test validation run.

    Attributes
    ----------
    model_name
        Human-readable model label used in reports.
    fitted_estimator
        Estimator fitted on the training split.
    metrics
        Flat dictionary with train/test regression metrics and, when available,
        uncertainty diagnostics.
    predictions
        Prediction table with one row per train and test sample.
    train_indices, test_indices
        Integer sample positions used for the split.
    """

    model_name: str
    fitted_estimator: object
    metrics: Mapping[str, float]
    predictions: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray

    def metrics_frame(self) -> pd.DataFrame:
        """Return metrics as a one-row dataframe."""
        return pd.DataFrame([{**{"model": self.model_name}, **dict(self.metrics)}])

    @property
    def train_predictions(self) -> pd.DataFrame:
        """Predictions for training samples."""
        return self.predictions[self.predictions["split"] == "train"].reset_index(drop=True)

    @property
    def test_predictions(self) -> pd.DataFrame:
        """Predictions for held-out test samples."""
        return self.predictions[self.predictions["split"] == "test"].reset_index(drop=True)


@dataclass(frozen=True)
class CoKrigingTrainTestValidationResult:
    """Result from target-fidelity validation of a row-wise co-kriging model.

    ``train_indices`` and ``test_indices`` refer only to rows at
    ``target_fidelity``. ``fit_indices`` contains every row used to fit the
    estimator: all lower-fidelity rows plus the selected target-fidelity
    training rows.
    """

    model_name: str
    fitted_estimator: object
    metrics: Mapping[str, float]
    predictions: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray
    fit_indices: np.ndarray
    lower_fidelity_indices: np.ndarray
    target_fidelity: str
    low_fidelity: str
    rho: float | None = None

    def metrics_frame(self) -> pd.DataFrame:
        """Return metrics and validation protocol metadata as a one-row dataframe."""
        row = {
            "model": self.model_name,
            "target_fidelity": self.target_fidelity,
            "low_fidelity": self.low_fidelity,
            "n_fit": len(self.fit_indices),
            "n_target_train": len(self.train_indices),
            "n_target_test": len(self.test_indices),
            "n_lower_fidelity": len(self.lower_fidelity_indices),
        }
        if self.rho is not None:
            row["rho"] = self.rho
        row.update(dict(self.metrics))
        return pd.DataFrame([row])

    @property
    def train_predictions(self) -> pd.DataFrame:
        """Target-fidelity predictions for training samples."""
        return self.predictions[self.predictions["split"] == "train"].reset_index(drop=True)

    @property
    def test_predictions(self) -> pd.DataFrame:
        """Target-fidelity predictions for held-out test samples."""
        return self.predictions[self.predictions["split"] == "test"].reset_index(drop=True)


@dataclass(frozen=True)
class CrossValidationResult:
    """Result from k-fold or splitter-based cross-validation."""

    model_name: str
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame

    def summary(self, metric_columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Summarize fold metrics with mean, standard deviation, min, and max."""
        return _summarize_metrics(self.fold_metrics, group_by=("model",), metric_columns=metric_columns)


@dataclass(frozen=True)
class LearningCurveResult:
    """Result from learning-curve evaluation across train-set sizes."""

    model_name: str
    runs: pd.DataFrame
    predictions: pd.DataFrame | None = None
    metric_names: Sequence[str] = ("RMSE", "R2", "MAE", "r")
    metric_splits: Sequence[str] = ("test",)

    def summary(
        self,
        metric_columns: Sequence[str] | None = None,
        *,
        metrics: Sequence[str] | str | None = None,
        splits: Sequence[str] | str | None = None,
    ) -> pd.DataFrame:
        """Summarize repeated runs by model and train-set size."""
        if metric_columns is None:
            metric_columns = _metric_columns(
                self.runs,
                metrics=self.metric_names if metrics is None else metrics,
                splits=self.metric_splits if splits is None else splits,
            )
        return _summarize_metrics(
            self.runs,
            group_by=("model", "train_size", "train_size_percent", "n_train"),
            metric_columns=metric_columns,
        )


@dataclass(frozen=True)
class MultitaskTrainTestValidationResult:
    """Result from a multitask train/test validation run.

    Attributes
    ----------
    model_name
        Human-readable model label used in reports.
    fitted_estimator
        Estimator fitted on the training split.
    task_metrics
        Per-task metrics with one row per split and task.
    predictions
        Long-form prediction table with one row per sample, split, and task.
    train_indices, test_indices
        Integer sample positions used for the split.
    """

    model_name: str
    fitted_estimator: object
    task_metrics: pd.DataFrame
    predictions: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray

    def metrics_frame(self) -> pd.DataFrame:
        """Return the per-task metrics dataframe."""
        return self.task_metrics.copy()

    def summary(
        self,
        metric_columns: Sequence[str] | None = None,
        *,
        group_by: Sequence[str] = ("model", "split"),
    ) -> pd.DataFrame:
        """Summarize per-task metrics across tasks for each split."""
        return _summarize_metrics(
            self.task_metrics,
            group_by=group_by,
            metric_columns=metric_columns,
        )

    @property
    def train_predictions(self) -> pd.DataFrame:
        """Predictions for training samples."""
        return self.predictions[self.predictions["split"] == "train"].reset_index(drop=True)

    @property
    def test_predictions(self) -> pd.DataFrame:
        """Predictions for held-out test samples."""
        return self.predictions[self.predictions["split"] == "test"].reset_index(drop=True)


@dataclass(frozen=True)
class SparseMultitaskTrainTestValidationResult:
    """Result from a sparse multitask train/test validation run.

    The prediction table keeps one row per sample and task, including entries
    where the target was unobserved. Use ``observed_predictions`` for
    parity-plot-ready rows with finite ground-truth values.
    """

    model_name: str
    fitted_estimator: object
    task_metrics: pd.DataFrame
    predictions: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray

    def metrics_frame(self) -> pd.DataFrame:
        """Return the per-task sparse metrics dataframe."""
        return self.task_metrics.copy()

    def summary(
        self,
        metric_columns: Sequence[str] | None = None,
        *,
        group_by: Sequence[str] = ("model", "split"),
    ) -> pd.DataFrame:
        """Summarize sparse per-task metrics across tasks for each split."""
        return _summarize_metrics(
            self.task_metrics,
            group_by=group_by,
            metric_columns=metric_columns,
        )

    @property
    def train_predictions(self) -> pd.DataFrame:
        """Predictions for training samples."""
        return self.predictions[self.predictions["split"] == "train"].reset_index(drop=True)

    @property
    def test_predictions(self) -> pd.DataFrame:
        """Predictions for held-out test samples."""
        return self.predictions[self.predictions["split"] == "test"].reset_index(drop=True)

    @property
    def observed_predictions(self) -> pd.DataFrame:
        """Prediction rows whose target values are observed."""
        return self.predictions[self.predictions["observed"]].reset_index(drop=True)


def evaluate_train_test_split(
    estimator,
    X,
    y,
    *,
    train_indices: Sequence[int] | None = None,
    test_indices: Sequence[int] | None = None,
    test_size: float | int = 0.2,
    train_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
    model_name: str | None = None,
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
) -> TrainTestValidationResult:
    """Fit an estimator on one train/test split and return report-ready outputs.

    This is the reusable package equivalent of the validation block commonly
    repeated in notebooks: split, fit, predict train/test values, calculate
    metrics, and store a parity-plot-ready prediction table.
    """
    n_samples = _validate_xy_length(X, y)
    labels = _sample_labels(y, n_samples)

    if train_indices is None and test_indices is None:
        train_indices, test_indices = train_test_split(
            np.arange(n_samples),
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    elif train_indices is None or test_indices is None:
        raise ValueError("train_indices and test_indices must be supplied together")
    else:
        train_indices = _to_index_array(train_indices, "train_indices", n_samples)
        test_indices = _to_index_array(test_indices, "test_indices", n_samples)

    result = _fit_evaluate_indices(
        estimator,
        X,
        y,
        train_indices=np.asarray(train_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        labels=labels,
        model_name=model_name or _estimator_name(estimator),
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
        fit_params=fit_params,
        random_state=random_state,
        context={},
    )
    return TrainTestValidationResult(
        model_name=result["model_name"],
        fitted_estimator=result["estimator"],
        metrics=result["metrics"],
        predictions=result["predictions"],
        train_indices=np.asarray(train_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
    )


def evaluate_cokriging_train_test_split(
    estimator,
    X,
    y,
    fidelity,
    *,
    target_fidelity: str | None = None,
    low_fidelity: str | None = None,
    fidelity_order: Sequence[str] | None = None,
    sample_id=None,
    train_indices: Sequence[int] | None = None,
    test_indices: Sequence[int] | None = None,
    test_size: float | int = 0.2,
    train_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
    model_name: str | None = None,
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
) -> CoKrigingTrainTestValidationResult:
    """Validate a two-level co-kriging estimator on target-fidelity rows.

    The split is performed only among rows labeled ``target_fidelity``. All
    lower-fidelity rows are retained for fitting in every split, which matches
    the usual low-data protocol for simulation-plus-experiment materials
    datasets. The returned prediction table is parity-plot-ready and preserves
    co-kriging component columns such as ``scaled_low_fidelity_pred`` and
    ``discrepancy_pred`` when the estimator exposes them.
    """
    n_samples = _validate_xy_length(X, y)
    labels = _sample_labels(y, n_samples)
    fidelity_labels = _to_fidelity_label_array(fidelity, n_samples=n_samples)
    sample_id_array = _resolve_optional_row_values(sample_id, n_samples=n_samples, name="sample_id")
    fidelity_names, target_name, low_name = _resolve_cokriging_validation_levels(
        fidelity_labels,
        estimator=estimator,
        target_fidelity=target_fidelity,
        low_fidelity=low_fidelity,
        fidelity_order=fidelity_order,
    )

    target_mask = fidelity_labels == target_name
    target_positions = np.flatnonzero(target_mask)
    if target_positions.size < 2:
        raise ValueError("At least two target-fidelity rows are required for validation")

    if train_indices is None and test_indices is None:
        train_indices, test_indices = train_test_split(
            target_positions,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    elif train_indices is None or test_indices is None:
        raise ValueError("train_indices and test_indices must be supplied together")
    else:
        train_indices = _to_index_array(train_indices, "train_indices", n_samples)
        test_indices = _to_index_array(test_indices, "test_indices", n_samples)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    _validate_split_indices(train_indices, test_indices, n_samples)
    _validate_target_fidelity_indices(
        train_indices,
        target_mask=target_mask,
        name="train_indices",
        target_fidelity=target_name,
    )
    _validate_target_fidelity_indices(
        test_indices,
        target_mask=target_mask,
        name="test_indices",
        target_fidelity=target_name,
    )

    lower_mask = fidelity_labels != target_name
    lower_fidelity_indices = np.flatnonzero(lower_mask)
    if lower_fidelity_indices.size == 0:
        raise ValueError("Co-kriging validation requires at least one lower-fidelity row")
    fit_mask = lower_mask.copy()
    fit_mask[train_indices] = True
    fit_indices = np.flatnonzero(fit_mask)

    result = _fit_evaluate_cokriging_indices(
        estimator,
        X,
        y,
        fidelity_labels=fidelity_labels,
        sample_id=sample_id_array,
        fit_indices=fit_indices,
        train_indices=train_indices,
        test_indices=test_indices,
        labels=labels,
        fidelity_names=fidelity_names,
        target_fidelity=target_name,
        low_fidelity=low_name,
        model_name=model_name or _estimator_name(estimator),
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
        fit_params=fit_params,
        random_state=random_state,
        context={
            "target_fidelity": target_name,
            "low_fidelity": low_name,
            "n_lower_fidelity": len(lower_fidelity_indices),
        },
    )

    return CoKrigingTrainTestValidationResult(
        model_name=result["model_name"],
        fitted_estimator=result["estimator"],
        metrics=result["metrics"],
        predictions=result["predictions"],
        train_indices=train_indices,
        test_indices=test_indices,
        fit_indices=fit_indices,
        lower_fidelity_indices=lower_fidelity_indices,
        target_fidelity=target_name,
        low_fidelity=low_name,
        rho=result["rho"],
    )


def summarize_multitask_predictions(
    y_true,
    y_pred,
    y_std=None,
    *,
    task_names: Sequence[str] | None = None,
    model_name: str = "model",
    split: str = "test",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Return per-task regression and uncertainty metrics.

    Parameters
    ----------
    y_true
        True target matrix with shape ``(n_samples, n_tasks)``. Dataframes are
        accepted; their columns are used as task names when ``task_names`` is
        not supplied.
    y_pred
        Predicted target matrix with the same shape as ``y_true``. Objects with
        ``mean``, ``std``, and ``task_names`` attributes, such as
        :class:`matgpr.MultitaskGPyTorchPrediction`, are also accepted.
    y_std
        Optional predictive standard deviations with the same shape as
        ``y_true``. If omitted and ``y_pred`` has a ``std`` attribute, that
        attribute is used.
    task_names
        Optional task names in target-column order.
    model_name, split
        Labels stored in the returned dataframe for report tables.
    confidence_level
        Gaussian interval level used for uncertainty coverage diagnostics.

    Returns
    -------
    pandas.DataFrame
        One row per task with RMSE, MAE, R2, Pearson ``r``, sample count, and
        uncertainty diagnostics when predictive standard deviations are
        available.
    """
    prediction_task_names = getattr(y_pred, "task_names", None)
    if y_std is None and _is_prediction_container(y_pred) and hasattr(y_pred, "std"):
        y_std = getattr(y_pred, "std")
    if _is_prediction_container(y_pred):
        y_pred = getattr(y_pred, "mean")

    y_true_array = _to_2d_float(y_true, "y_true")
    y_pred_array = _to_2d_float(y_pred, "y_pred")
    _validate_same_2d_shape(y_true_array, y_pred_array, "y_true", "y_pred")
    y_std_array = _to_optional_2d_std(y_std) if y_std is not None else None
    if y_std_array is not None:
        _validate_same_2d_shape(y_true_array, y_std_array, "y_true", "y_std")

    resolved_task_names = _resolve_multitask_names(
        task_names=task_names,
        n_tasks=y_true_array.shape[1],
        y=y_true,
        prediction_task_names=prediction_task_names,
    )
    rows = []
    for task_index, task_name in enumerate(resolved_task_names):
        metrics = regression_metrics(
            y_true_array[:, task_index],
            y_pred_array[:, task_index],
        )
        row = {
            "model": str(model_name),
            "split": str(split),
            "task": task_name,
            "task_index": task_index,
            "n_samples": y_true_array.shape[0],
            **metrics,
        }
        if y_std_array is not None:
            row.update(
                uncertainty_diagnostics(
                    y_true_array[:, task_index],
                    y_pred_array[:, task_index],
                    y_std_array[:, task_index],
                    confidence_level=confidence_level,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sparse_multitask_predictions(
    y_true,
    y_pred,
    y_std=None,
    *,
    task_names: Sequence[str] | None = None,
    model_name: str = "model",
    split: str = "test",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Return per-task metrics for sparse multitask targets.

    ``NaN`` values in ``y_true`` are treated as unobserved task entries and are
    ignored for regression and uncertainty metrics. The returned dataframe
    keeps the total row count, observed count, and missing fraction for each
    task so sparse validation protocols are transparent in reports.
    """
    prediction_task_names = getattr(y_pred, "task_names", None)
    if y_std is None and _is_prediction_container(y_pred) and hasattr(y_pred, "std"):
        y_std = getattr(y_pred, "std")
    if _is_prediction_container(y_pred):
        y_pred = getattr(y_pred, "mean")

    y_true_array = _to_2d_float_allow_nan(y_true, "y_true")
    y_pred_array = _to_2d_float(y_pred, "y_pred")
    _validate_same_2d_shape(y_true_array, y_pred_array, "y_true", "y_pred")
    y_std_array = _to_optional_2d_std(y_std) if y_std is not None else None
    if y_std_array is not None:
        _validate_same_2d_shape(y_true_array, y_std_array, "y_true", "y_std")

    resolved_task_names = _resolve_multitask_names(
        task_names=task_names,
        n_tasks=y_true_array.shape[1],
        y=y_true,
        prediction_task_names=prediction_task_names,
    )
    rows = []
    n_samples = y_true_array.shape[0]
    for task_index, task_name in enumerate(resolved_task_names):
        observed_mask = np.isfinite(y_true_array[:, task_index])
        n_observed = int(observed_mask.sum())
        n_missing = int(n_samples - n_observed)
        metrics = _observed_regression_metrics(
            y_true_array[observed_mask, task_index],
            y_pred_array[observed_mask, task_index],
        )
        row = {
            "model": str(model_name),
            "split": str(split),
            "task": task_name,
            "task_index": task_index,
            "n_samples": n_samples,
            "n_observed": n_observed,
            "n_missing": n_missing,
            "observed_fraction": n_observed / n_samples,
            "missing_fraction": n_missing / n_samples,
            **metrics,
        }
        if y_std_array is not None:
            if n_observed:
                row.update(
                    uncertainty_diagnostics(
                        y_true_array[observed_mask, task_index],
                        y_pred_array[observed_mask, task_index],
                        y_std_array[observed_mask, task_index],
                        confidence_level=confidence_level,
                    )
                )
            else:
                row.update(_empty_uncertainty_diagnostics(confidence_level))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_multitask_train_test_split(
    estimator,
    X,
    y,
    *,
    train_indices: Sequence[int] | None = None,
    test_indices: Sequence[int] | None = None,
    test_size: float | int = 0.2,
    train_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
    model_name: str | None = None,
    task_names: Sequence[str] | None = None,
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
) -> MultitaskTrainTestValidationResult:
    """Fit a multitask estimator on one split and return per-task summaries.

    This is the multitask counterpart of :func:`evaluate_train_test_split`.
    It keeps validation outputs in report-ready form: per-task metrics for the
    train and held-out test splits plus a long-form parity-plot table with
    predictive uncertainty when the estimator provides it.
    """
    n_samples, _ = _validate_multitask_xy_length(X, y)
    labels = _sample_labels(y, n_samples)

    if train_indices is None and test_indices is None:
        train_indices, test_indices = train_test_split(
            np.arange(n_samples),
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    elif train_indices is None or test_indices is None:
        raise ValueError("train_indices and test_indices must be supplied together")
    else:
        train_indices = _to_index_array(train_indices, "train_indices", n_samples)
        test_indices = _to_index_array(test_indices, "test_indices", n_samples)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    _validate_split_indices(train_indices, test_indices, n_samples)

    fitted_estimator = _clone_estimator(estimator, random_state=random_state)
    X_train = _safe_index(X, train_indices)
    X_test = _safe_index(X, test_indices)
    y_train = _safe_index(y, train_indices)
    y_test = _safe_index(y, test_indices)
    model_name = model_name or _estimator_name(estimator)

    fitted_estimator.fit(X_train, y_train, **dict(fit_params or {}))
    y_train_pred, y_train_std, train_prediction_task_names = _predict_multitask_estimator(
        fitted_estimator,
        X_train,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
    )
    y_test_pred, y_test_std, test_prediction_task_names = _predict_multitask_estimator(
        fitted_estimator,
        X_test,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
    )
    fitted_task_names = getattr(fitted_estimator, "task_names_", None)
    resolved_task_names = _resolve_multitask_names(
        task_names=task_names,
        n_tasks=_to_2d_float(y_train, "y_train").shape[1],
        y=y,
        prediction_task_names=_first_available_task_names(
            fitted_task_names,
            train_prediction_task_names,
            test_prediction_task_names,
        ),
    )

    train_metrics = summarize_multitask_predictions(
        y_train,
        y_train_pred,
        y_train_std,
        task_names=resolved_task_names,
        model_name=model_name,
        split="train",
        confidence_level=confidence_level,
    )
    test_metrics = summarize_multitask_predictions(
        y_test,
        y_test_pred,
        y_test_std,
        task_names=resolved_task_names,
        model_name=model_name,
        split="test",
        confidence_level=confidence_level,
    )
    predictions = pd.concat(
        [
            _multitask_prediction_frame(
                model_name,
                "train",
                train_indices,
                labels,
                y_train,
                y_train_pred,
                y_train_std,
                task_names=resolved_task_names,
                confidence_level=confidence_level,
            ),
            _multitask_prediction_frame(
                model_name,
                "test",
                test_indices,
                labels,
                y_test,
                y_test_pred,
                y_test_std,
                task_names=resolved_task_names,
                confidence_level=confidence_level,
            ),
        ],
        ignore_index=True,
    )
    return MultitaskTrainTestValidationResult(
        model_name=model_name,
        fitted_estimator=fitted_estimator,
        task_metrics=pd.concat([train_metrics, test_metrics], ignore_index=True),
        predictions=predictions,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def evaluate_sparse_multitask_train_test_split(
    estimator,
    X,
    y,
    *,
    train_indices: Sequence[int] | None = None,
    test_indices: Sequence[int] | None = None,
    test_size: float | int = 0.2,
    train_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
    model_name: str | None = None,
    task_names: Sequence[str] | None = None,
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
) -> SparseMultitaskTrainTestValidationResult:
    """Fit a sparse multitask estimator and evaluate observed target entries.

    This function is intended for target matrices where ``NaN`` means a task
    was not measured for that sample. Fitting receives the sparse target
    matrix unchanged. Metrics are calculated only over finite target entries,
    while the prediction table retains all sample-task predictions with an
    ``observed`` flag for filtering parity plots and diagnostics.
    """
    n_samples, _ = _validate_sparse_multitask_xy_length(X, y)
    labels = _sample_labels(y, n_samples)

    if train_indices is None and test_indices is None:
        train_indices, test_indices = train_test_split(
            np.arange(n_samples),
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
    elif train_indices is None or test_indices is None:
        raise ValueError("train_indices and test_indices must be supplied together")
    else:
        train_indices = _to_index_array(train_indices, "train_indices", n_samples)
        test_indices = _to_index_array(test_indices, "test_indices", n_samples)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    _validate_split_indices(train_indices, test_indices, n_samples)

    fitted_estimator = _clone_estimator(estimator, random_state=random_state)
    X_train = _safe_index(X, train_indices)
    X_test = _safe_index(X, test_indices)
    y_train = _safe_index(y, train_indices)
    y_test = _safe_index(y, test_indices)
    model_name = model_name or _estimator_name(estimator)

    fitted_estimator.fit(X_train, y_train, **dict(fit_params or {}))
    y_train_pred, y_train_std, train_prediction_task_names = _predict_multitask_estimator(
        fitted_estimator,
        X_train,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
    )
    y_test_pred, y_test_std, test_prediction_task_names = _predict_multitask_estimator(
        fitted_estimator,
        X_test,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
    )
    fitted_task_names = getattr(fitted_estimator, "task_names_", None)
    resolved_task_names = _resolve_multitask_names(
        task_names=task_names,
        n_tasks=_to_2d_float_allow_nan(y_train, "y_train").shape[1],
        y=y,
        prediction_task_names=_first_available_task_names(
            fitted_task_names,
            train_prediction_task_names,
            test_prediction_task_names,
        ),
    )

    train_metrics = summarize_sparse_multitask_predictions(
        y_train,
        y_train_pred,
        y_train_std,
        task_names=resolved_task_names,
        model_name=model_name,
        split="train",
        confidence_level=confidence_level,
    )
    test_metrics = summarize_sparse_multitask_predictions(
        y_test,
        y_test_pred,
        y_test_std,
        task_names=resolved_task_names,
        model_name=model_name,
        split="test",
        confidence_level=confidence_level,
    )
    predictions = pd.concat(
        [
            _sparse_multitask_prediction_frame(
                model_name,
                "train",
                train_indices,
                labels,
                y_train,
                y_train_pred,
                y_train_std,
                task_names=resolved_task_names,
                confidence_level=confidence_level,
            ),
            _sparse_multitask_prediction_frame(
                model_name,
                "test",
                test_indices,
                labels,
                y_test,
                y_test_pred,
                y_test_std,
                task_names=resolved_task_names,
                confidence_level=confidence_level,
            ),
        ],
        ignore_index=True,
    )
    return SparseMultitaskTrainTestValidationResult(
        model_name=model_name,
        fitted_estimator=fitted_estimator,
        task_metrics=pd.concat([train_metrics, test_metrics], ignore_index=True),
        predictions=predictions,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def cross_validate_regressor(
    estimator,
    X,
    y,
    *,
    cv: int | object = 5,
    shuffle: bool = True,
    random_state: int | None = None,
    model_name: str | None = None,
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
    store_train_predictions: bool = False,
) -> CrossValidationResult:
    """Run cross-validation and return fold metrics plus prediction rows.

    Parameters
    ----------
    estimator
        Any scikit-learn-style regressor with ``fit`` and ``predict``.
    X, y
        Feature matrix and target vector.
    cv
        Number of folds or any splitter object with ``split(X, y)``.
    store_train_predictions
        If ``True``, include train predictions for each fold. Test predictions
        are always stored as out-of-fold predictions.
    """
    n_samples = _validate_xy_length(X, y)
    labels = _sample_labels(y, n_samples)
    resolved_model_name = model_name or _estimator_name(estimator)
    metric_rows = []
    prediction_tables = []

    for fold, (train_indices, test_indices) in enumerate(
        _iter_cv_splits(cv, X, y, shuffle=shuffle, random_state=random_state),
        start=1,
    ):
        seed = _derive_seed(random_state, fold)
        result = _fit_evaluate_indices(
            estimator,
            X,
            y,
            train_indices=np.asarray(train_indices, dtype=int),
            test_indices=np.asarray(test_indices, dtype=int),
            labels=labels,
            model_name=resolved_model_name,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
            fit_params=fit_params,
            random_state=seed,
            context={"fold": fold},
        )
        metric_rows.append(
            {
                "model": resolved_model_name,
                "fold": fold,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                **result["metrics"],
            }
        )
        predictions = result["predictions"]
        if not store_train_predictions:
            predictions = predictions[predictions["split"] == "test"]
        prediction_tables.append(predictions)

    return CrossValidationResult(
        model_name=resolved_model_name,
        fold_metrics=pd.DataFrame(metric_rows),
        predictions=_concat_prediction_tables(prediction_tables),
    )


def learning_curve(
    estimators,
    X,
    y,
    *,
    train_sizes: Sequence[float | int] | None = None,
    train_size_start: float | int = 10,
    train_size_stop: float | int = 100,
    train_size_step: float | int = 10,
    train_size_unit: str = "percent",
    n_splits: int = 20,
    test_size: float | int = 0.2,
    random_state: int | None = None,
    model_names: str | Sequence[str] | None = None,
    metrics: Sequence[str] | str = ("RMSE", "R2", "MAE", "r"),
    metric_splits: Sequence[str] | str = ("test",),
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
    store_predictions: bool = False,
    min_train_samples: int = 2,
) -> LearningCurveResult:
    """Evaluate learning curves from repeated randomized holdout splits.

    Parameters
    ----------
    estimators
        One estimator, a sequence of estimators, a sequence of
        ``(name, estimator)`` pairs, or a mapping of model names to estimators.
    train_sizes
        Optional explicit train sizes. Values are interpreted according to
        ``train_size_unit``. If omitted, values are generated from
        ``train_size_start`` to ``train_size_stop`` using ``train_size_step``.
    train_size_unit
        ``"percent"``, ``"fraction"``, or ``"count"``.
    n_splits
        Number of randomized splits per train-size point.
    metrics
        Metric names to summarize by default, for example ``("RMSE", "R2")``.
    metric_splits
        ``"test"`` for held-out metrics only or ``("train", "test")`` for
        both train and test results.

    Notes
    -----
    The returned ``runs`` dataframe includes ``train_size`` as the realized
    fraction of that split's training pool, ``train_size_percent`` as percent,
    and ``n_train`` as the sample count.
    """
    n_samples = _validate_xy_length(X, y)
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if min_train_samples <= 0:
        raise ValueError("min_train_samples must be positive")

    model_specs = _normalize_estimators(estimators, model_names=model_names)
    normalized_train_sizes = _resolve_learning_curve_train_sizes(
        train_sizes,
        start=train_size_start,
        stop=train_size_stop,
        step=train_size_step,
        unit=train_size_unit,
    )
    metric_names = _normalize_metric_names(metrics)
    split_names = _normalize_metric_splits(metric_splits)
    labels = _sample_labels(y, n_samples)
    splitter = ShuffleSplit(
        n_splits=int(n_splits),
        test_size=test_size,
        random_state=random_state,
    )
    metric_rows = []
    prediction_tables = []

    for repeat, (train_pool_indices, test_indices) in enumerate(splitter.split(np.arange(n_samples)), start=1):
        train_pool_indices = np.asarray(train_pool_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)
        rng = np.random.default_rng(_derive_seed(random_state, repeat))

        for size_position, train_size_spec in enumerate(normalized_train_sizes, start=1):
            train_indices = _subsample_train_indices(
                train_pool_indices,
                train_size_spec["model_train_size"],
                rng=rng,
                min_train_samples=min_train_samples,
            )
            train_fraction = len(train_indices) / len(train_pool_indices)
            for model_position, (model_name, estimator) in enumerate(model_specs, start=1):
                seed = _derive_seed(random_state, repeat, size_position, model_position)
                result = _fit_evaluate_indices(
                    estimator,
                    X,
                    y,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    labels=labels,
                    model_name=model_name,
                    return_std=return_std,
                    confidence_level=confidence_level,
                    include_observation_noise=include_observation_noise,
                    fit_params=fit_params,
                    random_state=seed,
                    context={
                        "repeat": repeat,
                        "train_size": train_fraction,
                        "train_size_percent": 100.0 * train_fraction,
                        "requested_train_size": train_size_spec["requested_train_size"],
                        "requested_train_size_unit": train_size_spec["requested_train_size_unit"],
                    },
                )
                metric_rows.append(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "train_size": train_fraction,
                        "train_size_percent": 100.0 * train_fraction,
                        "requested_train_size": train_size_spec["requested_train_size"],
                        "requested_train_size_unit": train_size_spec["requested_train_size_unit"],
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        **result["metrics"],
                    }
                )
                if store_predictions:
                    prediction_tables.append(result["predictions"])

    predictions = _concat_prediction_tables(prediction_tables) if store_predictions else None
    return LearningCurveResult(
        model_name=", ".join(name for name, _ in model_specs),
        runs=pd.DataFrame(metric_rows),
        predictions=predictions,
        metric_names=metric_names,
        metric_splits=split_names,
    )


def multifidelity_learning_curve(
    estimators,
    X_high,
    y_high,
    *,
    low_fidelity_high=None,
    X_low=None,
    y_low=None,
    train_sizes: Sequence[float | int] | None = None,
    train_size_start: float | int = 10,
    train_size_stop: float | int = 100,
    train_size_step: float | int = 10,
    train_size_unit: str = "percent",
    n_splits: int = 20,
    test_size: float | int = 0.2,
    random_state: int | None = None,
    model_names: str | Sequence[str] | None = None,
    metrics: Sequence[str] | str = ("RMSE", "R2", "MAE", "r"),
    metric_splits: Sequence[str] | str = ("test",),
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    include_low_fidelity_uncertainty: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
    store_predictions: bool = False,
    min_train_samples: int = 2,
) -> LearningCurveResult:
    """Evaluate high-fidelity learning curves for multi-fidelity estimators.

    ``X_high`` and ``y_high`` define the scarce high-fidelity dataset whose
    train size is varied. Use ``low_fidelity_high`` when low-fidelity values
    are available at the same high-fidelity rows. Alternatively, provide a
    fixed low-fidelity training set through ``X_low`` and ``y_low`` so each
    fitted estimator can build or reuse a low-fidelity surrogate.

    The returned ``runs`` dataframe contains high-fidelity train/test metrics.
    When ``store_predictions=True``, the prediction table also includes
    low-fidelity and correction components when the estimator exposes them.
    """
    n_samples = _validate_xy_length(X_high, y_high)
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if min_train_samples <= 0:
        raise ValueError("min_train_samples must be positive")
    _validate_multifidelity_low_inputs(
        X_high,
        low_fidelity_high=low_fidelity_high,
        X_low=X_low,
        y_low=y_low,
    )

    model_specs = _normalize_estimators(estimators, model_names=model_names)
    normalized_train_sizes = _resolve_learning_curve_train_sizes(
        train_sizes,
        start=train_size_start,
        stop=train_size_stop,
        step=train_size_step,
        unit=train_size_unit,
    )
    metric_names = _normalize_metric_names(metrics)
    split_names = _normalize_metric_splits(metric_splits)
    labels = _sample_labels(y_high, n_samples)
    splitter = ShuffleSplit(
        n_splits=int(n_splits),
        test_size=test_size,
        random_state=random_state,
    )
    metric_rows = []
    prediction_tables = []

    for repeat, (train_pool_indices, test_indices) in enumerate(splitter.split(np.arange(n_samples)), start=1):
        train_pool_indices = np.asarray(train_pool_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)
        rng = np.random.default_rng(_derive_seed(random_state, repeat))

        for size_position, train_size_spec in enumerate(normalized_train_sizes, start=1):
            train_indices = _subsample_train_indices(
                train_pool_indices,
                train_size_spec["model_train_size"],
                rng=rng,
                min_train_samples=min_train_samples,
            )
            train_fraction = len(train_indices) / len(train_pool_indices)
            for model_position, (model_name, estimator) in enumerate(model_specs, start=1):
                seed = _derive_seed(random_state, repeat, size_position, model_position)
                result = _fit_evaluate_multifidelity_indices(
                    estimator,
                    X_high,
                    y_high,
                    low_fidelity_high=low_fidelity_high,
                    X_low=X_low,
                    y_low=y_low,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    labels=labels,
                    model_name=model_name,
                    return_std=return_std,
                    confidence_level=confidence_level,
                    include_observation_noise=include_observation_noise,
                    include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
                    fit_params=fit_params,
                    random_state=seed,
                    context={
                        "repeat": repeat,
                        "train_size": train_fraction,
                        "train_size_percent": 100.0 * train_fraction,
                        "requested_train_size": train_size_spec["requested_train_size"],
                        "requested_train_size_unit": train_size_spec["requested_train_size_unit"],
                    },
                )
                metric_rows.append(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "train_size": train_fraction,
                        "train_size_percent": 100.0 * train_fraction,
                        "requested_train_size": train_size_spec["requested_train_size"],
                        "requested_train_size_unit": train_size_spec["requested_train_size_unit"],
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "rho": result["rho"],
                        "intercept": result["intercept"],
                        **result["metrics"],
                    }
                )
                if store_predictions:
                    prediction_tables.append(result["predictions"])

    predictions = _concat_prediction_tables(prediction_tables) if store_predictions else None
    return LearningCurveResult(
        model_name=", ".join(name for name, _ in model_specs),
        runs=pd.DataFrame(metric_rows),
        predictions=predictions,
        metric_names=metric_names,
        metric_splits=split_names,
    )


def _fit_evaluate_multifidelity_indices(
    estimator,
    X_high,
    y_high,
    *,
    low_fidelity_high,
    X_low,
    y_low,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
    include_low_fidelity_uncertainty: bool | None,
    fit_params: Mapping[str, object] | None,
    random_state: int | None,
    context: Mapping[str, object],
) -> dict[str, object]:
    _validate_split_indices(train_indices, test_indices, len(labels))
    fitted_estimator = _clone_estimator(estimator, random_state=random_state)
    X_train = _safe_index(X_high, train_indices)
    X_test = _safe_index(X_high, test_indices)
    y_train = _safe_index_y(y_high, train_indices)
    y_test = _safe_index_y(y_high, test_indices)
    train_low = _safe_index_y(low_fidelity_high, train_indices) if low_fidelity_high is not None else None
    test_low = _safe_index_y(low_fidelity_high, test_indices) if low_fidelity_high is not None else None

    estimator_fit_params = dict(fit_params or {})
    if train_low is not None:
        estimator_fit_params["low_fidelity"] = train_low
    else:
        estimator_fit_params["X_low"] = X_low
        estimator_fit_params["y_low"] = y_low

    fitted_estimator.fit(X_train, y_train, **estimator_fit_params)
    y_train_pred, y_train_std, train_components = _predict_multifidelity_estimator(
        fitted_estimator,
        X_train,
        low_fidelity=train_low,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
        include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
    )
    y_test_pred, y_test_std, test_components = _predict_multifidelity_estimator(
        fitted_estimator,
        X_test,
        low_fidelity=test_low,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
        include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
    )

    metrics = train_test_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
    metrics.update(
        _prefixed_uncertainty_diagnostics(
            y_train,
            y_train_pred,
            y_train_std,
            prefix="train",
            confidence_level=confidence_level,
        )
    )
    metrics.update(
        _prefixed_uncertainty_diagnostics(
            y_test,
            y_test_pred,
            y_test_std,
            prefix="test",
            confidence_level=confidence_level,
        )
    )

    rho = _component_or_attribute(test_components, fitted_estimator, "rho", "rho_")
    intercept = _component_or_attribute(test_components, fitted_estimator, "intercept", "intercept_")
    predictions = pd.concat(
        [
            _multifidelity_prediction_frame(
                model_name,
                "train",
                train_indices,
                labels,
                y_train,
                y_train_pred,
                y_train_std,
                low_fidelity_input=train_low,
                components=train_components,
                context=context,
            ),
            _multifidelity_prediction_frame(
                model_name,
                "test",
                test_indices,
                labels,
                y_test,
                y_test_pred,
                y_test_std,
                low_fidelity_input=test_low,
                components=test_components,
                context=context,
            ),
        ],
        ignore_index=True,
    )
    return {
        "model_name": model_name,
        "estimator": fitted_estimator,
        "metrics": metrics,
        "predictions": predictions,
        "rho": rho,
        "intercept": intercept,
    }


def _fit_evaluate_cokriging_indices(
    estimator,
    X,
    y,
    *,
    fidelity_labels: np.ndarray,
    sample_id: np.ndarray | None,
    fit_indices: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    fidelity_names: Sequence[str],
    target_fidelity: str,
    low_fidelity: str,
    model_name: str,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
    fit_params: Mapping[str, object] | None,
    random_state: int | None,
    context: Mapping[str, object],
) -> dict[str, object]:
    _validate_split_indices(train_indices, test_indices, len(labels))
    fitted_estimator = _clone_estimator(estimator, random_state=random_state)
    _set_supported_cokriging_params(
        fitted_estimator,
        fidelity_order=fidelity_names,
        target_fidelity=target_fidelity,
        low_fidelity=low_fidelity,
    )

    X_fit = _safe_index(X, fit_indices)
    y_fit = _safe_index_y(y, fit_indices)
    X_train = _safe_index(X, train_indices)
    X_test = _safe_index(X, test_indices)
    y_train = _safe_index_y(y, train_indices)
    y_test = _safe_index_y(y, test_indices)

    estimator_fit_params = dict(fit_params or {})
    estimator_fit_params["fidelity"] = fidelity_labels[fit_indices]
    if sample_id is not None and "sample_id" not in estimator_fit_params:
        estimator_fit_params["sample_id"] = sample_id[fit_indices]

    fitted_estimator.fit(X_fit, y_fit, **estimator_fit_params)
    train_prediction, y_train_pred, y_train_std = _predict_cokriging_estimator(
        fitted_estimator,
        X_train,
        target_fidelity=target_fidelity,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
    )
    test_prediction, y_test_pred, y_test_std = _predict_cokriging_estimator(
        fitted_estimator,
        X_test,
        target_fidelity=target_fidelity,
        return_std=return_std,
        confidence_level=confidence_level,
        include_observation_noise=include_observation_noise,
    )

    metrics = train_test_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
    metrics.update(
        _prefixed_uncertainty_diagnostics(
            y_train,
            y_train_pred,
            y_train_std,
            prefix="train",
            confidence_level=confidence_level,
        )
    )
    metrics.update(
        _prefixed_uncertainty_diagnostics(
            y_test,
            y_test_pred,
            y_test_std,
            prefix="test",
            confidence_level=confidence_level,
        )
    )

    rho = _prediction_or_attribute(train_prediction, fitted_estimator, "rho", "rho_")
    predictions = pd.concat(
        [
            _cokriging_prediction_frame(
                train_prediction,
                model_name,
                "train",
                train_indices,
                labels,
                y_train,
                y_train_pred,
                y_train_std,
                target_fidelity=target_fidelity,
                context=context,
            ),
            _cokriging_prediction_frame(
                test_prediction,
                model_name,
                "test",
                test_indices,
                labels,
                y_test,
                y_test_pred,
                y_test_std,
                target_fidelity=target_fidelity,
                context=context,
            ),
        ],
        ignore_index=True,
    )
    return {
        "model_name": model_name,
        "estimator": fitted_estimator,
        "metrics": metrics,
        "predictions": predictions,
        "rho": rho,
    }


def _fit_evaluate_indices(
    estimator,
    X,
    y,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
    fit_params: Mapping[str, object] | None,
    random_state: int | None,
    context: Mapping[str, object],
) -> dict[str, object]:
    _validate_split_indices(train_indices, test_indices, len(labels))
    fitted_estimator = _clone_estimator(estimator, random_state=random_state)
    X_train = _safe_index(X, train_indices)
    X_test = _safe_index(X, test_indices)
    y_train = _safe_index_y(y, train_indices)
    y_test = _safe_index_y(y, test_indices)

    fitted_estimator.fit(X_train, y_train, **dict(fit_params or {}))
    y_train_pred, y_train_std = _predict_estimator(
        fitted_estimator,
        X_train,
        return_std=return_std,
        include_observation_noise=include_observation_noise,
    )
    y_test_pred, y_test_std = _predict_estimator(
        fitted_estimator,
        X_test,
        return_std=return_std,
        include_observation_noise=include_observation_noise,
    )

    metrics = train_test_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
    metrics.update(
        _prefixed_uncertainty_diagnostics(
            y_train,
            y_train_pred,
            y_train_std,
            prefix="train",
            confidence_level=confidence_level,
        )
    )
    metrics.update(
        _prefixed_uncertainty_diagnostics(
            y_test,
            y_test_pred,
            y_test_std,
            prefix="test",
            confidence_level=confidence_level,
        )
    )

    predictions = pd.concat(
        [
            _prediction_frame(
                model_name,
                "train",
                train_indices,
                labels,
                y_train,
                y_train_pred,
                y_train_std,
                context=context,
            ),
            _prediction_frame(
                model_name,
                "test",
                test_indices,
                labels,
                y_test,
                y_test_pred,
                y_test_std,
                context=context,
            ),
        ],
        ignore_index=True,
    )
    return {
        "model_name": model_name,
        "estimator": fitted_estimator,
        "metrics": metrics,
        "predictions": predictions,
    }


def _predict_estimator(
    estimator,
    X,
    *,
    return_std: bool,
    include_observation_noise: bool | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if return_std:
        try:
            prediction = _predict_with_std(
                estimator,
                X,
                include_observation_noise=include_observation_noise,
            )
            return _coerce_prediction_output(prediction)
        except TypeError:
            pass

    return _to_1d_float(estimator.predict(X), "y_pred"), None


def _predict_with_std(estimator, X, *, include_observation_noise: bool | None):
    if include_observation_noise is None:
        return estimator.predict(X, return_std=True)
    try:
        return estimator.predict(
            X,
            return_std=True,
            include_observation_noise=include_observation_noise,
        )
    except TypeError:
        return estimator.predict(X, return_std=True)


def _coerce_prediction_output(prediction) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(prediction, tuple) and len(prediction) >= 2:
        return _to_1d_float(prediction[0], "y_pred"), _to_optional_std(prediction[1])
    return _to_1d_float(prediction, "y_pred"), None


def _prefixed_uncertainty_diagnostics(
    y_true,
    y_pred,
    y_std,
    *,
    prefix: str,
    confidence_level: float,
) -> dict[str, float]:
    if y_std is None:
        return {}
    diagnostics = uncertainty_diagnostics(
        y_true,
        y_pred,
        y_std,
        confidence_level=confidence_level,
    )
    return {f"{prefix}_{key}": value for key, value in diagnostics.items()}


def _prediction_frame(
    model_name: str,
    split: str,
    positions: np.ndarray,
    labels: np.ndarray,
    y_true,
    y_pred,
    y_std,
    *,
    context: Mapping[str, object],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "model": model_name,
            "split": split,
            "sample_position": positions,
            "sample_label": labels[positions],
            "y_true": _to_1d_float(y_true, "y_true"),
            "y_pred": _to_1d_float(y_pred, "y_pred"),
        }
    )
    if y_std is not None:
        frame["y_std"] = _to_optional_std(y_std)
    for key, value in context.items():
        frame[key] = value
    return frame


def _predict_multifidelity_estimator(
    estimator,
    X,
    *,
    low_fidelity,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
    include_low_fidelity_uncertainty: bool | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray | float | None]]:
    if hasattr(estimator, "predict_distribution"):
        prediction = _call_multifidelity_predict_distribution(
            estimator,
            X,
            low_fidelity=low_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
            include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
        )
        return _coerce_multifidelity_prediction_output(prediction)

    if return_std:
        try:
            prediction = _predict_multifidelity_with_std(
                estimator,
                X,
                low_fidelity=low_fidelity,
                include_observation_noise=include_observation_noise,
                include_low_fidelity_uncertainty=include_low_fidelity_uncertainty,
            )
            y_pred, y_std = _coerce_prediction_output(prediction)
            return y_pred, y_std, {}
        except TypeError:
            pass

    try:
        prediction = estimator.predict(X, low_fidelity=low_fidelity)
    except TypeError:
        prediction = estimator.predict(X)
    return _to_1d_float(prediction, "y_pred"), None, {}


def _call_multifidelity_predict_distribution(
    estimator,
    X,
    *,
    low_fidelity,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
    include_low_fidelity_uncertainty: bool | None,
):
    keyword_options = [
        {
            "low_fidelity": low_fidelity,
            "return_std": return_std,
            "confidence_level": confidence_level,
            "include_observation_noise": include_observation_noise,
            "include_low_fidelity_uncertainty": include_low_fidelity_uncertainty,
        },
        {
            "low_fidelity": low_fidelity,
            "return_std": return_std,
            "confidence_level": confidence_level,
            "include_observation_noise": include_observation_noise,
        },
        {
            "low_fidelity": low_fidelity,
            "return_std": return_std,
            "confidence_level": confidence_level,
        },
        {"low_fidelity": low_fidelity, "return_std": return_std},
        {"low_fidelity": low_fidelity},
        {"return_std": return_std},
        {},
    ]
    for keywords in keyword_options:
        keywords = {
            key: value
            for key, value in keywords.items()
            if value is not None or key == "low_fidelity"
        }
        if keywords.get("low_fidelity") is None:
            keywords = {key: value for key, value in keywords.items() if key != "low_fidelity"}
        try:
            return estimator.predict_distribution(X, **keywords)
        except TypeError:
            continue
    return estimator.predict_distribution(X)


def _predict_multifidelity_with_std(
    estimator,
    X,
    *,
    low_fidelity,
    include_observation_noise: bool | None,
    include_low_fidelity_uncertainty: bool | None,
):
    keyword_options = [
        {
            "low_fidelity": low_fidelity,
            "return_std": True,
            "include_observation_noise": include_observation_noise,
            "include_low_fidelity_uncertainty": include_low_fidelity_uncertainty,
        },
        {
            "low_fidelity": low_fidelity,
            "return_std": True,
            "include_observation_noise": include_observation_noise,
        },
        {"low_fidelity": low_fidelity, "return_std": True},
        {"return_std": True},
    ]
    for keywords in keyword_options:
        keywords = {
            key: value
            for key, value in keywords.items()
            if value is not None or key == "low_fidelity"
        }
        if keywords.get("low_fidelity") is None:
            keywords = {key: value for key, value in keywords.items() if key != "low_fidelity"}
        try:
            return estimator.predict(X, **keywords)
        except TypeError:
            continue
    return estimator.predict(X, return_std=True)


def _coerce_multifidelity_prediction_output(
    prediction,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray | float | None]]:
    if isinstance(prediction, tuple) and len(prediction) >= 2:
        return _to_1d_float(prediction[0], "y_pred"), _to_optional_std(prediction[1]), {}
    if _is_prediction_container(prediction):
        y_std = getattr(prediction, "std", None)
        components = {
            "low_fidelity_mean": getattr(prediction, "low_fidelity_mean", None),
            "low_fidelity_std": getattr(prediction, "low_fidelity_std", None),
            "correction_mean": getattr(prediction, "correction_mean", None),
            "correction_std": getattr(prediction, "correction_std", None),
            "rho": getattr(prediction, "rho", None),
            "intercept": getattr(prediction, "intercept", None),
        }
        return (
            _to_1d_float(getattr(prediction, "mean"), "y_pred"),
            _to_optional_std(y_std) if y_std is not None else None,
            components,
        )
    return _to_1d_float(prediction, "y_pred"), None, {}


def _multifidelity_prediction_frame(
    model_name: str,
    split: str,
    positions: np.ndarray,
    labels: np.ndarray,
    y_true,
    y_pred,
    y_std,
    *,
    low_fidelity_input,
    components: Mapping[str, np.ndarray | float | None],
    context: Mapping[str, object],
) -> pd.DataFrame:
    frame = _prediction_frame(
        model_name,
        split,
        positions,
        labels,
        y_true,
        y_pred,
        y_std,
        context=context,
    )
    if low_fidelity_input is not None:
        frame["low_fidelity_input"] = _to_1d_float(low_fidelity_input, "low_fidelity_input")
    _add_optional_component_column(frame, "low_fidelity_pred", components.get("low_fidelity_mean"))
    _add_optional_component_column(frame, "low_fidelity_std", components.get("low_fidelity_std"))
    _add_optional_component_column(frame, "correction_pred", components.get("correction_mean"))
    _add_optional_component_column(frame, "correction_std", components.get("correction_std"))
    if components.get("rho") is not None:
        frame["rho"] = float(components["rho"])
    if components.get("intercept") is not None:
        frame["intercept"] = float(components["intercept"])
    return frame


def _add_optional_component_column(frame: pd.DataFrame, column: str, values) -> None:
    if values is None:
        return
    array = _to_1d_float(values, column)
    if array.shape[0] != frame.shape[0]:
        raise ValueError(f"{column} must contain one value per prediction row")
    frame[column] = array


def _component_or_attribute(
    components: Mapping[str, np.ndarray | float | None],
    estimator,
    component_name: str,
    attribute_name: str,
) -> float:
    value = components.get(component_name)
    if value is None and hasattr(estimator, attribute_name):
        value = getattr(estimator, attribute_name)
    if value is None:
        return np.nan
    return float(value)


def _predict_cokriging_estimator(
    estimator,
    X,
    *,
    target_fidelity: str,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
) -> tuple[object | None, np.ndarray, np.ndarray | None]:
    if hasattr(estimator, "predict_distribution"):
        prediction = _call_cokriging_predict_distribution(
            estimator,
            X,
            target_fidelity=target_fidelity,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )
        y_pred, y_std = _coerce_cokriging_prediction_output(prediction)
        return prediction, y_pred, y_std

    if return_std:
        try:
            prediction = _predict_cokriging_with_std(
                estimator,
                X,
                target_fidelity=target_fidelity,
                include_observation_noise=include_observation_noise,
            )
            y_pred, y_std = _coerce_prediction_output(prediction)
            return None, y_pred, y_std
        except TypeError:
            pass

    try:
        prediction = estimator.predict(X, target_fidelity=target_fidelity)
    except TypeError:
        prediction = estimator.predict(X)
    return None, _to_1d_float(prediction, "y_pred"), None


def _call_cokriging_predict_distribution(
    estimator,
    X,
    *,
    target_fidelity: str,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
):
    keyword_options = [
        {
            "target_fidelity": target_fidelity,
            "return_std": return_std,
            "confidence_level": confidence_level,
            "include_observation_noise": include_observation_noise,
        },
        {
            "target_fidelity": target_fidelity,
            "return_std": return_std,
            "confidence_level": confidence_level,
        },
        {"target_fidelity": target_fidelity, "return_std": return_std},
        {"target_fidelity": target_fidelity},
        {"return_std": return_std, "confidence_level": confidence_level},
        {"return_std": return_std},
        {},
    ]
    for keywords in keyword_options:
        if keywords.get("include_observation_noise") is None:
            keywords = {
                key: value
                for key, value in keywords.items()
                if key != "include_observation_noise"
            }
        try:
            return estimator.predict_distribution(X, **keywords)
        except TypeError:
            continue
    return estimator.predict_distribution(X)


def _predict_cokriging_with_std(
    estimator,
    X,
    *,
    target_fidelity: str,
    include_observation_noise: bool | None,
):
    keyword_options = [
        {
            "target_fidelity": target_fidelity,
            "return_std": True,
            "include_observation_noise": include_observation_noise,
        },
        {"target_fidelity": target_fidelity, "return_std": True},
        {"return_std": True},
    ]
    for keywords in keyword_options:
        if keywords.get("include_observation_noise") is None:
            keywords = {
                key: value
                for key, value in keywords.items()
                if key != "include_observation_noise"
            }
        try:
            return estimator.predict(X, **keywords)
        except TypeError:
            continue
    return estimator.predict(X, return_std=True)


def _coerce_cokriging_prediction_output(prediction) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(prediction, tuple) and len(prediction) >= 2:
        return _to_1d_float(prediction[0], "y_pred"), _to_optional_std(prediction[1])
    if _is_prediction_container(prediction):
        y_std = getattr(prediction, "std", None)
        return (
            _to_1d_float(getattr(prediction, "mean"), "y_pred"),
            _to_optional_std(y_std) if y_std is not None else None,
        )
    return _to_1d_float(prediction, "y_pred"), None


def _cokriging_prediction_frame(
    prediction,
    model_name: str,
    split: str,
    positions: np.ndarray,
    labels: np.ndarray,
    y_true,
    y_pred,
    y_std,
    *,
    target_fidelity: str,
    context: Mapping[str, object],
) -> pd.DataFrame:
    if _is_prediction_container(prediction):
        frame = decompose_multifidelity_prediction(
            prediction,
            y_true=y_true,
            sample_labels=labels[positions],
            model_name=model_name,
            split=split,
        )
        frame["sample_position"] = positions
    else:
        frame = _prediction_frame(
            model_name,
            split,
            positions,
            labels,
            y_true,
            y_pred,
            y_std,
            context={},
        )
    frame["fidelity"] = target_fidelity
    for key, value in context.items():
        frame[key] = value
    return frame


def _prediction_or_attribute(prediction, estimator, prediction_attribute: str, fitted_attribute: str):
    if prediction is not None and hasattr(prediction, prediction_attribute):
        value = getattr(prediction, prediction_attribute)
    else:
        value = getattr(estimator, fitted_attribute, None)
    if value is None:
        return None
    return float(value)


def _predict_multitask_estimator(
    estimator,
    X,
    *,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...] | None]:
    if hasattr(estimator, "predict_distribution"):
        prediction = _call_multitask_predict_distribution(
            estimator,
            X,
            return_std=return_std,
            confidence_level=confidence_level,
            include_observation_noise=include_observation_noise,
        )
        return _coerce_multitask_prediction_output(prediction)

    if return_std:
        try:
            if include_observation_noise is None:
                prediction = estimator.predict(X, return_std=True)
            else:
                prediction = estimator.predict(
                    X,
                    return_std=True,
                    include_observation_noise=include_observation_noise,
                )
            return _coerce_multitask_prediction_output(prediction)
        except TypeError:
            pass

    return _to_2d_float(estimator.predict(X), "y_pred"), None, None


def _call_multitask_predict_distribution(
    estimator,
    X,
    *,
    return_std: bool,
    confidence_level: float,
    include_observation_noise: bool | None,
):
    keyword_options = [
        {
            "return_std": return_std,
            "confidence_level": confidence_level,
            "include_observation_noise": include_observation_noise,
        },
        {
            "return_std": return_std,
            "confidence_level": confidence_level,
        },
        {"return_std": return_std},
        {},
    ]
    for keywords in keyword_options:
        if keywords.get("include_observation_noise") is None:
            keywords = {
                key: value
                for key, value in keywords.items()
                if key != "include_observation_noise"
            }
        try:
            return estimator.predict_distribution(X, **keywords)
        except TypeError:
            continue
    return estimator.predict_distribution(X)


def _coerce_multitask_prediction_output(
    prediction,
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...] | None]:
    task_names = getattr(prediction, "task_names", None)
    if isinstance(prediction, tuple) and len(prediction) >= 2:
        return (
            _to_2d_float(prediction[0], "y_pred"),
            _to_optional_2d_std(prediction[1]),
            _normalize_optional_task_names(task_names),
        )
    if _is_prediction_container(prediction):
        y_std = getattr(prediction, "std", None)
        return (
            _to_2d_float(getattr(prediction, "mean"), "y_pred"),
            _to_optional_2d_std(y_std) if y_std is not None else None,
            _normalize_optional_task_names(task_names),
        )
    return _to_2d_float(prediction, "y_pred"), None, _normalize_optional_task_names(task_names)


def _is_prediction_container(value) -> bool:
    return (
        hasattr(value, "mean")
        and not isinstance(value, (np.ndarray, pd.DataFrame, pd.Series, list, tuple))
    )


def _multitask_prediction_frame(
    model_name: str,
    split: str,
    positions: np.ndarray,
    labels: np.ndarray,
    y_true,
    y_pred,
    y_std,
    *,
    task_names: Sequence[str],
    confidence_level: float,
) -> pd.DataFrame:
    y_true_array = _to_2d_float(y_true, "y_true")
    y_pred_array = _to_2d_float(y_pred, "y_pred")
    _validate_same_2d_shape(y_true_array, y_pred_array, "y_true", "y_pred")
    y_std_array = _to_optional_2d_std(y_std) if y_std is not None else None
    if y_std_array is not None:
        _validate_same_2d_shape(y_true_array, y_std_array, "y_true", "y_std")
    resolved_task_names = _validate_task_names(task_names, y_true_array.shape[1])

    rows = []
    for task_index, task_name in enumerate(resolved_task_names):
        frame = pd.DataFrame(
            {
                "model": str(model_name),
                "split": str(split),
                "sample_position": positions,
                "sample_label": labels[positions],
                "task": task_name,
                "task_index": task_index,
                "y_true": y_true_array[:, task_index],
                "y_pred": y_pred_array[:, task_index],
            }
        )
        if y_std_array is not None:
            frame["y_std"] = y_std_array[:, task_index]
            lower, upper = prediction_interval_bounds(
                y_pred_array[:, task_index],
                y_std_array[:, task_index],
                confidence_level=confidence_level,
            )
            frame["y_lower"] = lower
            frame["y_upper"] = upper
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _sparse_multitask_prediction_frame(
    model_name: str,
    split: str,
    positions: np.ndarray,
    labels: np.ndarray,
    y_true,
    y_pred,
    y_std,
    *,
    task_names: Sequence[str],
    confidence_level: float,
) -> pd.DataFrame:
    y_true_array = _to_2d_float_allow_nan(y_true, "y_true")
    y_pred_array = _to_2d_float(y_pred, "y_pred")
    _validate_same_2d_shape(y_true_array, y_pred_array, "y_true", "y_pred")
    y_std_array = _to_optional_2d_std(y_std) if y_std is not None else None
    if y_std_array is not None:
        _validate_same_2d_shape(y_true_array, y_std_array, "y_true", "y_std")
    resolved_task_names = _validate_task_names(task_names, y_true_array.shape[1])

    rows = []
    for task_index, task_name in enumerate(resolved_task_names):
        observed = np.isfinite(y_true_array[:, task_index])
        frame = pd.DataFrame(
            {
                "model": str(model_name),
                "split": str(split),
                "sample_position": positions,
                "sample_label": labels[positions],
                "task": task_name,
                "task_index": task_index,
                "observed": observed,
                "y_true": y_true_array[:, task_index],
                "y_pred": y_pred_array[:, task_index],
            }
        )
        if y_std_array is not None:
            frame["y_std"] = y_std_array[:, task_index]
            lower, upper = prediction_interval_bounds(
                y_pred_array[:, task_index],
                y_std_array[:, task_index],
                confidence_level=confidence_level,
            )
            frame["y_lower"] = lower
            frame["y_upper"] = upper
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _summarize_metrics(
    frame: pd.DataFrame,
    *,
    group_by: Sequence[str],
    metric_columns: Sequence[str] | None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_by = tuple(group_by)
    missing = [column for column in group_by if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing group_by columns: {missing}")
    if metric_columns is None:
        excluded = set(group_by).union({"fold", "repeat", "n_train", "n_test", "requested_train_size"})
        metric_columns = [
            column
            for column in frame.select_dtypes(include=np.number).columns
            if column not in excluded
        ]
    metric_columns = tuple(metric_columns)
    if not metric_columns:
        raise ValueError("No metric columns are available to summarize")

    summary = (
        frame.groupby(list(group_by), dropna=False)[list(metric_columns)]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def _iter_cv_splits(cv, X, y, *, shuffle: bool, random_state: int | None):
    if isinstance(cv, int):
        if cv < 2:
            raise ValueError("cv must be at least 2")
        splitter = KFold(
            n_splits=cv,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        return splitter.split(X, y)
    if hasattr(cv, "split"):
        return cv.split(X, y)
    raise ValueError("cv must be an integer or a splitter object with split(X, y)")


def _normalize_estimators(estimators, *, model_names: str | Sequence[str] | None):
    if hasattr(estimators, "fit"):
        if model_names is None:
            name = _estimator_name(estimators)
        elif isinstance(model_names, str):
            name = model_names
        else:
            names = tuple(str(name) for name in model_names)
            if len(names) != 1:
                raise ValueError("model_names must contain one name for one estimator")
            name = names[0]
        items = ((str(name), estimators),)
        _validate_estimator_specs(items)
        return items

    if isinstance(estimators, Mapping):
        if model_names is not None:
            raise ValueError("model_names should not be supplied when estimators is a mapping")
        items = tuple((str(name), estimator) for name, estimator in estimators.items())
        if not items:
            raise ValueError("At least one estimator is required")
        _validate_estimator_specs(items)
        return items

    if isinstance(estimators, (str, bytes)):
        raise ValueError("estimators must be estimator objects, not strings")

    try:
        estimator_items = tuple(estimators)
    except TypeError as exc:
        raise ValueError("estimators must be an estimator, mapping, or sequence") from exc

    if not estimator_items:
        raise ValueError("At least one estimator is required")

    if all(_is_named_estimator_pair(item) for item in estimator_items):
        items = tuple((str(name), estimator) for name, estimator in estimator_items)
    else:
        names = _resolve_model_names(model_names, estimator_items)
        items = tuple(zip(names, estimator_items))

    _validate_estimator_specs(items)
    return items


def _is_named_estimator_pair(value) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and isinstance(value[0], str)
    )


def _resolve_model_names(
    model_names: str | Sequence[str] | None,
    estimators: Sequence[object],
) -> tuple[str, ...]:
    if model_names is None:
        return tuple(_estimator_name(estimator) for estimator in estimators)
    if isinstance(model_names, str):
        if len(estimators) != 1:
            raise ValueError("A single model name can only be used with one estimator")
        return (model_names,)
    names = tuple(str(name) for name in model_names)
    if len(names) != len(estimators):
        raise ValueError("model_names must match the number of estimators")
    return names


def _validate_estimator_specs(items: Sequence[tuple[str, object]]) -> None:
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique")
    missing_fit = [
        name
        for name, estimator in items
        if not hasattr(estimator, "fit")
    ]
    missing_predict = [
        name
        for name, estimator in items
        if not hasattr(estimator, "predict")
    ]
    if missing_fit or missing_predict:
        raise ValueError("Each estimator must provide fit and predict methods")


def _resolve_learning_curve_train_sizes(
    train_sizes: Sequence[float | int] | None,
    *,
    start: float | int,
    stop: float | int,
    step: float | int,
    unit: str,
) -> tuple[dict[str, float | int | str], ...]:
    unit = _normalize_train_size_unit(unit)
    values = (
        tuple(train_sizes)
        if train_sizes is not None
        else _inclusive_train_size_range(start=start, stop=stop, step=step)
    )
    if not values:
        raise ValueError("train_sizes must contain at least one value")

    resolved = []
    for value in values:
        requested_value = float(value)
        if unit == "percent":
            if not 0 < requested_value <= 100:
                raise ValueError("Percent train sizes must be in the interval (0, 100]")
            model_train_size: float | int = requested_value / 100.0
        elif unit == "fraction":
            if not 0 < requested_value <= 1:
                raise ValueError("Fraction train sizes must be in the interval (0, 1]")
            model_train_size = requested_value
        else:
            requested_float = float(value)
            if not requested_float.is_integer():
                raise ValueError("Count train sizes must be integers")
            count = int(requested_float)
            if count <= 0:
                raise ValueError("Count train sizes must be positive")
            model_train_size = count
            requested_value = count
        resolved.append(
            {
                "requested_train_size": requested_value,
                "requested_train_size_unit": unit,
                "model_train_size": model_train_size,
            }
        )
    return tuple(resolved)


def _inclusive_train_size_range(
    *,
    start: float | int,
    stop: float | int,
    step: float | int,
) -> tuple[float, ...]:
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step <= 0:
        raise ValueError("train_size_step must be positive")
    if stop < start:
        raise ValueError("train_size_stop must be greater than or equal to train_size_start")

    values = []
    current = start
    tolerance = abs(step) * 1e-9
    while current <= stop + tolerance:
        values.append(round(current, 12))
        current += step
    return tuple(values)


def _normalize_train_size_unit(unit: str) -> str:
    normalized = str(unit).strip().lower()
    aliases = {
        "percent": "percent",
        "percentage": "percent",
        "%": "percent",
        "fraction": "fraction",
        "proportion": "fraction",
        "count": "count",
        "counts": "count",
        "n": "count",
    }
    if normalized not in aliases:
        raise ValueError("train_size_unit must be one of: percent, fraction, count")
    return aliases[normalized]


def _normalize_metric_names(metrics: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(metrics, str):
        metrics = (metrics,)
    names = tuple(dict.fromkeys(_normalize_metric_name(metric) for metric in metrics))
    if not names:
        raise ValueError("metrics must contain at least one metric name")
    return names


def _normalize_metric_name(metric: str) -> str:
    normalized = str(metric).strip().lower()
    aliases = {
        "rmse": "RMSE",
        "root_mean_squared_error": "RMSE",
        "r2": "R2",
        "r^2": "R2",
        "mae": "MAE",
        "mean_absolute_error": "MAE",
        "r": "r",
        "pearson": "r",
        "pearson_r": "r",
    }
    if normalized not in aliases:
        raise ValueError("metrics can include only: RMSE, R2, MAE, r")
    return aliases[normalized]


def _normalize_metric_splits(splits: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(splits, str):
        if splits.lower().replace("_", "-") in {"train-test", "both"}:
            return ("train", "test")
        splits = (splits,)
    normalized = tuple(dict.fromkeys(str(split).strip().lower() for split in splits))
    if not normalized:
        raise ValueError("metric_splits must contain at least one split")
    unknown = [split for split in normalized if split not in {"train", "test"}]
    if unknown:
        raise ValueError("metric_splits can include only 'train' and 'test'")
    return normalized


def _metric_column_name(split: str, metric: str) -> str:
    split = _normalize_metric_splits(split)[0]
    metric = _normalize_metric_name(metric)
    return f"{split}_{metric}"


def _metric_columns(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] | str,
    splits: Sequence[str] | str,
) -> tuple[str, ...]:
    columns = tuple(
        _metric_column_name(split, metric)
        for split in _normalize_metric_splits(splits)
        for metric in _normalize_metric_names(metrics)
    )
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Metric columns are not present in the result: {missing}")
    return columns


def _subsample_train_indices(
    train_pool_indices: np.ndarray,
    requested_train_size: float | int,
    *,
    rng: np.random.Generator,
    min_train_samples: int,
) -> np.ndarray:
    n_pool = len(train_pool_indices)
    if isinstance(requested_train_size, float) and 0 < requested_train_size <= 1:
        n_train = int(np.ceil(requested_train_size * n_pool))
    else:
        n_train = int(requested_train_size)
    if n_train < min_train_samples:
        raise ValueError(
            f"train size {requested_train_size!r} gives {n_train} samples; "
            f"min_train_samples is {min_train_samples}"
        )
    if n_train > n_pool:
        raise ValueError(
            f"train size {requested_train_size!r} requests {n_train} samples, "
            f"but only {n_pool} are available"
        )
    if n_train == n_pool:
        return np.asarray(train_pool_indices, dtype=int)
    return np.asarray(rng.choice(train_pool_indices, size=n_train, replace=False), dtype=int)


def _clone_estimator(estimator, *, random_state: int | None):
    try:
        cloned = clone(estimator)
    except TypeError:
        cloned = copy.deepcopy(estimator)

    if random_state is not None and hasattr(cloned, "get_params") and hasattr(cloned, "set_params"):
        params = cloned.get_params(deep=False)
        if "random_state" in params:
            cloned.set_params(random_state=random_state)
    return cloned


def _set_supported_cokriging_params(
    estimator,
    *,
    fidelity_order: Sequence[str],
    target_fidelity: str,
    low_fidelity: str,
) -> None:
    if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
        return
    params = estimator.get_params(deep=False)
    updates = {}
    if "fidelity_order" in params:
        updates["fidelity_order"] = tuple(fidelity_order)
    if "target_fidelity" in params:
        updates["target_fidelity"] = target_fidelity
    if "low_fidelity" in params:
        updates["low_fidelity"] = low_fidelity
    if updates:
        estimator.set_params(**updates)


def _validate_multitask_xy_length(X, y) -> tuple[int, int]:
    n_x = len(X)
    y_values = _to_2d_float(y, "y")
    if n_x != y_values.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if n_x < 2:
        raise ValueError("At least two samples are required")
    if y_values.shape[1] < 2:
        raise ValueError("Multitask validation requires at least two target columns")
    return int(n_x), int(y_values.shape[1])


def _validate_sparse_multitask_xy_length(X, y) -> tuple[int, int]:
    n_x = len(X)
    y_values = _to_2d_float_allow_nan(y, "y")
    if n_x != y_values.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if n_x < 2:
        raise ValueError("At least two samples are required")
    if y_values.shape[1] < 2:
        raise ValueError("Sparse multitask validation requires at least two target columns")
    if not np.isfinite(y_values).any():
        raise ValueError("Sparse multitask validation requires at least one observed target")
    return int(n_x), int(y_values.shape[1])


def _validate_xy_length(X, y) -> int:
    n_x = len(X)
    y_values = _to_1d_float(y, "y")
    if n_x != y_values.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if n_x < 2:
        raise ValueError("At least two samples are required")
    return int(n_x)


def _validate_multifidelity_low_inputs(
    X_high,
    *,
    low_fidelity_high,
    X_low,
    y_low,
) -> None:
    if low_fidelity_high is not None:
        if X_low is not None or y_low is not None:
            raise ValueError("Use either low_fidelity_high or X_low/y_low, not both")
        low_values = _to_1d_float(low_fidelity_high, "low_fidelity_high")
        if len(X_high) != low_values.shape[0]:
            raise ValueError("low_fidelity_high must contain one value per high-fidelity sample")
        return

    if X_low is None or y_low is None:
        raise ValueError(
            "Provide low_fidelity_high, or provide both X_low and y_low for a "
            "low-fidelity surrogate"
        )
    _validate_xy_length(X_low, y_low)
    high_width = _feature_count(X_high, "X_high")
    low_width = _feature_count(X_low, "X_low")
    if high_width != low_width:
        raise ValueError(
            f"X_high has {high_width} features, but X_low has {low_width} features"
        )


def _to_fidelity_label_array(values, *, n_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=object).reshape(-1)
    if array.shape[0] != n_samples:
        raise ValueError(
            f"fidelity must contain {n_samples} value(s); got {array.shape[0]}"
        )
    labels = [_normalize_fidelity_label(value, "fidelity") for value in array]
    return np.asarray(labels, dtype=object)


def _resolve_optional_row_values(values, *, n_samples: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=object).reshape(-1)
    if array.shape[0] != n_samples:
        raise ValueError(f"{name} must contain {n_samples} value(s); got {array.shape[0]}")
    return array


def _resolve_cokriging_validation_levels(
    fidelity_labels: np.ndarray,
    *,
    estimator,
    target_fidelity: str | None,
    low_fidelity: str | None,
    fidelity_order: Sequence[str] | None,
) -> tuple[tuple[str, ...], str, str]:
    estimator_order = getattr(estimator, "fidelity_order", None)
    resolved_order = fidelity_order if fidelity_order is not None else estimator_order
    fidelity_names = _resolve_validation_fidelity_names(fidelity_labels, resolved_order)
    if len(fidelity_names) != 2:
        raise ValueError(
            "Co-kriging train/test validation currently supports exactly two "
            f"fidelity levels; got {len(fidelity_names)}"
        )

    estimator_target = getattr(estimator, "target_fidelity", None)
    target_value = target_fidelity if target_fidelity is not None else estimator_target
    target_name = (
        fidelity_names[-1]
        if target_value is None
        else _normalize_fidelity_label(target_value, "target_fidelity")
    )
    if target_name not in fidelity_names:
        raise ValueError(f"target_fidelity must be one of {fidelity_names}; got {target_name!r}")

    estimator_low = getattr(estimator, "low_fidelity", None)
    low_value = low_fidelity if low_fidelity is not None else estimator_low
    low_candidates = [name for name in fidelity_names if name != target_name]
    low_name = (
        low_candidates[0]
        if low_value is None
        else _normalize_fidelity_label(low_value, "low_fidelity")
    )
    if low_name not in fidelity_names:
        raise ValueError(f"low_fidelity must be one of {fidelity_names}; got {low_name!r}")
    if low_name == target_name:
        raise ValueError("low_fidelity and target_fidelity must be different")
    return fidelity_names, target_name, low_name


def _resolve_validation_fidelity_names(
    fidelity_labels: np.ndarray,
    fidelity_order: Sequence[str] | None,
) -> tuple[str, ...]:
    if fidelity_order is None:
        fidelity_names = tuple(dict.fromkeys(fidelity_labels.tolist()))
    else:
        fidelity_names = tuple(
            _normalize_fidelity_label(name, "fidelity_order") for name in fidelity_order
        )
    if not fidelity_names:
        raise ValueError("fidelity_order must contain at least one fidelity level")
    if len(set(fidelity_names)) != len(fidelity_names):
        raise ValueError("fidelity_order must contain unique fidelity labels")

    observed = set(fidelity_labels.tolist())
    expected = set(fidelity_names)
    unknown = sorted(observed - expected)
    if unknown:
        raise ValueError(
            "fidelity contains labels not present in fidelity_order: "
            f"{unknown}; expected one of {fidelity_names}"
        )
    missing = [name for name in fidelity_names if name not in observed]
    if missing:
        raise ValueError(f"fidelity_order contains unobserved fidelity level(s): {missing}")
    return fidelity_names


def _normalize_fidelity_label(value, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must not contain missing labels")
    label = str(value).strip()
    if label == "" or label.lower() == "nan":
        raise ValueError(f"{name} must not contain missing labels")
    return label


def _validate_target_fidelity_indices(
    indices: np.ndarray,
    *,
    target_mask: np.ndarray,
    name: str,
    target_fidelity: str,
) -> None:
    if not np.all(target_mask[indices]):
        raise ValueError(f"{name} must contain only target_fidelity={target_fidelity!r} rows")


def _feature_count(values, name: str) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    return int(array.shape[1])


def _to_index_array(indices: Sequence[int], name: str, n_samples: int) -> np.ndarray:
    array = np.asarray(indices, dtype=int).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one index")
    if np.any(array < 0) or np.any(array >= n_samples):
        raise ValueError(f"{name} contains indices outside the dataset")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} must not contain duplicates")
    return array


def _validate_split_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
) -> None:
    _to_index_array(train_indices, "train_indices", n_samples)
    _to_index_array(test_indices, "test_indices", n_samples)
    overlap = np.intersect1d(train_indices, test_indices)
    if overlap.size:
        raise ValueError("train_indices and test_indices must not overlap")


def _safe_index(values, indices: np.ndarray):
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return np.asarray(values)[indices]


def _safe_index_y(y, indices: np.ndarray) -> np.ndarray:
    return _to_1d_float(_safe_index(y, indices), "y")


def _to_2d_float(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix with shape (n_samples, n_tasks)")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_2d_float_allow_nan(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix with shape (n_samples, n_tasks)")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values")
    return array


def _to_1d_float(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _to_optional_std(values) -> np.ndarray:
    std = _to_1d_float(values, "y_std")
    if np.any(std <= 0):
        raise ValueError("y_std must contain only positive values")
    return std


def _to_optional_2d_std(values) -> np.ndarray:
    std = _to_2d_float(values, "y_std")
    if np.any(std <= 0):
        raise ValueError("y_std must contain only positive values")
    return std


def _validate_same_2d_shape(
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} must have the same shape")


def _resolve_multitask_names(
    *,
    task_names: Sequence[str] | None,
    n_tasks: int,
    y=None,
    prediction_task_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if task_names is not None:
        return _validate_task_names(task_names, n_tasks)
    if prediction_task_names is not None:
        return _validate_task_names(prediction_task_names, n_tasks)
    inferred = _task_names_from_multitask_target(y)
    if inferred is not None:
        return _validate_task_names(inferred, n_tasks)
    return tuple(f"task_{index}" for index in range(n_tasks))


def _normalize_optional_task_names(task_names) -> tuple[str, ...] | None:
    if task_names is None:
        return None
    return tuple(str(name) for name in task_names)


def _first_available_task_names(*task_name_sources) -> tuple[str, ...] | None:
    for task_names in task_name_sources:
        normalized = _normalize_optional_task_names(task_names)
        if normalized is not None:
            return normalized
    return None


def _task_names_from_multitask_target(y) -> tuple[str, ...] | None:
    if hasattr(y, "columns"):
        columns = tuple(str(column) for column in y.columns)
        if columns:
            return columns
    return None


def _validate_task_names(task_names: Sequence[str], n_tasks: int) -> tuple[str, ...]:
    names = tuple(str(name) for name in task_names)
    if len(names) != n_tasks:
        raise ValueError(f"task_names must contain {n_tasks} names; got {len(names)}")
    if any(not name for name in names):
        raise ValueError("task_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("task_names must be unique")
    return names


def _observed_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.shape[0] == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "r": np.nan}
    if y_true.shape[0] == 1:
        error = float(y_true[0] - y_pred[0])
        return {
            "R2": np.nan,
            "RMSE": abs(error),
            "MAE": abs(error),
            "r": np.nan,
        }
    return regression_metrics(y_true, y_pred)


def _empty_uncertainty_diagnostics(confidence_level: float) -> dict[str, float]:
    return {
        "NLPD": np.nan,
        "mean_std": np.nan,
        "median_std": np.nan,
        "mean_absolute_error": np.nan,
        "mean_standardized_residual": np.nan,
        "std_standardized_residual": np.nan,
        "uncertainty_error_spearman": np.nan,
        "confidence_level": float(confidence_level),
        "expected_coverage": float(confidence_level),
        "observed_coverage": np.nan,
        "coverage_error": np.nan,
        "mean_interval_width": np.nan,
        "median_interval_width": np.nan,
    }


def _sample_labels(y, n_samples: int) -> np.ndarray:
    if hasattr(y, "index"):
        index = np.asarray(y.index, dtype=object)
        if index.shape[0] == n_samples:
            return index
    return np.arange(n_samples, dtype=int)


def _estimator_name(estimator) -> str:
    return estimator.__class__.__name__


def _derive_seed(random_state: int | None, *parts: int) -> int | None:
    if random_state is None:
        return None
    seed = int(random_state)
    for part in parts:
        seed = (seed * 1_000_003 + int(part)) % (2**32 - 1)
    return seed


def _concat_prediction_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    return pd.concat(list(tables), ignore_index=True)
