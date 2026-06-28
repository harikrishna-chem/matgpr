from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split

from .metrics import train_test_regression_metrics
from .uncertainty import uncertainty_diagnostics

__all__ = [
    "CrossValidationResult",
    "LearningCurveResult",
    "TrainTestValidationResult",
    "cross_validate_regressor",
    "evaluate_train_test_split",
    "repeated_learning_curve",
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
    """Result from repeated learning-curve evaluation."""

    model_name: str
    runs: pd.DataFrame
    predictions: pd.DataFrame | None = None

    def summary(self, metric_columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Summarize repeated runs by model and train-set size."""
        return _summarize_metrics(
            self.runs,
            group_by=("model", "train_size", "train_size_percent"),
            metric_columns=metric_columns,
        )


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


def repeated_learning_curve(
    estimator,
    X,
    y,
    *,
    train_sizes: Sequence[float | int] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    n_repeats: int = 20,
    test_size: float | int = 0.2,
    random_state: int | None = None,
    model_name: str | None = None,
    return_std: bool = True,
    confidence_level: float = 0.95,
    include_observation_noise: bool | None = None,
    fit_params: Mapping[str, object] | None = None,
    store_predictions: bool = False,
    min_train_samples: int = 2,
) -> LearningCurveResult:
    """Evaluate repeated learning curves from randomized holdout splits.

    Each repeat first creates a train/test holdout split. Each value in
    ``train_sizes`` then selects a subset of that repeat's training pool. A
    value of ``1.0`` therefore means 100 percent of the training pool, not 100
    percent of the full dataset.
    """
    n_samples = _validate_xy_length(X, y)
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if min_train_samples <= 0:
        raise ValueError("min_train_samples must be positive")

    labels = _sample_labels(y, n_samples)
    resolved_model_name = model_name or _estimator_name(estimator)
    splitter = ShuffleSplit(
        n_splits=int(n_repeats),
        test_size=test_size,
        random_state=random_state,
    )
    metric_rows = []
    prediction_tables = []

    for repeat, (train_pool_indices, test_indices) in enumerate(splitter.split(np.arange(n_samples)), start=1):
        train_pool_indices = np.asarray(train_pool_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)
        rng = np.random.default_rng(_derive_seed(random_state, repeat))

        for size_position, requested_train_size in enumerate(train_sizes, start=1):
            train_indices = _subsample_train_indices(
                train_pool_indices,
                requested_train_size,
                rng=rng,
                min_train_samples=min_train_samples,
            )
            train_fraction = len(train_indices) / len(train_pool_indices)
            seed = _derive_seed(random_state, repeat, size_position)
            result = _fit_evaluate_indices(
                estimator,
                X,
                y,
                train_indices=train_indices,
                test_indices=test_indices,
                labels=labels,
                model_name=resolved_model_name,
                return_std=return_std,
                confidence_level=confidence_level,
                include_observation_noise=include_observation_noise,
                fit_params=fit_params,
                random_state=seed,
                context={
                    "repeat": repeat,
                    "train_size": train_fraction,
                    "train_size_percent": 100.0 * train_fraction,
                    "requested_train_size": requested_train_size,
                },
            )
            metric_rows.append(
                {
                    "model": resolved_model_name,
                    "repeat": repeat,
                    "train_size": train_fraction,
                    "train_size_percent": 100.0 * train_fraction,
                    "requested_train_size": requested_train_size,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    **result["metrics"],
                }
            )
            if store_predictions:
                prediction_tables.append(result["predictions"])

    predictions = _concat_prediction_tables(prediction_tables) if store_predictions else None
    return LearningCurveResult(
        model_name=resolved_model_name,
        runs=pd.DataFrame(metric_rows),
        predictions=predictions,
    )


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


def _validate_xy_length(X, y) -> int:
    n_x = len(X)
    y_values = _to_1d_float(y, "y")
    if n_x != y_values.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if n_x < 2:
        raise ValueError("At least two samples are required")
    return int(n_x)


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
