from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "regression_metrics",
    "train_test_regression_metrics",
]


def regression_metrics(
    y_true,
    y_pred,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Calculate common regression metrics for model analysis.

    The returned dictionary contains R2, RMSE, MAE, and Pearson r. Use
    ``prefix`` to distinguish train/test metrics when storing experiment
    results, for example ``prefix="test"`` gives ``test_R2``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan

    metric_prefix = f"{prefix}_" if prefix else ""
    return {
        f"{metric_prefix}R2": r2,
        f"{metric_prefix}RMSE": rmse,
        f"{metric_prefix}MAE": mae,
        f"{metric_prefix}r": r,
    }


def train_test_regression_metrics(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
) -> dict[str, float]:
    """Calculate regression metrics for both train and test predictions."""
    metrics = {}
    metrics.update(
        regression_metrics(
            y_train_true,
            y_train_pred,
            prefix="train",
        )
    )
    metrics.update(
        regression_metrics(
            y_test_true,
            y_test_pred,
            prefix="test",
        )
    )
    return metrics
