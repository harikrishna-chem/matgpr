from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from matgpr import (
    CrossValidationResult,
    LearningCurveResult,
    MultitaskTrainTestValidationResult,
    TrainTestValidationResult,
    cross_validate_regressor,
    evaluate_multitask_train_test_split,
    evaluate_train_test_split,
    learning_curve,
    summarize_multitask_predictions,
)


class LinearMeanStdRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, constant_std: float = 0.25, random_state: int | None = None):
        self.constant_std = constant_std
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        design = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients_ = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, X, return_std: bool = False):
        X = np.asarray(X, dtype=float)
        design = np.column_stack([np.ones(X.shape[0]), X])
        mean = design @ self.coefficients_
        if return_std:
            return mean, np.full(mean.shape, self.constant_std, dtype=float)
        return mean


class MultiOutputMeanStdRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, constant_std: float = 0.25, random_state: int | None = None):
        self.constant_std = constant_std
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients_ = np.linalg.lstsq(design, y_array, rcond=None)[0]
        if hasattr(y, "columns"):
            self.task_names_ = tuple(str(column) for column in y.columns)
        return self

    def predict(self, X, return_std: bool = False):
        X = np.asarray(X, dtype=float)
        design = np.column_stack([np.ones(X.shape[0]), X])
        mean = design @ self.coefficients_
        if return_std:
            std = np.full(mean.shape, self.constant_std, dtype=float)
            return mean, std
        return mean


class ValidationApiTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0.0, 1.0, 18)
        self.X = pd.DataFrame({"x": x})
        self.y = pd.Series(2.0 * x + 0.5, index=[f"sample_{i}" for i in range(len(x))])
        self.estimator = LinearMeanStdRegressor(constant_std=0.2)

    def test_evaluate_train_test_split_returns_metrics_predictions_and_uncertainty(self):
        result = evaluate_train_test_split(
            self.estimator,
            self.X,
            self.y,
            train_indices=np.arange(12),
            test_indices=np.arange(12, 18),
            model_name="linear_std",
            confidence_level=0.9,
            include_observation_noise=True,
        )

        self.assertIsInstance(result, TrainTestValidationResult)
        self.assertEqual(result.metrics_frame().loc[0, "model"], "linear_std")
        self.assertIn("test_RMSE", result.metrics)
        self.assertIn("test_NLPD", result.metrics)
        self.assertEqual(result.train_predictions.shape[0], 12)
        self.assertEqual(result.test_predictions.shape[0], 6)
        self.assertIn("y_std", result.predictions.columns)
        self.assertEqual(result.test_predictions["sample_label"].iloc[0], "sample_12")

    def test_cross_validate_regressor_returns_fold_metrics_summary_and_oof_predictions(self):
        result = cross_validate_regressor(
            self.estimator,
            self.X,
            self.y,
            cv=3,
            random_state=4,
            model_name="cv_model",
        )
        summary = result.summary(metric_columns=["test_RMSE", "test_R2"])

        self.assertIsInstance(result, CrossValidationResult)
        self.assertEqual(result.fold_metrics.shape[0], 3)
        self.assertEqual(result.predictions.shape[0], len(self.y))
        self.assertTrue(np.all(result.predictions["split"] == "test"))
        self.assertIn("test_RMSE_mean", summary.columns)
        self.assertIn("test_R2_std", summary.columns)

    def test_cross_validate_regressor_can_store_train_predictions(self):
        result = cross_validate_regressor(
            self.estimator,
            self.X,
            self.y,
            cv=3,
            store_train_predictions=True,
        )

        self.assertIn("train", set(result.predictions["split"]))
        self.assertIn("test", set(result.predictions["split"]))
        self.assertGreater(result.predictions.shape[0], len(self.y))

    def test_learning_curve_returns_run_metrics_summary_and_predictions(self):
        result = learning_curve(
            self.estimator,
            self.X,
            self.y,
            train_sizes=(50, 100),
            train_size_unit="percent",
            n_splits=3,
            test_size=6,
            random_state=9,
            model_names="lc_model",
            metric_splits=("train", "test"),
            store_predictions=True,
        )
        summary = result.summary(metric_columns=["test_RMSE"])

        self.assertIsInstance(result, LearningCurveResult)
        self.assertEqual(result.runs.shape[0], 6)
        self.assertEqual(set(result.runs["train_size_percent"]), {50.0, 100.0})
        self.assertIsNotNone(result.predictions)
        self.assertIn("test_RMSE_mean", summary.columns)
        self.assertEqual(summary.shape[0], 2)

    def test_learning_curve_accepts_multiple_models_intervals_and_metric_choices(self):
        result = learning_curve(
            {
                "linear_a": LinearMeanStdRegressor(constant_std=0.2),
                "linear_b": LinearMeanStdRegressor(constant_std=0.3),
            },
            self.X,
            self.y,
            train_size_start=10,
            train_size_stop=30,
            train_size_step=10,
            train_size_unit="percent",
            n_splits=2,
            test_size=6,
            random_state=12,
            metrics=("RMSE", "R2"),
            metric_splits=("train", "test"),
        )
        summary = result.summary()

        self.assertEqual(result.runs.shape[0], 12)
        self.assertEqual(result.metric_names, ("RMSE", "R2"))
        self.assertEqual(result.metric_splits, ("train", "test"))
        self.assertEqual(set(result.runs["requested_train_size"]), {10.0, 20.0, 30.0})
        self.assertIn("train_RMSE_mean", summary.columns)
        self.assertIn("test_R2_std", summary.columns)
        self.assertEqual(summary.shape[0], 6)

    def test_validation_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            evaluate_train_test_split(self.estimator, self.X.iloc[:3], self.y.iloc[:2])
        with self.assertRaises(ValueError):
            cross_validate_regressor(self.estimator, self.X, self.y, cv=1)
        with self.assertRaises(ValueError):
            learning_curve(
                self.estimator,
                self.X,
                self.y,
                train_sizes=(1,),
                train_size_unit="percent",
                n_splits=1,
                min_train_samples=2,
            )
        with self.assertRaises(ValueError):
            learning_curve(self.estimator, self.X, self.y, metrics=("not_a_metric",))
        with self.assertRaises(ValueError):
            learning_curve({}, self.X, self.y)
        with self.assertRaises(ValueError):
            learning_curve(
                self.estimator,
                self.X,
                self.y,
                train_sizes=(2.5,),
                train_size_unit="count",
            )

    def test_summarize_multitask_predictions_returns_per_task_metrics(self):
        y_true = pd.DataFrame(
            {
                "strength": [1.0, 2.0, 3.0, 4.0],
                "ductility": [4.0, 3.0, 2.0, 1.0],
            }
        )
        prediction = SimpleNamespace(
            mean=np.asarray(y_true) + np.array([[0.1, -0.1]]),
            std=np.full((4, 2), 0.3),
            task_names=("strength", "ductility"),
        )

        summary = summarize_multitask_predictions(
            y_true,
            prediction,
            model_name="multi",
            split="test",
            confidence_level=0.9,
        )

        self.assertEqual(summary.shape[0], 2)
        self.assertEqual(summary["task"].tolist(), ["strength", "ductility"])
        self.assertIn("RMSE", summary.columns)
        self.assertIn("observed_coverage", summary.columns)
        self.assertTrue(np.all(summary["n_samples"] == 4))

    def test_evaluate_multitask_train_test_split_returns_task_tables(self):
        x = np.linspace(0.0, 1.0, 12)
        X = pd.DataFrame({"x": x})
        y = pd.DataFrame(
            {
                "strength": 2.0 * x + 0.5,
                "ductility": -1.5 * x + 2.0,
            },
            index=[f"sample_{index}" for index in range(len(x))],
        )

        result = evaluate_multitask_train_test_split(
            MultiOutputMeanStdRegressor(constant_std=0.2),
            X,
            y,
            train_indices=np.arange(8),
            test_indices=np.arange(8, 12),
            model_name="multi_linear",
            confidence_level=0.9,
        )

        self.assertIsInstance(result, MultitaskTrainTestValidationResult)
        self.assertEqual(result.task_metrics.shape[0], 4)
        self.assertEqual(set(result.task_metrics["split"]), {"train", "test"})
        self.assertEqual(set(result.task_metrics["task"]), {"strength", "ductility"})
        self.assertEqual(result.train_predictions.shape[0], 16)
        self.assertEqual(result.test_predictions.shape[0], 8)
        self.assertIn("y_std", result.predictions.columns)
        self.assertIn("y_lower", result.predictions.columns)
        self.assertIn("y_upper", result.predictions.columns)
        self.assertEqual(result.test_predictions["sample_label"].iloc[0], "sample_8")
        self.assertIn("RMSE_mean", result.summary(metric_columns=["RMSE"]).columns)

    def test_multitask_validation_errors_are_explicit(self):
        y_true = np.ones((3, 2))
        y_pred = np.ones((3, 1))

        with self.assertRaisesRegex(ValueError, "same shape"):
            summarize_multitask_predictions(y_true, y_pred)
        with self.assertRaisesRegex(ValueError, "task_names"):
            summarize_multitask_predictions(
                y_true,
                y_true,
                task_names=("only_one",),
            )


if __name__ == "__main__":
    unittest.main()
