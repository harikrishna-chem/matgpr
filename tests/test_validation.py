from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from matgpr import (
    CoKrigingTrainTestValidationResult,
    CrossValidationResult,
    LearningCurveResult,
    MultitaskTrainTestValidationResult,
    SparseMultitaskTrainTestValidationResult,
    TrainTestValidationResult,
    cokriging_learning_curve,
    cross_validate_regressor,
    evaluate_cokriging_train_test_split,
    evaluate_multitask_train_test_split,
    evaluate_sparse_multitask_train_test_split,
    evaluate_train_test_split,
    learning_curve,
    multifidelity_learning_curve,
    summarize_multitask_predictions,
    summarize_sparse_multitask_predictions,
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


class SimpleMultiFidelityRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, constant_std: float = 0.15, random_state: int | None = None):
        self.constant_std = constant_std
        self.random_state = random_state

    def fit(self, X, y, *, low_fidelity=None, X_low=None, y_low=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if low_fidelity is None:
            if X_low is None or y_low is None:
                raise ValueError("low_fidelity or X_low/y_low is required")
            self.low_coefficients_ = self._fit_linear_model(X_low, y_low)
            low_fidelity = self._predict_low_fidelity(X)
        else:
            self.low_coefficients_ = None

        low = np.asarray(low_fidelity, dtype=float).ravel()
        design = np.column_stack([low, np.ones_like(low), X])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        self.rho_ = float(coefficients[0])
        self.intercept_ = float(coefficients[1])
        self.correction_coefficients_ = coefficients[2:]
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
    ):
        del confidence_level, include_observation_noise
        X = np.asarray(X, dtype=float)
        if low_fidelity is None:
            low_mean = self._predict_low_fidelity(X)
            low_std = np.full(low_mean.shape, self.constant_std, dtype=float)
        else:
            low_mean = np.asarray(low_fidelity, dtype=float).ravel()
            low_std = None

        correction_mean = X @ self.correction_coefficients_
        mean = self.rho_ * low_mean + self.intercept_ + correction_mean
        correction_std = np.full(mean.shape, self.constant_std, dtype=float)
        std = None
        if return_std:
            std = correction_std.copy()
            include_low = True if include_low_fidelity_uncertainty is None else include_low_fidelity_uncertainty
            if include_low and low_std is not None:
                std = np.sqrt(std**2 + (self.rho_ * low_std) ** 2)

        return SimpleNamespace(
            mean=mean,
            std=std,
            low_fidelity_mean=low_mean,
            low_fidelity_std=low_std,
            correction_mean=correction_mean,
            correction_std=correction_std if return_std else None,
            rho=self.rho_,
            intercept=self.intercept_,
        )

    @staticmethod
    def _fit_linear_model(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        design = np.column_stack([np.ones(X.shape[0]), X])
        return np.linalg.lstsq(design, y, rcond=None)[0]

    def _predict_low_fidelity(self, X):
        if self.low_coefficients_ is None:
            raise ValueError("low_fidelity is required because no low-fidelity model was fitted")
        design = np.column_stack([np.ones(X.shape[0]), X])
        return design @ self.low_coefficients_


class SimpleRowWiseCoKrigingRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        constant_std: float = 0.15,
        fidelity_order=None,
        low_fidelity=None,
        target_fidelity=None,
        random_state: int | None = None,
    ):
        self.constant_std = constant_std
        self.fidelity_order = fidelity_order
        self.low_fidelity = low_fidelity
        self.target_fidelity = target_fidelity
        self.random_state = random_state

    def fit(self, X, y, *, fidelity, sample_id=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        fidelity = np.asarray(fidelity, dtype=object).reshape(-1)
        if self.fidelity_order is None:
            self.fidelity_names_ = tuple(dict.fromkeys(fidelity.tolist()))
        else:
            self.fidelity_names_ = tuple(self.fidelity_order)
        self.target_fidelity_ = self.target_fidelity or self.fidelity_names_[-1]
        low_candidates = [name for name in self.fidelity_names_ if name != self.target_fidelity_]
        self.low_fidelity_ = self.low_fidelity or low_candidates[0]
        self.fit_fidelity_ = fidelity.copy()
        self.fit_sample_id_ = None if sample_id is None else np.asarray(sample_id, dtype=object)

        low_mask = fidelity == self.low_fidelity_
        target_mask = fidelity == self.target_fidelity_
        self.low_coefficients_ = self._fit_linear(X[low_mask], y[low_mask])
        low_at_target = self._predict_low(X[target_mask])
        target_design = np.column_stack([low_at_target, np.ones_like(low_at_target), X[target_mask]])
        coefficients = np.linalg.lstsq(target_design, y[target_mask], rcond=None)[0]
        self.rho_ = float(coefficients[0])
        self.intercept_ = float(coefficients[1])
        self.discrepancy_coefficients_ = coefficients[2:]
        return self

    def predict(self, X, *, target_fidelity=None, return_std: bool = False):
        prediction = self.predict_distribution(
            X,
            target_fidelity=target_fidelity,
            return_std=return_std,
        )
        if return_std:
            return prediction.mean, prediction.std
        return prediction.mean

    def predict_distribution(
        self,
        X,
        *,
        target_fidelity=None,
        return_std: bool = True,
        confidence_level: float | None = None,
        include_observation_noise: bool | None = None,
    ):
        del include_observation_noise
        X = np.asarray(X, dtype=float)
        fidelity = target_fidelity or self.target_fidelity_
        low_mean = self._predict_low(X)
        low_std = np.full(low_mean.shape, self.constant_std, dtype=float) if return_std else None
        if fidelity == self.low_fidelity_:
            mean = low_mean
            std = low_std
            return SimpleNamespace(
                mean=mean,
                std=std,
                low_fidelity_mean=low_mean,
                low_fidelity_std=low_std,
                fidelity=fidelity,
                rho=self.rho_,
            )

        discrepancy_mean = self.intercept_ + X @ self.discrepancy_coefficients_
        scaled_low = self.rho_ * low_mean
        mean = scaled_low + discrepancy_mean
        discrepancy_std = (
            np.full(mean.shape, self.constant_std, dtype=float) if return_std else None
        )
        scaled_low_std = np.abs(self.rho_) * low_std if low_std is not None else None
        std = None
        lower = None
        upper = None
        if return_std:
            std = np.sqrt(scaled_low_std**2 + discrepancy_std**2)
            if confidence_level is not None:
                lower = mean - 1.96 * std
                upper = mean + 1.96 * std
        return SimpleNamespace(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            low_fidelity_mean=low_mean,
            low_fidelity_std=low_std,
            scaled_low_fidelity_mean=scaled_low,
            scaled_low_fidelity_std=scaled_low_std,
            discrepancy_mean=discrepancy_mean,
            discrepancy_std=discrepancy_std,
            fidelity=fidelity,
            rho=self.rho_,
        )

    @staticmethod
    def _fit_linear(X, y):
        design = np.column_stack([np.ones(X.shape[0]), X])
        return np.linalg.lstsq(design, y, rcond=None)[0]

    def _predict_low(self, X):
        design = np.column_stack([np.ones(X.shape[0]), X])
        return design @ self.low_coefficients_


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


class SparseMultiOutputMeanStdRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, constant_std: float = 0.25, random_state: int | None = None):
        self.constant_std = constant_std
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(X.shape[0]), X])
        coefficients = []
        for task_index in range(y_array.shape[1]):
            observed = np.isfinite(y_array[:, task_index])
            if not observed.any():
                raise ValueError("Each task needs at least one observed target")
            coefficients.append(np.linalg.lstsq(design[observed], y_array[observed, task_index], rcond=None)[0])
        self.coefficients_ = np.column_stack(coefficients)
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

    def test_multifidelity_learning_curve_accepts_supplied_low_fidelity_values(self):
        x = np.linspace(0.0, 1.0, 18)
        X = pd.DataFrame({"x": x, "x2": x**2})
        low_fidelity = 0.8 + 0.6 * x - 0.2 * x**2
        y = pd.Series(1.35 * low_fidelity + 0.15 + 0.05 * x, index=[f"sample_{i}" for i in range(len(x))])

        result = multifidelity_learning_curve(
            SimpleMultiFidelityRegressor(constant_std=0.12),
            X,
            y,
            low_fidelity_high=low_fidelity,
            train_sizes=(50, 100),
            train_size_unit="percent",
            n_splits=3,
            test_size=6,
            random_state=15,
            model_names="delta_mf",
            metric_splits=("train", "test"),
            store_predictions=True,
        )
        summary = result.summary(metric_columns=["test_RMSE"])

        self.assertIsInstance(result, LearningCurveResult)
        self.assertEqual(result.runs.shape[0], 6)
        self.assertEqual(set(result.runs["train_size_percent"]), {50.0, 100.0})
        self.assertTrue(np.all(np.isfinite(result.runs["rho"])))
        self.assertTrue(np.all(np.isfinite(result.runs["intercept"])))
        self.assertIsNotNone(result.predictions)
        self.assertIn("low_fidelity_input", result.predictions.columns)
        self.assertIn("low_fidelity_pred", result.predictions.columns)
        self.assertIn("correction_pred", result.predictions.columns)
        self.assertIn("rho", result.predictions.columns)
        self.assertIn("test_RMSE_mean", summary.columns)

    def test_multifidelity_learning_curve_accepts_low_fidelity_surrogate_data(self):
        x_low = np.linspace(-0.2, 1.2, 30)
        X_low = pd.DataFrame({"x": x_low, "x2": x_low**2})
        y_low = 0.8 + 0.6 * x_low - 0.2 * x_low**2
        x_high = np.linspace(0.0, 1.0, 16)
        X_high = pd.DataFrame({"x": x_high, "x2": x_high**2})
        low_at_high = 0.8 + 0.6 * x_high - 0.2 * x_high**2
        y_high = 1.25 * low_at_high + 0.1 + 0.04 * x_high

        result = multifidelity_learning_curve(
            SimpleMultiFidelityRegressor(constant_std=0.1),
            X_high,
            y_high,
            X_low=X_low,
            y_low=y_low,
            train_sizes=(100,),
            train_size_unit="percent",
            n_splits=2,
            test_size=4,
            random_state=21,
            model_names="delta_surrogate",
            store_predictions=True,
        )

        self.assertEqual(result.runs.shape[0], 2)
        self.assertIsNotNone(result.predictions)
        self.assertIn("low_fidelity_pred", result.predictions.columns)
        self.assertIn("low_fidelity_std", result.predictions.columns)
        self.assertIn("correction_std", result.predictions.columns)
        self.assertTrue(np.all(result.predictions["low_fidelity_std"] > 0.0))

    def test_evaluate_cokriging_train_test_split_uses_target_split_and_lower_rows(self):
        x_low = np.linspace(-0.1, 1.1, 10)
        x_target = np.linspace(0.0, 1.0, 8)
        X = pd.DataFrame({"x": np.concatenate([x_low, x_target])})
        low_y = 0.5 + 0.8 * x_low - 0.1 * x_low**2
        target_low = 0.5 + 0.8 * x_target - 0.1 * x_target**2
        target_y = 1.4 * target_low + 0.25 + 0.05 * x_target
        y = pd.Series(
            np.concatenate([low_y, target_y]),
            index=[f"row_{index}" for index in range(18)],
        )
        fidelity = np.asarray(["simulation"] * 10 + ["experiment"] * 8, dtype=object)
        sample_id = np.asarray([f"mat_{index}" for index in range(18)], dtype=object)
        train_indices = np.asarray([10, 11, 12, 13, 14])
        test_indices = np.asarray([15, 16, 17])

        result = evaluate_cokriging_train_test_split(
            SimpleRowWiseCoKrigingRegressor(constant_std=0.12),
            X,
            y,
            fidelity,
            fidelity_order=("simulation", "experiment"),
            target_fidelity="experiment",
            sample_id=sample_id,
            train_indices=train_indices,
            test_indices=test_indices,
            model_name="co-kriging",
            confidence_level=0.9,
        )

        self.assertIsInstance(result, CoKrigingTrainTestValidationResult)
        np.testing.assert_array_equal(result.train_indices, train_indices)
        np.testing.assert_array_equal(result.test_indices, test_indices)
        np.testing.assert_array_equal(result.lower_fidelity_indices, np.arange(10))
        np.testing.assert_array_equal(result.fit_indices, np.arange(15))
        self.assertEqual(result.target_fidelity, "experiment")
        self.assertEqual(result.low_fidelity, "simulation")
        self.assertEqual(result.fitted_estimator.fit_fidelity_.shape[0], 15)
        self.assertEqual(result.fitted_estimator.fit_sample_id_.shape[0], 15)
        self.assertIn("test_RMSE", result.metrics)
        self.assertIn("test_NLPD", result.metrics)
        self.assertEqual(result.train_predictions.shape[0], 5)
        self.assertEqual(result.test_predictions.shape[0], 3)
        self.assertTrue(np.all(result.predictions["fidelity"] == "experiment"))
        self.assertIn("scaled_low_fidelity_pred", result.predictions.columns)
        self.assertIn("discrepancy_pred", result.predictions.columns)
        self.assertIn("correction_pred", result.predictions.columns)
        self.assertIn("reconstructed_y_pred", result.predictions.columns)
        self.assertIn("y_lower", result.predictions.columns)
        np.testing.assert_allclose(
            result.predictions["reconstructed_y_pred"],
            result.predictions["y_pred"],
            atol=1e-12,
        )
        metrics_frame = result.metrics_frame()
        self.assertEqual(metrics_frame.loc[0, "n_lower_fidelity"], 10)
        self.assertEqual(metrics_frame.loc[0, "n_target_train"], 5)
        self.assertTrue(np.isfinite(metrics_frame.loc[0, "rho"]))

    def test_evaluate_cokriging_train_test_split_rejects_non_target_split_indices(self):
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 6)})
        y = np.linspace(0.0, 1.0, 6)
        fidelity = ["low", "low", "low", "high", "high", "high"]

        with self.assertRaisesRegex(ValueError, "target_fidelity='high'"):
            evaluate_cokriging_train_test_split(
                SimpleRowWiseCoKrigingRegressor(),
                X,
                y,
                fidelity,
                fidelity_order=("low", "high"),
                target_fidelity="high",
                train_indices=[0, 3],
                test_indices=[4, 5],
            )

    def test_cokriging_learning_curve_varies_target_rows_and_keeps_lower_rows(self):
        x_low = np.linspace(-0.1, 1.1, 12)
        x_target = np.linspace(0.0, 1.0, 10)
        X = pd.DataFrame({"x": np.concatenate([x_low, x_target])})
        low_y = 0.5 + 0.8 * x_low - 0.1 * x_low**2
        target_low = 0.5 + 0.8 * x_target - 0.1 * x_target**2
        target_y = 1.4 * target_low + 0.25 + 0.05 * x_target
        y = pd.Series(
            np.concatenate([low_y, target_y]),
            index=[f"row_{index}" for index in range(22)],
        )
        fidelity = np.asarray(["simulation"] * 12 + ["experiment"] * 10, dtype=object)

        result = cokriging_learning_curve(
            SimpleRowWiseCoKrigingRegressor(constant_std=0.12),
            X,
            y,
            fidelity,
            fidelity_order=("simulation", "experiment"),
            target_fidelity="experiment",
            train_sizes=(50, 100),
            train_size_unit="percent",
            n_splits=2,
            test_size=4,
            random_state=31,
            model_names="co-kriging",
            metric_splits=("train", "test"),
            store_predictions=True,
        )
        summary = result.summary(metric_columns=["test_RMSE"])

        self.assertIsInstance(result, LearningCurveResult)
        self.assertEqual(result.runs.shape[0], 4)
        self.assertEqual(set(result.runs["requested_train_size"]), {50.0, 100.0})
        self.assertTrue(np.all(result.runs["n_lower_fidelity"] == 12))
        self.assertTrue(np.all(result.runs["n_fit"] == result.runs["n_train"] + 12))
        self.assertEqual(set(result.runs["target_fidelity"]), {"experiment"})
        self.assertEqual(set(result.runs["low_fidelity"]), {"simulation"})
        self.assertTrue(np.all(np.isfinite(result.runs["rho"])))
        self.assertIsNotNone(result.predictions)
        self.assertTrue(np.all(result.predictions["fidelity"] == "experiment"))
        self.assertIn("scaled_low_fidelity_pred", result.predictions.columns)
        self.assertIn("discrepancy_pred", result.predictions.columns)
        self.assertIn("reconstructed_y_pred", result.predictions.columns)
        self.assertIn("repeat", result.predictions.columns)
        self.assertIn("train_size_percent", result.predictions.columns)
        self.assertIn("test_RMSE_mean", summary.columns)

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
        with self.assertRaisesRegex(ValueError, "Provide low_fidelity_high"):
            multifidelity_learning_curve(
                SimpleMultiFidelityRegressor(),
                self.X,
                self.y,
                train_sizes=(50,),
                n_splits=1,
            )
        with self.assertRaisesRegex(ValueError, "Use either low_fidelity_high"):
            multifidelity_learning_curve(
                SimpleMultiFidelityRegressor(),
                self.X,
                self.y,
                low_fidelity_high=np.ones(len(self.y)),
                X_low=self.X,
                y_low=self.y,
                train_sizes=(50,),
                n_splits=1,
            )
        with self.assertRaisesRegex(ValueError, "one value per high-fidelity sample"):
            multifidelity_learning_curve(
                SimpleMultiFidelityRegressor(),
                self.X,
                self.y,
                low_fidelity_high=np.ones(len(self.y) - 1),
                train_sizes=(50,),
                n_splits=1,
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

    def test_summarize_sparse_multitask_predictions_ignores_missing_targets(self):
        y_true = pd.DataFrame(
            {
                "strength": [1.0, np.nan, 3.0, np.nan],
                "ductility": [4.0, 3.0, np.nan, 1.0],
            }
        )
        prediction = SimpleNamespace(
            mean=np.asarray(
                [
                    [1.1, 3.8],
                    [2.0, 2.9],
                    [2.8, 2.0],
                    [4.0, 1.2],
                ]
            ),
            std=np.full((4, 2), 0.3),
            task_names=("strength", "ductility"),
        )

        summary = summarize_sparse_multitask_predictions(
            y_true,
            prediction,
            model_name="sparse",
            split="test",
            confidence_level=0.9,
        )

        self.assertEqual(summary.shape[0], 2)
        self.assertEqual(summary["task"].tolist(), ["strength", "ductility"])
        self.assertEqual(summary["n_samples"].tolist(), [4, 4])
        self.assertEqual(summary["n_observed"].tolist(), [2, 3])
        self.assertEqual(summary["n_missing"].tolist(), [2, 1])
        self.assertIn("observed_coverage", summary.columns)
        self.assertTrue(np.all(np.isfinite(summary["RMSE"])))

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

    def test_evaluate_sparse_multitask_train_test_split_returns_observed_tables(self):
        x = np.linspace(0.0, 1.0, 12)
        X = pd.DataFrame({"x": x})
        y = pd.DataFrame(
            {
                "strength": 2.0 * x + 0.5,
                "ductility": -1.5 * x + 2.0,
            },
            index=[f"sample_{index}" for index in range(len(x))],
        )
        y.iloc[[1, 3, 9], y.columns.get_loc("strength")] = np.nan
        y.iloc[[2, 6, 10], y.columns.get_loc("ductility")] = np.nan

        result = evaluate_sparse_multitask_train_test_split(
            SparseMultiOutputMeanStdRegressor(constant_std=0.2),
            X,
            y,
            train_indices=np.arange(8),
            test_indices=np.arange(8, 12),
            model_name="sparse_multi_linear",
            confidence_level=0.9,
        )

        self.assertIsInstance(result, SparseMultitaskTrainTestValidationResult)
        self.assertEqual(result.task_metrics.shape[0], 4)
        self.assertEqual(set(result.task_metrics["split"]), {"train", "test"})
        self.assertEqual(set(result.task_metrics["task"]), {"strength", "ductility"})
        self.assertIn("n_observed", result.task_metrics.columns)
        self.assertIn("missing_fraction", result.task_metrics.columns)
        self.assertEqual(result.train_predictions.shape[0], 16)
        self.assertEqual(result.test_predictions.shape[0], 8)
        self.assertIn("observed", result.predictions.columns)
        self.assertTrue(np.all(np.isfinite(result.observed_predictions["y_true"])))
        self.assertLess(result.observed_predictions.shape[0], result.predictions.shape[0])
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
        with self.assertRaisesRegex(ValueError, "finite values"):
            summarize_multitask_predictions(
                np.asarray([[1.0, np.nan], [2.0, 3.0]]),
                np.ones((2, 2)),
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            summarize_sparse_multitask_predictions(
                np.asarray([[1.0, np.nan], [2.0, 3.0]]),
                np.ones((2, 1)),
            )


if __name__ == "__main__":
    unittest.main()
