from __future__ import annotations

import os
import unittest
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from matgpr import CompositionFeaturizer, MatGPRRegressor, PhysicsInformedGPRRegressor
from matgpr.gpytorch_gpr import GPyTorchPrediction


def linear_physics_equation(features, parameters):
    return parameters["slope"] * features["x"] + parameters["intercept"]


class MatGPRRegressorTests(unittest.TestCase):
    def test_standard_estimator_fit_predict_and_score(self):
        x = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y = 1.5 * x.ravel() + 0.2

        estimator = MatGPRRegressor(
            kernel="rbf",
            training_iter=3,
            initial_noise=0.05,
            random_state=7,
        )
        estimator.fit(x, y)
        mean, std = estimator.predict(x[:3], return_std=True)
        prediction = estimator.predict_distribution(x[:3], confidence_level=0.90)
        score = estimator.score(x, y)

        self.assertEqual(estimator.n_features_in_, 1)
        self.assertEqual(len(estimator.loss_history_), 3)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertIsInstance(prediction, GPyTorchPrediction)
        self.assertEqual(prediction.lower.shape, (3,))
        self.assertEqual(prediction.upper.shape, (3,))
        self.assertTrue(np.isfinite(score))

    def test_standard_estimator_is_cloneable(self):
        estimator = MatGPRRegressor(kernel="matern", training_iter=2, dtype="float32")
        cloned = clone(estimator)

        self.assertEqual(cloned.kernel, "matern")
        self.assertEqual(cloned.training_iter, 2)
        self.assertEqual(cloned.dtype, "float32")

    def test_standard_estimator_passes_sklearn_estimator_checks(self):
        estimator = MatGPRRegressor(
            kernel="rbf",
            training_iter=1,
            initial_noise=0.05,
            random_state=3,
            dtype="float64",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_estimator(estimator)

    def test_rejects_prediction_with_wrong_feature_count(self):
        x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = x.ravel()
        estimator = MatGPRRegressor(training_iter=2).fit(x, y)

        with self.assertRaises(ValueError):
            estimator.predict(np.zeros((2, 2)))

    def test_refit_with_array_clears_dataframe_feature_names(self):
        dataframe_x = pd.DataFrame({"x": np.linspace(0.0, 1.0, 8)})
        array_x = dataframe_x.to_numpy()
        y = dataframe_x["x"].to_numpy()

        estimator = MatGPRRegressor(training_iter=2)
        estimator.fit(dataframe_x, y)
        self.assertTrue(hasattr(estimator, "feature_names_in_"))

        estimator.fit(array_x, y)
        self.assertFalse(hasattr(estimator, "feature_names_in_"))


class PhysicsInformedGPRRegressorTests(unittest.TestCase):
    def test_physics_estimator_fit_predict_and_parameter_report(self):
        x = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y = 2.0 * x.ravel() + 0.5

        estimator = PhysicsInformedGPRRegressor(
            equation=linear_physics_equation,
            feature_indices={"x": 0},
            learnable_parameters={"slope": 1.0},
            fixed_parameters={"intercept": 0.5},
            positive_parameters=("slope",),
            kernel="rbf",
            training_iter=3,
            initial_noise=0.05,
            random_state=11,
        )
        estimator.fit(x, y)
        mean, lower, upper = estimator.predict_interval(x[:2], confidence_level=0.80)

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(lower.shape, (2,))
        self.assertEqual(upper.shape, (2,))
        self.assertIn("slope", estimator.learned_physics_parameters_)
        self.assertIn("intercept", estimator.learned_physics_parameters_)
        self.assertGreater(estimator.learned_physics_parameters_["slope"], 0.0)

    def test_physics_feature_names_are_mapped_from_dataframe_columns(self):
        x = pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, 8),
                "descriptor": np.linspace(1.0, 2.0, 8),
            }
        )
        y = 1.2 * x["x"].to_numpy() + 0.3

        estimator = PhysicsInformedGPRRegressor(
            equation=linear_physics_equation,
            physics_features=("x",),
            learnable_parameters={"slope": 1.0, "intercept": 0.0},
            positive_parameters=("slope",),
            training_iter=2,
            dtype=torch.float64,
        )
        estimator.fit(x, y)

        self.assertEqual(estimator.feature_names_in_.tolist(), ["x", "descriptor"])
        self.assertEqual(estimator.mean_module_.feature_indices, {"x": 0})

        with self.assertRaises(ValueError):
            estimator.predict(x[["descriptor", "x"]])

    def test_requires_equation_before_fitting(self):
        x = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        y = x.ravel()
        estimator = PhysicsInformedGPRRegressor(feature_indices={"x": 0}, training_iter=2)

        with self.assertRaises(ValueError):
            estimator.fit(x, y)


class EstimatorPipelineTests(unittest.TestCase):
    def test_standard_estimator_works_in_pipeline_grid_search(self):
        x = np.linspace(0.0, 1.0, 12)
        X = pd.DataFrame(
            {
                "x": x,
                "x_squared": x**2,
            }
        )
        y = 1.5 * x + 0.4 * x**2
        pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "gpr",
                    MatGPRRegressor(
                        training_iter=1,
                        initial_noise=0.05,
                        random_state=5,
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid={"gpr__kernel": ["rbf", "matern"], "gpr__ard": [False]},
            cv=2,
        )

        search.fit(X, y)
        predictions = search.predict(X.iloc[:3])

        self.assertEqual(predictions.shape, (3,))
        self.assertIn(search.best_params_["gpr__kernel"], {"rbf", "matern"})
        self.assertTrue(np.isfinite(search.best_score_))

    def test_composition_featurizer_pipeline_fits_gpr(self):
        X = pd.DataFrame(
            {
                "formula": ["B4C", "BN", "SiC", "AlN", "GaN", "Al2O3"],
                "load": [0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
            }
        )
        y = np.array([28.0, 33.0, 25.0, 18.0, 14.0, 20.0])
        feature_builder = ColumnTransformer(
            transformers=[
                (
                    "composition",
                    CompositionFeaturizer(
                        properties=("atomic_number",),
                        statistics=("fwm", "max"),
                        return_dataframe=False,
                    ),
                    ["formula"],
                ),
                ("load", "passthrough", ["load"]),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("features", feature_builder),
                ("scale", StandardScaler()),
                (
                    "gpr",
                    MatGPRRegressor(
                        kernel="rbf",
                        training_iter=1,
                        initial_noise=0.05,
                        random_state=13,
                    ),
                ),
            ]
        )

        pipeline.fit(X, y)
        predictions, std = pipeline.predict(X.iloc[:2], return_std=True)

        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertEqual(pipeline.named_steps["features"].transform(X).shape[1], 3)

    def test_physics_informed_estimator_works_in_pipeline(self):
        x = np.linspace(0.0, 1.0, 10)
        X = pd.DataFrame(
            {
                "x": x,
                "descriptor": np.linspace(1.0, 2.0, 10),
            }
        )
        y = 2.0 * x + 0.5
        pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "gpr",
                    PhysicsInformedGPRRegressor(
                        equation=linear_physics_equation,
                        feature_indices={"x": 0},
                        learnable_parameters={"slope": 1.0, "intercept": 0.0},
                        positive_parameters=("slope",),
                        kernel="rbf",
                        training_iter=1,
                        initial_noise=0.05,
                        random_state=17,
                    ),
                ),
            ]
        )

        pipeline.fit(X, y)
        predictions = pipeline.predict(X.iloc[:2])

        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(pipeline.named_steps["gpr"].mean_module_.feature_indices, {"x": 0})


if __name__ == "__main__":
    unittest.main()
