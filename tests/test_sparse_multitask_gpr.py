from __future__ import annotations

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np
import pandas as pd

from matgpr.sparse_multitask_gpr import (
    SparseMultitaskGPyTorchResult,
    SparseMultitaskObservationData,
    fit_sparse_multitask_gpytorch_gpr,
    prepare_sparse_multitask_observations,
    train_sparse_multitask_gpytorch_gpr,
)


class SparseMultitaskGPyTorchTests(unittest.TestCase):
    def test_prepare_sparse_observations_preserves_partial_targets(self):
        x = pd.DataFrame({"descriptor": [0.0, 0.5, 1.0]})
        y = pd.DataFrame(
            {
                "strength": [1.0, np.nan, 2.0],
                "ductility": [np.nan, 0.6, 0.8],
            }
        )

        observations = prepare_sparse_multitask_observations(
            x,
            y,
            min_observations_per_task=1,
        )

        self.assertIsInstance(observations, SparseMultitaskObservationData)
        self.assertEqual(observations.X_observed.shape, (4, 1))
        self.assertEqual(observations.task_names, ("strength", "ductility"))
        self.assertEqual(observations.task_observation_counts.tolist(), [2, 2])
        self.assertEqual(observations.sample_indices.tolist(), [0, 1, 2, 2])
        self.assertEqual(observations.task_indices.tolist(), [0, 1, 0, 1])

    def test_fit_predict_sparse_multitask_model(self):
        x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = np.column_stack(
            [
                1.5 * x.ravel() + 0.2,
                -0.5 * x.ravel() + 1.1,
            ]
        )
        y[1, 0] = np.nan
        y[4, 1] = np.nan

        result = fit_sparse_multitask_gpytorch_gpr(
            x,
            y,
            task_names=("strength", "ductility"),
            kernel="rbf",
            training_iter=3,
            initial_noise=0.05,
            verbose=False,
        )
        prediction = result.predict(x[:3], confidence_level=0.90)

        self.assertIsInstance(result, SparseMultitaskGPyTorchResult)
        self.assertEqual(result.task_names, ("strength", "ductility"))
        self.assertEqual(result.observation_data.task_observation_counts.tolist(), [7, 7])
        self.assertEqual(len(result.loss_history), 3)
        self.assertEqual(prediction.mean.shape, (3, 2))
        self.assertEqual(prediction.std.shape, (3, 2))
        self.assertEqual(prediction.lower.shape, (3, 2))
        self.assertEqual(prediction.task_names, ("strength", "ductility"))

    def test_train_wrapper_preserves_tuple_return(self):
        x = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        y = np.column_stack([x.ravel(), 1.0 - x.ravel()])
        y[0, 1] = np.nan

        model, likelihood = train_sparse_multitask_gpytorch_gpr(
            x,
            y,
            task_names=("hardness", "modulus"),
            training_iter=2,
            verbose=False,
        )

        self.assertEqual(model.num_tasks, 2)
        self.assertIsNotNone(likelihood)

    def test_rejects_tasks_with_too_few_observations(self):
        x = np.linspace(0.0, 1.0, 4).reshape(-1, 1)
        y = np.column_stack([x.ravel(), 1.0 - x.ravel()])
        y[1:, 1] = np.nan

        with self.assertRaisesRegex(ValueError, "low-count"):
            prepare_sparse_multitask_observations(
                x,
                y,
                min_observations_per_task=2,
            )


if __name__ == "__main__":
    unittest.main()
