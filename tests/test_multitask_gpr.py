from __future__ import annotations

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matgpr")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matgpr-cache")

import numpy as np

from matgpr.multitask_gpr import (
    MultitaskGPyTorchResult,
    fit_multitask_gpytorch_gpr,
    predict_multitask_gpytorch_gpr,
    train_multitask_gpytorch_gpr,
)


class MultitaskGPyTorchTests(unittest.TestCase):
    def test_fit_result_predicts_all_tasks_in_original_units(self):
        x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = np.column_stack(
            [
                1.5 * x.ravel() + 0.2,
                -0.5 * x.ravel() + 1.1,
            ]
        )

        result = fit_multitask_gpytorch_gpr(
            x,
            y,
            task_names=("strength", "ductility"),
            kernel="rbf",
            training_iter=3,
            initial_noise=0.05,
            initial_task_noises=(0.05, 0.08),
            verbose=False,
        )
        prediction = result.predict(x[:3], confidence_level=0.90)

        self.assertIsInstance(result, MultitaskGPyTorchResult)
        self.assertEqual(result.task_names, ("strength", "ductility"))
        self.assertEqual(result.num_tasks, 2)
        self.assertEqual(len(result.loss_history), 3)
        self.assertEqual(prediction.mean.shape, (3, 2))
        self.assertEqual(prediction.std.shape, (3, 2))
        self.assertEqual(prediction.lower.shape, (3, 2))
        self.assertEqual(prediction.upper.shape, (3, 2))
        self.assertEqual(prediction.task_names, ("strength", "ductility"))

    def test_train_wrapper_preserves_tuple_return(self):
        x = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        y = np.column_stack([x.ravel() ** 2, 1.0 - x.ravel()])

        model, likelihood = train_multitask_gpytorch_gpr(
            x,
            y,
            task_names=("hardness", "modulus"),
            training_iter=2,
            verbose=False,
        )
        mean, std = predict_multitask_gpytorch_gpr(model, likelihood, x[:2])

        self.assertEqual(mean.shape, (2, 2))
        self.assertEqual(std.shape, (2, 2))

    def test_rejects_single_target_or_incomplete_targets(self):
        x = np.linspace(0.0, 1.0, 5).reshape(-1, 1)

        with self.assertRaisesRegex(ValueError, "2D target matrix"):
            fit_multitask_gpytorch_gpr(
                x,
                x.ravel(),
                training_iter=2,
                verbose=False,
            )

        y_with_nan = np.column_stack([x.ravel(), 1.0 - x.ravel()])
        y_with_nan[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "complete target observations"):
            fit_multitask_gpytorch_gpr(
                x,
                y_with_nan,
                training_iter=2,
                verbose=False,
            )

    def test_rejects_task_metadata_mismatches(self):
        x = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
        y = np.column_stack([x.ravel(), 1.0 - x.ravel()])

        with self.assertRaisesRegex(ValueError, "task_names"):
            fit_multitask_gpytorch_gpr(
                x,
                y,
                task_names=("only_one",),
                training_iter=2,
                verbose=False,
            )

        with self.assertRaisesRegex(ValueError, "task_covar_rank"):
            fit_multitask_gpytorch_gpr(
                x,
                y,
                task_covar_rank=3,
                training_iter=2,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
