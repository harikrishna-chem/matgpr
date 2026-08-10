from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from matgpr import log_experiment_result


class ExperimentLoggingTests(unittest.TestCase):
    def test_log_experiment_result_appends_same_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.csv"

            log_experiment_result(
                {"RMSE": 1.2, "MAE": 0.8},
                metadata={"model": "standard_gpr"},
                path=path,
            )
            log_experiment_result(
                {"RMSE": 0.9, "MAE": 0.6},
                metadata={"model": "physics_gpr"},
                path=path,
            )

            log = pd.read_csv(path)

        self.assertEqual(log.columns.tolist(), ["model", "RMSE", "MAE"])
        self.assertEqual(log["model"].tolist(), ["standard_gpr", "physics_gpr"])
        self.assertTrue(np.allclose(log["RMSE"], [1.2, 0.9]))

    def test_log_experiment_result_preserves_round_trip_when_schema_evolves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.csv"

            log_experiment_result(
                {"RMSE": 1.2},
                metadata={"model": "standard_gpr", "train_fraction": 0.2},
                path=path,
            )
            log_experiment_result(
                {"MAE": 0.5, "RMSE": 0.8},
                metadata={"train_fraction": 0.2, "model": "physics_gpr"},
                path=path,
            )
            log_experiment_result(
                {"RMSE": 0.7},
                metadata={"model": "physics_gpr", "kernel": "matern"},
                path=path,
            )

            log = pd.read_csv(path)

        self.assertEqual(log.columns.tolist(), ["model", "train_fraction", "RMSE", "MAE", "kernel"])
        self.assertEqual(log.shape, (3, 5))
        self.assertTrue(np.isnan(log.loc[0, "MAE"]))
        self.assertEqual(log.loc[1, "MAE"], 0.5)
        self.assertTrue(np.isnan(log.loc[1, "kernel"]))
        self.assertEqual(log.loc[2, "kernel"], "matern")

    def test_log_experiment_result_rejects_duplicate_metadata_metric_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.csv"

            with self.assertRaisesRegex(ValueError, "Duplicate experiment result column"):
                log_experiment_result(
                    {"model": 0.7},
                    metadata={"model": "standard_gpr"},
                    path=path,
                )


if __name__ == "__main__":
    unittest.main()
