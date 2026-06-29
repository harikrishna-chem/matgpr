from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from matgpr import (
    append_closed_loop_records,
    load_closed_loop_log,
    log_bo_recommendations,
    log_observations,
    log_selected_experiments,
    summarize_closed_loop_log,
)


class ExperimentLoggingTests(unittest.TestCase):
    def test_closed_loop_logging_appends_recommendations_selections_and_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "bo_campaign.csv"
            timestamp = datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)

            recommendations = pd.DataFrame(
                {
                    "candidate_id": ["mat_1", "mat_2"],
                    "matgpr_rank": [1, 2],
                    "matgpr_acquisition": [1.2, 0.9],
                }
            )
            appended = log_bo_recommendations(
                recommendations,
                path=log_path,
                campaign_id="opv_screen",
                iteration=0,
                model_name="physics_informed_gpr",
                acquisition_function="log_expected_improvement",
                timestamp=timestamp,
            )

            self.assertEqual(appended["matgpr_record_type"].tolist(), ["recommendation", "recommendation"])
            self.assertIn("matgpr_metadata_model_name", appended.columns)
            self.assertIn("matgpr_metadata_acquisition_function", appended.columns)

            selected = pd.DataFrame({"candidate_id": ["mat_1"], "batch": [1]})
            log_selected_experiments(
                selected,
                path=log_path,
                campaign_id="opv_screen",
                iteration=0,
                selection_policy="top_ranked",
                timestamp=timestamp,
            )

            observations = pd.DataFrame(
                {
                    "candidate_id": ["mat_1"],
                    "pce_percent": [12.4],
                    "reported_std": [0.3],
                }
            )
            log_observations(
                observations,
                path=log_path,
                campaign_id="opv_screen",
                iteration=1,
                target_column="pce_percent",
                metadata={"lab": "internal"},
                timestamp=timestamp,
            )

            log = load_closed_loop_log(log_path)

            self.assertEqual(log.shape[0], 4)
            self.assertEqual(
                log["matgpr_record_type"].tolist(),
                ["recommendation", "recommendation", "selection", "observation"],
            )
            self.assertEqual(log["matgpr_campaign_id"].unique().tolist(), ["opv_screen"])
            self.assertEqual(log["matgpr_timestamp_utc"].unique().tolist(), [timestamp.isoformat()])
            self.assertIn("pce_percent", log.columns)
            self.assertIn("matgpr_metadata_selection_policy", log.columns)
            self.assertTrue(np.isnan(log.loc[0, "pce_percent"]))
            self.assertEqual(log.loc[3, "pce_percent"], 12.4)

    def test_summarize_closed_loop_log_counts_records_and_target_values(self):
        log = pd.DataFrame(
            {
                "matgpr_campaign_id": ["screen", "screen", "screen", "other"],
                "matgpr_iteration": [0, 0, 1, 0],
                "matgpr_record_type": [
                    "recommendation",
                    "recommendation",
                    "observation",
                    "observation",
                ],
                "target": [np.nan, np.nan, 1.5, 9.0],
            }
        )

        summary = summarize_closed_loop_log(
            log,
            campaign_id="screen",
            target_column="target",
        )

        self.assertEqual(summary["matgpr_record_count"].tolist(), [2, 1])
        self.assertEqual(summary["matgpr_record_type"].tolist(), ["recommendation", "observation"])
        self.assertEqual(summary["matgpr_target_count"].tolist(), [0, 1])
        self.assertTrue(np.isnan(summary.loc[0, "matgpr_target_mean"]))
        self.assertEqual(summary.loc[1, "matgpr_target_mean"], 1.5)
        self.assertEqual(summary.loc[1, "matgpr_target_min"], 1.5)
        self.assertEqual(summary.loc[1, "matgpr_target_max"], 1.5)

    def test_append_closed_loop_records_supports_custom_record_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "custom.csv"

            appended = append_closed_loop_records(
                pd.DataFrame({"candidate_id": ["x"], "note": ["manual review"]}),
                path=log_path,
                campaign_id="campaign",
                iteration=2,
                record_type="manual-check",
                metadata={"operator name": "Hari"},
                timestamp="2026-06-29T12:00:00+00:00",
            )

            self.assertEqual(appended["matgpr_record_type"].tolist(), ["manual_check"])
            self.assertIn("matgpr_metadata_operator_name", appended.columns)

    def test_closed_loop_logging_validation_errors_are_explicit(self):
        records = pd.DataFrame({"candidate_id": ["a"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "campaign.csv"

            with self.assertRaises(TypeError):
                append_closed_loop_records(
                    [{"candidate_id": "a"}],
                    path=log_path,
                    campaign_id="screen",
                    iteration=0,
                    record_type="recommendation",
                )
            with self.assertRaises(ValueError):
                append_closed_loop_records(
                    pd.DataFrame(),
                    path=log_path,
                    campaign_id="screen",
                    iteration=0,
                    record_type="recommendation",
                )
            with self.assertRaises(ValueError):
                append_closed_loop_records(
                    records,
                    path=log_path,
                    campaign_id=" ",
                    iteration=0,
                    record_type="recommendation",
                )
            with self.assertRaises(ValueError):
                append_closed_loop_records(
                    records,
                    path=log_path,
                    campaign_id="screen",
                    iteration=-1,
                    record_type="recommendation",
                )
            with self.assertRaises(ValueError):
                append_closed_loop_records(
                    pd.DataFrame({"matgpr_campaign_id": ["existing"]}),
                    path=log_path,
                    campaign_id="screen",
                    iteration=0,
                    record_type="recommendation",
                )
            with self.assertRaises(TypeError):
                append_closed_loop_records(
                    records,
                    path=log_path,
                    campaign_id="screen",
                    iteration=0,
                    record_type="recommendation",
                    metadata={"list_value": [1, 2]},
                )
            with self.assertRaises(ValueError):
                log_bo_recommendations(
                    records,
                    path=log_path,
                    campaign_id="screen",
                    iteration=0,
                    model_name="gpr",
                    metadata={"model name": "duplicate"},
                )
            with self.assertRaises(ValueError):
                summarize_closed_loop_log(records)
            with self.assertRaises(ValueError):
                summarize_closed_loop_log(
                    pd.DataFrame(
                        {
                            "matgpr_campaign_id": ["screen"],
                            "matgpr_iteration": [0],
                            "matgpr_record_type": ["observation"],
                        }
                    ),
                    target_column="missing_target",
                )


if __name__ == "__main__":
    unittest.main()
