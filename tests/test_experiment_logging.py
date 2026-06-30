from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from matgpr import (
    BOCampaignState,
    apply_candidate_duplicate_policy,
    append_closed_loop_records,
    infer_next_bo_iteration,
    load_closed_loop_log,
    log_bo_recommendations,
    log_observations,
    log_selected_experiments,
    resume_bo_campaign,
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

    def test_resume_bo_campaign_filters_available_and_pending_candidates(self):
        log = pd.DataFrame(
            {
                "matgpr_campaign_id": ["screen", "screen", "screen", "other"],
                "matgpr_iteration": [0, 0, 1, 0],
                "matgpr_record_type": [
                    "recommendation",
                    "selection",
                    "observation",
                    "selection",
                ],
                "candidate_id": ["a", "b", "a", "z"],
                "target": [np.nan, np.nan, 1.5, np.nan],
            }
        )
        candidate_pool = pd.DataFrame(
            {
                "candidate_id": ["a", "b", "c", "d"],
                "descriptor_x": [0.0, 0.5, 1.0, 1.5],
            }
        )

        state = resume_bo_campaign(
            log,
            campaign_id="screen",
            candidate_pool=candidate_pool,
            key_columns=("candidate_id",),
        )

        self.assertIsInstance(state, BOCampaignState)
        self.assertEqual(state.current_iteration, 1)
        self.assertEqual(state.last_recommendation_iteration, 0)
        self.assertEqual(state.next_iteration, 1)
        self.assertEqual(state.pending_experiments["candidate_id"].tolist(), ["b"])
        self.assertEqual(state.completed_experiments["candidate_id"].tolist(), ["a"])
        self.assertEqual(state.unavailable_candidates["candidate_id"].tolist(), ["a", "b"])
        self.assertEqual(state.available_candidates["candidate_id"].tolist(), ["c", "d"])
        self.assertEqual(state.candidate_pool_size, 4)

        filtered = apply_candidate_duplicate_policy(
            candidate_pool,
            state.duplicate_policy(),
        )
        self.assertEqual(filtered["candidate_id"].tolist(), ["c", "d"])

    def test_resume_bo_campaign_supports_missing_log_for_first_iteration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "not_created_yet.csv"
            candidate_pool = pd.DataFrame(
                {
                    "candidate_id": ["a", "b"],
                    "descriptor_x": [0.0, 1.0],
                }
            )

            state = resume_bo_campaign(
                log_path,
                campaign_id="new_screen",
                candidate_pool=candidate_pool,
                key_columns="candidate_id",
            )

        self.assertIsNone(state.current_iteration)
        self.assertIsNone(state.last_recommendation_iteration)
        self.assertEqual(state.next_iteration, 0)
        self.assertTrue(state.pending_experiments.empty)
        self.assertTrue(state.unavailable_candidates.empty)
        self.assertEqual(state.available_candidates["candidate_id"].tolist(), ["a", "b"])

    def test_infer_next_bo_iteration_uses_recommendation_rows(self):
        log = pd.DataFrame(
            {
                "matgpr_campaign_id": ["screen", "screen", "screen", "other"],
                "matgpr_iteration": [0, 1, 2, 5],
                "matgpr_record_type": [
                    "recommendation",
                    "observation",
                    "recommendation",
                    "recommendation",
                ],
            }
        )

        self.assertEqual(infer_next_bo_iteration(log, campaign_id="screen"), 3)
        self.assertEqual(infer_next_bo_iteration(log, campaign_id="missing"), 0)
        self.assertEqual(
            infer_next_bo_iteration(
                log[log["matgpr_record_type"] == "observation"],
                campaign_id="screen",
            ),
            0,
        )

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

    def test_resume_bo_campaign_validation_errors_are_explicit(self):
        log = pd.DataFrame(
            {
                "matgpr_campaign_id": ["screen"],
                "matgpr_iteration": [0],
                "matgpr_record_type": ["selection"],
                "candidate_id": ["a"],
            }
        )

        with self.assertRaises(ValueError):
            resume_bo_campaign(log, campaign_id="screen", key_columns=())
        with self.assertRaises(ValueError):
            resume_bo_campaign(
                log,
                campaign_id="screen",
                key_columns=("candidate_id", "candidate_id"),
            )
        with self.assertRaises(ValueError):
            resume_bo_campaign(
                log,
                campaign_id="screen",
                key_columns=("missing_id",),
            )
        with self.assertRaises(ValueError):
            resume_bo_campaign(
                log,
                campaign_id="screen",
                candidate_pool=pd.DataFrame({"other_id": ["a"]}),
            )
        with self.assertRaises(ValueError):
            resume_bo_campaign(
                pd.DataFrame(
                    {
                        "matgpr_campaign_id": ["screen"],
                        "matgpr_record_type": ["recommendation"],
                    }
                ),
                campaign_id="screen",
            )


if __name__ == "__main__":
    unittest.main()
