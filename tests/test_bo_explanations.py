from __future__ import annotations

import pandas as pd
import pytest

from matgpr import (
    BOExplanationTables,
    BOWhatIfSlice,
    build_bo_explanation_tables,
    build_bo_what_if_slice,
)


def test_build_bo_explanation_tables_returns_plot_ready_rows() -> None:
    ranked = _ranked_candidates()
    reference = pd.DataFrame(
        {
            "laser_power_W": [100.0, 200.0],
            "scan_speed_mm_s": [300.0, 900.0],
            "hatch_spacing_mm": [0.08, 0.12],
        }
    )

    result = build_bo_explanation_tables(
        ranked,
        top_k=2,
        output_constraints=(
            {
                "target_name": "density",
                "lower_bound": 0.97,
                "probability_threshold": 0.6,
            },
        ),
        reference_features=reference,
        feature_columns=(
            "laser_power_W",
            "scan_speed_mm_s",
            "hatch_spacing_mm",
        ),
    )

    assert isinstance(result, BOExplanationTables)
    assert result.recommendation_table["candidate_or_input_id"].tolist() == [
        "candidate-b",
        "candidate-a",
    ]
    assert result.acquisition_table["is_recommended"].tolist() == [True, True, False]
    assert len(result.constraint_table) == 3
    assert result.constraint_table["is_feasible"].tolist() == [True, True, False]
    assert set(result.feature_space_table["point_role"]) == {
        "reference",
        "recommended",
        "candidate",
    }
    assert result.metadata["embedding_method"] == "pca"
    assert len(result.metadata["embedding_explained_variance_ratio"]) == 2
    assert result.metadata["reference_point_count"] == 2


def test_build_bo_explanation_tables_retains_two_feature_axes() -> None:
    result = build_bo_explanation_tables(
        _ranked_candidates(),
        top_k=1,
        feature_columns=("laser_power_W", "scan_speed_mm_s"),
    )

    feature_rows = result.feature_space_table
    assert feature_rows["embedding_method"].unique().tolist() == ["selected_features"]
    assert feature_rows["feature_x"].unique().tolist() == ["laser_power_W"]
    assert feature_rows["feature_y"].unique().tolist() == ["scan_speed_mm_s"]
    assert feature_rows["x"].tolist() == [240.0, 120.0, 360.0]


def test_build_bo_explanation_tables_validates_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_bo_explanation_tables(
            pd.DataFrame({"candidate_id": ["candidate-a"]}),
            top_k=1,
        )


def test_build_bo_explanation_tables_sorts_by_rank_before_selecting() -> None:
    ranked = _ranked_candidates().iloc[[2, 0, 1]].reset_index(drop=True)

    result = build_bo_explanation_tables(ranked, top_k=1)

    assert result.recommendation_table["candidate_or_input_id"].tolist() == ["candidate-b"]
    assert result.acquisition_table["rank"].tolist() == [1, 2, 3]


def test_build_bo_explanation_tables_rejects_duplicate_ranks() -> None:
    ranked = _ranked_candidates()
    ranked.loc[1, "matgpr_rank"] = 1

    with pytest.raises(ValueError, match="must be unique"):
        build_bo_explanation_tables(ranked, top_k=1)


def test_explanation_tables_use_package_batch_selection_annotations() -> None:
    ranked = _ranked_candidates()
    ranked["row_id"] = ranked["candidate_id"]
    ranked["matgpr_acquisition_rank"] = ranked["matgpr_rank"]
    ranked["matgpr_eligible_rank"] = [1, 2, 3]
    ranked["matgpr_selected_for_batch"] = [False, True, False]
    ranked["matgpr_batch_order"] = [pd.NA, 1, pd.NA]
    ranked["matgpr_eligible_for_recommendation"] = True
    ranked["matgpr_duplicate_status"] = "novel_candidate"
    ranked["matgpr_applicability_status"] = ["inside_domain", "inside_domain", "unknown"]
    ranked["matgpr_applicability"] = [
        {"status": "inside_domain"},
        {"status": "inside_domain"},
        None,
    ]

    result = build_bo_explanation_tables(
        ranked,
        top_k=2,
        output_constraints=(
            {
                "target_name": "density",
                "lower_bound": 0.97,
                "probability_threshold": 0.6,
            },
        ),
    )

    assert result.recommendation_table["candidate_or_input_id"].tolist() == ["candidate-a"]
    assert result.recommendation_table["batch_order"].tolist() == [1]
    assert result.acquisition_table["selected_for_batch"].tolist() == [
        False,
        True,
        False,
    ]
    assert result.recommendation_table["candidate_or_input_id"].tolist() == ["candidate-a"]
    candidate_feature_rows = result.feature_space_table.loc[
        result.feature_space_table["point_role"] != "reference"
    ]
    assert candidate_feature_rows["applicability_status"].tolist() == [
        "inside_domain",
        "inside_domain",
        "unknown",
    ]
    assert result.constraint_table["duplicate_status"].tolist() == [
        "novel_candidate",
        "novel_candidate",
        "novel_candidate",
    ]
    assert result.metadata["batch_complete"] is False


def test_what_if_slice_preserves_package_batch_and_applicability_annotations() -> None:
    ranked = _what_if_candidates()
    ranked["matgpr_selected_for_batch"] = [False, False, True, False]
    ranked["matgpr_batch_order"] = [pd.NA, pd.NA, 1, pd.NA]
    ranked["matgpr_applicability_status"] = ["inside_domain"] * 4
    ranked["matgpr_applicability"] = [
        {"status": "inside_domain", "nearest_neighbor_distance": float(index)} for index in range(4)
    ]

    result = build_bo_what_if_slice(
        ranked,
        x_feature="laser_power_W",
        fixed_feature_values={"scan_speed_mm_s": 500.0},
        top_k=2,
    )

    assert result.rows["candidate_or_input_id"].tolist() == [
        "candidate-3",
        "candidate-1",
    ]
    assert result.rows["is_recommended"].tolist() == [True, False]
    assert result.rows["batch_order"].iloc[0] == 1
    assert pd.isna(result.rows["batch_order"].iloc[1])
    assert result.rows["applicability_status"].tolist() == [
        "inside_domain",
        "inside_domain",
    ]


def test_build_bo_what_if_slice_returns_conditioned_one_feature_rows() -> None:
    result = build_bo_what_if_slice(
        _what_if_candidates(),
        x_feature="laser_power_W",
        fixed_feature_values={"scan_speed_mm_s": 500.0},
        top_k=2,
        uncertainty_multiplier=2.0,
    )

    assert isinstance(result, BOWhatIfSlice)
    assert result.rows["x"].tolist() == [100.0, 200.0]
    assert result.rows["y"].tolist() == [None, None]
    assert result.rows["predicted_lower"].tolist() == pytest.approx([0.93, 0.97])
    assert result.rows["predicted_upper"].tolist() == pytest.approx([1.01, 1.01])
    assert result.rows["point_role"].tolist() == ["candidate", "recommended"]
    assert result.metadata["slice_dimension"] == 1
    assert result.metadata["fixed_feature_values"] == {"scan_speed_mm_s": 500.0}
    assert result.metadata["constant_context_values"]["alloy"] == "316L"
    assert result.metadata["interpolation_method"] == "none"


def test_build_bo_what_if_slice_returns_sorted_two_feature_surface() -> None:
    result = build_bo_what_if_slice(
        _what_if_candidates().iloc[[3, 0, 2, 1]],
        x_feature="laser_power_W",
        y_feature="scan_speed_mm_s",
        top_k=1,
    )

    assert result.rows[["x", "y"]].values.tolist() == [
        [100.0, 500.0],
        [200.0, 500.0],
        [100.0, 1000.0],
        [200.0, 1000.0],
    ]
    assert result.rows["candidate_or_input_id"].tolist() == [
        "candidate-3",
        "candidate-1",
        "candidate-4",
        "candidate-2",
    ]
    assert result.metadata["slice_dimension"] == 2
    assert result.metadata["matched_candidate_count"] == 4


def test_build_bo_what_if_slice_rejects_ambiguous_hidden_context() -> None:
    ranked = _what_if_candidates()
    duplicate = ranked.iloc[[0]].copy()
    duplicate["candidate_id"] = "candidate-5"
    duplicate["alloy"] = "Ti-6Al-4V"
    duplicate["matgpr_rank"] = 5
    duplicate["matgpr_candidate_index"] = 4
    ranked = pd.concat((ranked, duplicate), ignore_index=True)

    with pytest.raises(ValueError, match="additional fixed_feature_values"):
        build_bo_what_if_slice(
            ranked,
            x_feature="laser_power_W",
            y_feature="scan_speed_mm_s",
        )


def test_build_bo_what_if_slice_validates_axes_and_fixed_context() -> None:
    ranked = _what_if_candidates()

    with pytest.raises(ValueError, match="must be numeric"):
        build_bo_what_if_slice(ranked, x_feature="alloy")
    with pytest.raises(ValueError, match="identifier column"):
        build_bo_what_if_slice(
            ranked.assign(batch_number=[1, 2, 3, 4]),
            x_feature="batch_number",
            identifier_columns=("batch_number",),
        )
    with pytest.raises(ValueError, match="do not match any"):
        build_bo_what_if_slice(
            ranked,
            x_feature="laser_power_W",
            fixed_feature_values={"scan_speed_mm_s": 750.0},
        )
    with pytest.raises(ValueError, match="at least two distinct"):
        build_bo_what_if_slice(
            ranked.iloc[[0, 1]],
            x_feature="laser_power_W",
        )


def test_build_bo_what_if_slice_ignores_missing_categorical_context_values() -> None:
    ranked = _what_if_candidates()
    ranked["alloy"] = pd.Series(
        ["316L", "316L", pd.NA, pd.NA],
        dtype="string",
    )

    result = build_bo_what_if_slice(
        ranked,
        x_feature="scan_speed_mm_s",
        fixed_feature_values={"alloy": "316L"},
    )

    assert result.rows["x"].tolist() == [500.0, 1000.0]
    assert result.metadata["matched_candidate_count"] == 2


def _ranked_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["candidate-b", "candidate-a", "candidate-c"],
            "laser_power_W": [240.0, 120.0, 360.0],
            "scan_speed_mm_s": [800.0, 200.0, 1400.0],
            "hatch_spacing_mm": [0.10, 0.08, 0.14],
            "matgpr_rank": [1, 2, 3],
            "matgpr_candidate_index": [1, 0, 2],
            "matgpr_predicted_mean": [0.99, 0.98, 0.94],
            "matgpr_predicted_std": [0.01, 0.02, 0.04],
            "matgpr_preference_score": [0.90, 0.75, 0.20],
            "matgpr_objective_component": [0.90, 0.75, 0.20],
            "matgpr_feasibility_probability": [0.95, 0.85, 0.30],
            "matgpr_acquisition": [0.855, 0.6375, 0.06],
            "matgpr_constraint_probability_density": [0.95, 0.85, 0.30],
            "matgpr_feasible": [True, True, False],
            "matgpr_constraint_violations": ["", "", "density"],
        }
    )


def _what_if_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [
                "candidate-1",
                "candidate-2",
                "candidate-3",
                "candidate-4",
            ],
            "laser_power_W": [200.0, 200.0, 100.0, 100.0],
            "scan_speed_mm_s": [500.0, 1000.0, 500.0, 1000.0],
            "alloy": ["316L"] * 4,
            "matgpr_rank": [1, 2, 3, 4],
            "matgpr_candidate_index": [0, 1, 2, 3],
            "matgpr_predicted_mean": [0.99, 0.98, 0.97, 0.96],
            "matgpr_predicted_std": [0.01, 0.015, 0.02, 0.025],
            "matgpr_preference_score": [0.95, 0.85, 0.75, 0.65],
            "matgpr_objective_component": [0.95, 0.85, 0.75, 0.65],
            "matgpr_feasibility_probability": [0.99, 0.95, 0.90, 0.80],
            "matgpr_acquisition": [0.94, 0.81, 0.68, 0.52],
            "matgpr_constraint_probability_density": [0.99, 0.95, 0.90, 0.80],
            "matgpr_feasible": [True, True, True, True],
            "matgpr_constraint_violations": ["", "", "", ""],
        }
    )
