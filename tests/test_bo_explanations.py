from __future__ import annotations

import pandas as pd
import pytest

from matgpr import BOExplanationTables, build_bo_explanation_tables


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
