from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from matgpr import bayesian_optimization as bo

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" FILTER ", "filter"), ("Annotate", "annotate")],
)
def test_candidate_policy_normalization(value: str, expected: str) -> None:
    assert bo._validate_candidate_policy(value) == expected


def test_candidate_policy_and_output_name_validation_errors() -> None:
    with pytest.raises(ValueError, match="candidate policy"):
        bo._validate_candidate_policy("remove")
    assert bo._validate_output_name(" score ", name="output") == "score"
    with pytest.raises(ValueError, match="output must be a non-empty column name"):
        bo._validate_output_name("  ", name="output")
    bo._validate_distinct_names("a", "b")
    with pytest.raises(ValueError, match="output column names must be distinct"):
        bo._validate_distinct_names("a", "a")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("euclidean", "euclidean"),
        ("L2", "euclidean"),
        ("cityblock", "manhattan"),
        ("l1", "manhattan"),
        ("L-Inf", "chebyshev"),
        ("max", "chebyshev"),
    ],
)
def test_distance_metric_aliases(value: str, expected: str) -> None:
    assert bo._normalize_distance_metric(value) == expected


def test_distance_metric_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="metric must be"):
        bo._normalize_distance_metric("cosine")


def test_policy_feature_matrix_uses_requested_or_inferred_dataframe_columns() -> None:
    frame = pd.DataFrame({1: [1.0, 2.0], "x": [3.0, 4.0], "label": ["a", "b"]})

    explicit = bo._policy_feature_matrix(
        frame,
        feature_columns=("1", "x"),
        allow_empty_rows=False,
        name="features",
    )
    inferred = bo._policy_feature_matrix(
        frame,
        feature_columns=None,
        allow_empty_rows=False,
        name="features",
    )

    np.testing.assert_allclose(explicit, [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(inferred, explicit)


def test_policy_feature_matrix_normalizes_one_dimensional_arrays() -> None:
    matrix = bo._policy_feature_matrix(
        np.asarray([1.0, 2.0]),
        feature_columns=("a", "b"),
        expected_features=2,
        expected_rows=1,
        allow_empty_rows=False,
        name="features",
    )
    assert matrix.shape == (1, 2)


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        (
            pd.DataFrame({"x": [1.0]}),
            {"feature_columns": ("missing",), "allow_empty_rows": False},
            "feature columns are missing",
        ),
        (
            pd.DataFrame({"label": ["a"]}),
            {"feature_columns": None, "allow_empty_rows": False},
            "No numeric columns",
        ),
        (
            pd.DataFrame({"label": ["a"]}),
            {"feature_columns": ("label",), "allow_empty_rows": False},
            "feature columns must be numeric",
        ),
        (
            np.ones((1, 1, 1)),
            {"feature_columns": None, "allow_empty_rows": False},
            "must be a 2D numeric array",
        ),
        (
            np.empty((0, 2)),
            {"feature_columns": None, "allow_empty_rows": False},
            "must contain at least one row",
        ),
        (
            np.empty((1, 0)),
            {"feature_columns": None, "allow_empty_rows": True},
            "must contain at least one feature",
        ),
        (
            np.ones((1, 2)),
            {
                "feature_columns": ("only_one",),
                "allow_empty_rows": False,
            },
            "feature_columns length",
        ),
        (
            np.ones((1, 2)),
            {
                "feature_columns": None,
                "expected_features": 3,
                "allow_empty_rows": False,
            },
            "feature count must match",
        ),
        (
            np.ones((1, 2)),
            {
                "feature_columns": None,
                "expected_rows": 2,
                "allow_empty_rows": False,
            },
            "rows must match",
        ),
        (
            np.asarray([[np.nan]]),
            {"feature_columns": None, "allow_empty_rows": False},
            "contains NaN or infinite values",
        ),
    ],
)
def test_policy_feature_matrix_validation(
    values: pd.DataFrame | np.ndarray,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bo._policy_feature_matrix(values, name="features", **kwargs)


def test_policy_feature_scales_support_raw_standardized_and_explicit_modes() -> None:
    candidates = np.asarray([[0.0, 2.0], [2.0, 2.0]])
    references = np.asarray([[4.0, 2.0]])

    np.testing.assert_allclose(
        bo._resolve_policy_feature_scales(None, candidates, references, name="scales"),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        bo._resolve_policy_feature_scales("std", candidates, references, name="scales"),
        [np.std([0.0, 2.0, 4.0]), 1.0],
    )
    np.testing.assert_allclose(
        bo._resolve_policy_feature_scales([2.0, 4.0], candidates, references, name="scales"),
        [2.0, 4.0],
    )


def test_policy_feature_scales_handle_empty_candidate_or_reference_rows() -> None:
    candidate_only = bo._resolve_policy_feature_scales(
        "standardize",
        np.asarray([[0.0], [2.0]]),
        np.empty((0, 1)),
        name="scales",
    )
    reference_only = bo._resolve_policy_feature_scales(
        "standard_deviation",
        np.empty((0, 1)),
        np.asarray([[0.0], [4.0]]),
        name="scales",
    )
    np.testing.assert_allclose(candidate_only, [1.0])
    np.testing.assert_allclose(reference_only, [2.0])


@pytest.mark.parametrize(
    ("scales", "message"),
    [
        ("range", "must be None"),
        ([1.0], "length must match"),
        ([1.0, 0.0], "finite positive"),
        ([1.0, np.inf], "finite positive"),
    ],
)
def test_policy_feature_scale_validation(scales: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bo._resolve_policy_feature_scales(
            scales,
            np.ones((1, 2)),
            np.ones((1, 2)),
            name="scales",
        )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("euclidean", [np.sqrt(17.0), 1.0]),
        ("manhattan", [5.0, 1.0]),
        ("chebyshev", [4.0, 1.0]),
    ],
)
def test_minimum_scaled_distances(metric: str, expected: list[float]) -> None:
    distances = bo._minimum_scaled_distances(
        np.asarray([[3.0, 4.0], [1.0, 0.0]]),
        np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        metric=metric,
        scales=np.ones(2),
    )
    np.testing.assert_allclose(distances, expected)


def test_minimum_scaled_distance_empty_and_invalid_metric_paths() -> None:
    empty = bo._minimum_scaled_distances(
        np.empty((0, 2)), np.ones((1, 2)), metric="euclidean", scales=np.ones(2)
    )
    no_references = bo._minimum_scaled_distances(
        np.ones((2, 2)), np.empty((0, 2)), metric="euclidean", scales=np.ones(2)
    )
    assert empty.shape == (0,)
    assert np.isinf(no_references).all()
    with pytest.raises(ValueError, match="metric must be"):
        bo._minimum_scaled_distances(
            np.ones((1, 1)), np.ones((1, 1)), metric="cosine", scales=np.ones(1)
        )


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_minimum_scaled_distance_matches_returns_stable_nearest_positions(metric: str) -> None:
    distances, positions = bo._minimum_scaled_distance_matches(
        np.asarray([[1.0, 0.0], [9.0, 0.0]]),
        np.asarray([[0.0, 0.0], [10.0, 0.0]]),
        metric=metric,
        scales=np.ones(2),
    )
    np.testing.assert_allclose(distances, [1.0, 1.0])
    assert positions.tolist() == [0, 1]


def test_minimum_scaled_distance_matches_empty_and_invalid_metric_paths() -> None:
    empty_distances, empty_positions = bo._minimum_scaled_distance_matches(
        np.empty((0, 1)), np.ones((1, 1)), metric="euclidean", scales=np.ones(1)
    )
    distances, positions = bo._minimum_scaled_distance_matches(
        np.ones((2, 1)), np.empty((0, 1)), metric="euclidean", scales=np.ones(1)
    )
    assert empty_distances.shape == empty_positions.shape == (0,)
    assert np.isinf(distances).all()
    assert positions.tolist() == [-1, -1]
    with pytest.raises(ValueError, match="metric must be"):
        bo._minimum_scaled_distance_matches(
            np.ones((1, 1)), np.ones((1, 1)), metric="cosine", scales=np.ones(1)
        )


def test_existing_candidate_ids_support_explicit_ids_indexes_and_arrays() -> None:
    frame = pd.DataFrame({"experiment_id": [17, " exp-18 "]}, index=[10, 20])
    assert bo._existing_candidate_ids(frame, "experiment_id").tolist() == ["17", "exp-18"]
    assert bo._existing_candidate_ids(frame, None).tolist() == ["10", "20"]
    assert bo._existing_candidate_ids(np.asarray([1.0, 2.0]), None).tolist() == ["0"]
    assert bo._existing_candidate_ids(np.ones((2, 1)), None).tolist() == ["0", "1"]


@pytest.mark.parametrize(
    ("frame", "column", "message"),
    [
        (pd.DataFrame({"id": ["a"]}), "missing", "ID column is missing"),
        (pd.DataFrame({"id": [None]}), "id", "must not contain missing"),
        (pd.DataFrame({"id": [" "]}), "id", "must not contain empty"),
        (pd.DataFrame({"id": ["a", "a"]}), "id", "must be unique"),
    ],
)
def test_existing_candidate_id_validation(
    frame: pd.DataFrame,
    column: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bo._existing_candidate_ids(frame, column)


def test_existing_candidate_id_column_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="requires existing_candidates to be a dataframe"):
        bo._existing_candidate_ids(np.ones((1, 1)), "id")


def test_candidate_policy_row_keys_use_a_stable_missing_sentinel() -> None:
    frame = pd.DataFrame({"composition": ["Ni", None, np.nan], "batch": [1, 2, 2]})
    keys = bo._candidate_policy_row_keys(
        frame,
        ("composition", "batch"),
        label="candidate key columns",
    )
    assert keys[1] == keys[2]
    assert keys[0] == ("Ni", 1)
    with pytest.raises(ValueError, match="candidate key columns are missing"):
        bo._candidate_policy_row_keys(frame, ("missing",), label="candidate key columns")


def test_trust_region_annotations_apply_scaling_and_allow_empty_candidates() -> None:
    region = bo.CandidateTrustRegion(
        centers=pd.DataFrame({"x": [0.0], "y": [0.0]}),
        radius=1.0,
        metric="manhattan",
        feature_columns=("x", "y"),
        feature_scales=(2.0, 1.0),
    )
    distances, in_region = bo._trust_region_annotations(
        pd.DataFrame({"x": [2.0, 4.0], "y": [0.0, 1.0]}),
        region,
        expected_rows=2,
    )
    empty_distances, empty_mask = bo._trust_region_annotations(
        pd.DataFrame({"x": [], "y": []}),
        region,
        expected_rows=0,
    )
    np.testing.assert_allclose(distances, [1.0, 3.0])
    assert in_region.tolist() == [True, False]
    assert empty_distances.shape == empty_mask.shape == (0,)


def test_duplicate_annotations_combine_key_and_feature_reasons_with_id_precedence() -> None:
    candidates = pd.DataFrame({"material": ["Ni", "Co", "Fe"], "x": [0.0, 1.05, 4.0]})
    existing = pd.DataFrame(
        {"experiment_id": ["exp-ni", "exp-co"], "material": ["Ni", "Cu"], "x": [0.0, 1.0]}
    )
    policy = bo.CandidateDuplicatePolicy(
        existing_candidates=existing,
        key_columns=("material",),
        feature_columns=("x",),
        feature_tolerance=0.1,
        existing_id_column="experiment_id",
    )

    duplicates, distances, reasons, matched_ids = bo._duplicate_policy_annotations(
        candidates,
        candidates.loc[:, ["x"]],
        policy,
        expected_rows=3,
    )

    assert duplicates.tolist() == [True, True, False]
    np.testing.assert_allclose(distances, [0.0, 0.05, 3.0])
    assert reasons.tolist() == ["key; feature", "feature", ""]
    assert matched_ids.tolist() == ["exp-ni", "exp-co", None]


def test_duplicate_annotations_require_dataframe_for_key_matching() -> None:
    policy = bo.CandidateDuplicatePolicy(
        existing_candidates=np.asarray([[0.0]]),
        key_columns=("material",),
    )
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        bo._duplicate_policy_annotations(
            pd.DataFrame({"material": ["Ni"]}),
            np.asarray([[0.0]]),
            policy,
            expected_rows=1,
        )


def test_duplicate_annotations_support_key_only_policy() -> None:
    candidates = pd.DataFrame({"material": ["Ni", "Fe"]})
    policy = bo.CandidateDuplicatePolicy(
        existing_candidates=pd.DataFrame({"material": ["Ni"]}),
        key_columns=("material",),
    )
    duplicates, distances, reasons, matched_ids = bo._duplicate_policy_annotations(
        candidates,
        np.empty((2, 0)),
        policy,
        expected_rows=2,
    )
    assert duplicates.tolist() == [True, False]
    assert distances is None
    assert reasons.tolist() == ["key", ""]
    assert matched_ids.tolist() == ["0", None]


def test_duplicate_annotations_support_empty_existing_feature_pool() -> None:
    policy = bo.CandidateDuplicatePolicy(
        existing_candidates=np.empty((0, 1)),
        feature_tolerance=0.1,
    )
    duplicates, distances, reasons, matched_ids = bo._duplicate_policy_annotations(
        pd.DataFrame({"material": ["Ni"]}),
        np.asarray([[0.0]]),
        policy,
        expected_rows=1,
    )
    assert duplicates.tolist() == [False]
    assert np.isinf(distances).all()
    assert reasons.tolist() == [""]
    assert matched_ids.tolist() == [None]


def test_combined_candidate_domain_policy_preserves_annotations_and_mask() -> None:
    candidates = pd.DataFrame({"id": ["old", "near", "far"], "x": [0.0, 0.5, 3.0]})
    trust_region = bo.CandidateTrustRegion(
        centers=np.asarray([[0.0]]),
        radius=1.0,
        feature_columns=("x",),
    )
    duplicate_policy = bo.CandidateDuplicatePolicy(
        existing_candidates=pd.DataFrame({"id": ["old"], "x": [0.0]}),
        key_columns=("id",),
        feature_columns=("x",),
        feature_tolerance=0.1,
    )

    result, mask = bo._apply_candidate_domain_policies_with_mask(
        candidates,
        candidates.loc[:, ["x"]],
        trust_region=trust_region,
        trust_region_policy="filter",
        duplicate_policy=duplicate_policy,
        duplicate_policy_action="filter",
    )

    assert mask.tolist() == [False, True, False]
    assert result["id"].tolist() == ["near"]
    assert "matgpr_trust_region_distance" in result
    assert "matgpr_duplicate_reason" in result


def test_candidate_domain_policy_noop_and_type_validation() -> None:
    candidates = pd.DataFrame({"id": ["a"], "x": [0.0]})
    result, mask = bo._apply_candidate_domain_policies_with_mask(
        candidates,
        candidates.loc[:, ["x"]],
        trust_region=None,
        trust_region_policy="annotate",
        duplicate_policy=None,
        duplicate_policy_action="annotate",
    )
    pd.testing.assert_frame_equal(result, candidates.reset_index(drop=True))
    assert mask.tolist() == [True]

    with pytest.raises(TypeError, match="candidates must be a pandas DataFrame"):
        bo._apply_candidate_domain_policies_with_mask(
            np.asarray([[0.0]]),
            np.asarray([[0.0]]),
            trust_region=None,
            trust_region_policy="filter",
            duplicate_policy=None,
            duplicate_policy_action="filter",
        )
    with pytest.raises(TypeError, match="trust_region must be"):
        bo._apply_candidate_domain_policies_with_mask(
            candidates,
            np.asarray([[0.0]]),
            trust_region=object(),
            trust_region_policy="filter",
            duplicate_policy=None,
            duplicate_policy_action="filter",
        )
    with pytest.raises(TypeError, match="duplicate_policy must be"):
        bo._apply_candidate_domain_policies_with_mask(
            candidates,
            np.asarray([[0.0]]),
            trust_region=None,
            trust_region_policy="filter",
            duplicate_policy=object(),
            duplicate_policy_action="filter",
        )


def test_public_candidate_policy_helpers_support_custom_columns_and_candidate_features() -> None:
    candidates = pd.DataFrame({"id": ["a", "b"]}, index=[4, 8])
    annotated = bo.apply_candidate_trust_region(
        candidates,
        bo.CandidateTrustRegion(centers=np.asarray([[0.0]]), radius=0.5),
        candidate_features=np.asarray([[0.0], [1.0]]),
        policy="annotate",
        in_region_column="inside",
        distance_column="distance",
    )
    deduplicated = bo.apply_candidate_duplicate_policy(
        candidates,
        bo.CandidateDuplicatePolicy(
            existing_candidates=np.asarray([[0.0]]),
            feature_tolerance=0.1,
        ),
        candidate_features=np.asarray([[0.0], [1.0]]),
        policy="annotate",
        duplicate_column="duplicate",
        distance_column="distance",
        reason_column="reason",
        matched_id_column="matched",
    )
    assert annotated.index.tolist() == [0, 1]
    assert annotated["inside"].tolist() == [True, False]
    assert deduplicated["duplicate"].tolist() == [True, False]
    assert deduplicated["matched"].tolist() == ["0", None]

    with pytest.raises(ValueError, match="must be distinct"):
        bo.apply_candidate_trust_region(
            candidates,
            bo.CandidateTrustRegion(centers=np.asarray([[0.0]]), radius=1.0),
            candidate_features=np.asarray([[0.0], [1.0]]),
            in_region_column="same",
            distance_column="same",
        )
    with pytest.raises(ValueError, match="must be distinct"):
        bo.apply_candidate_duplicate_policy(
            candidates,
            bo.CandidateDuplicatePolicy(
                existing_candidates=np.asarray([[0.0]]), feature_tolerance=0.1
            ),
            candidate_features=np.asarray([[0.0], [1.0]]),
            duplicate_column="same",
            reason_column="same",
        )


def test_parent_candidate_domain_helpers_are_exact_internal_aliases() -> None:
    policy = importlib.import_module("matgpr._bo_candidate_domain")
    names = (
        "_candidate_policy_row_keys",
        "_duplicate_policy_annotations",
        "_existing_candidate_ids",
        "_minimum_scaled_distance_matches",
        "_minimum_scaled_distances",
        "_normalize_distance_metric",
        "_policy_feature_matrix",
        "_resolve_policy_feature_columns",
        "_resolve_policy_feature_scales",
        "_trust_region_annotations",
        "_validate_candidate_policy",
        "_validate_distinct_names",
        "_validate_output_name",
    )
    for name in names:
        assert getattr(bo, name) is getattr(policy, name)
