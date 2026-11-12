from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from matgpr import bayesian_optimization as bo

pytestmark = pytest.mark.unit


def test_finite_numeric_vector_converts_values_and_rejects_nonfinite() -> None:
    vector = bo._as_finite_numeric_vector(pd.Series(["1.5", 2]), name="score")
    np.testing.assert_allclose(vector, [1.5, 2.0])

    with pytest.raises(ValueError, match="score contains NaN or infinite values"):
        bo._as_finite_numeric_vector(pd.Series([1.0, np.inf]), name="score")


def test_finite_numeric_vector_rejects_non_vector_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Converted:
        def to_numpy(self, *, dtype: type[float]) -> np.ndarray:
            assert dtype is float
            return np.ones((1, 1), dtype=float)

    monkeypatch.setattr(pd, "to_numeric", lambda *args, **kwargs: _Converted())
    with pytest.raises(ValueError, match="score must be one-dimensional"):
        bo._as_finite_numeric_vector(pd.Series([1.0]), name="score")


def test_vector_normalization_handles_constant_and_varying_scores() -> None:
    np.testing.assert_allclose(bo._normalize_vector(np.asarray([3.0, 3.0])), [1.0, 1.0])
    np.testing.assert_allclose(bo._normalize_vector(np.asarray([-2.0, 0.0, 2.0])), [0.0, 0.5, 1.0])


def test_top_score_positions_and_descending_ranks_use_stable_ties() -> None:
    scores = np.asarray([2.0, 3.0, 3.0, 1.0])
    assert bo._top_score_positions(scores, 10).tolist() == [1, 2, 0, 3]
    assert bo._rank_descending(pd.Series(scores, name="utility")).tolist() == [3, 1, 2, 4]
    assert bo._rank_descending(pd.Series([1.0, 1.0])).tolist() == [1, 2]


def test_stable_argmax_prefers_first_tie_and_handles_no_finite_value() -> None:
    assert bo._argmax_stable(np.asarray([1.0, 2.0, 2.0])) == 1
    assert bo._argmax_stable(np.asarray([np.nan, -np.inf])) is None


def test_explicit_diversity_feature_resolution_preserves_order() -> None:
    candidates = pd.DataFrame({"x": [0.0], "y": [1.0], "score": [2.0]})
    assert bo._resolve_diversity_feature_columns(
        candidates,
        score_column="score",
        feature_columns=("y", "x"),
    ) == ("y", "x")

    with pytest.raises(ValueError, match="feature columns are missing"):
        bo._resolve_diversity_feature_columns(
            candidates,
            score_column="score",
            feature_columns=("missing",),
        )
    with pytest.raises(ValueError, match="at least one column"):
        bo._resolve_diversity_feature_columns(
            candidates,
            score_column="score",
            feature_columns=(),
        )


def test_inferred_diversity_features_exclude_scores_outputs_and_labels() -> None:
    candidates = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "label": ["a", "b"],
            "score": [1.0, 0.5],
            "matgpr_batch_score": [0.2, 0.1],
            "matgpr_unreserved_numeric": [2.0, 3.0],
        }
    )
    assert bo._resolve_diversity_feature_columns(
        candidates,
        score_column="score",
        feature_columns=None,
    ) == ("x", "matgpr_unreserved_numeric")

    with pytest.raises(ValueError, match="No numeric diversity feature columns"):
        bo._resolve_diversity_feature_columns(
            pd.DataFrame({"score": [1.0], "label": ["a"]}),
            score_column="score",
            feature_columns=None,
        )


def test_diversity_matrix_removes_constant_columns_without_standardizing() -> None:
    candidates = pd.DataFrame({"x": [0.0, 2.0], "constant": [5.0, 5.0], "score": [1.0, 0.5]})
    matrix, columns = bo._candidate_diversity_matrix(
        candidates,
        score_column="score",
        feature_columns=("constant", "x"),
        standardize_features=False,
    )
    np.testing.assert_allclose(matrix, [[0.0], [2.0]])
    assert columns == ("x",)


def test_diversity_matrix_standardizes_nonconstant_features() -> None:
    candidates = pd.DataFrame({"x": [0.0, 2.0, 4.0], "score": [1.0, 0.5, 0.0]})
    matrix, columns = bo._candidate_diversity_matrix(
        candidates,
        score_column="score",
        feature_columns=("x",),
        standardize_features=True,
    )
    np.testing.assert_allclose(matrix.mean(axis=0), [0.0], atol=1e-12)
    np.testing.assert_allclose(matrix.std(axis=0), [1.0])
    assert columns == ("x",)


@pytest.mark.parametrize(
    ("candidates", "features", "message"),
    [
        (
            pd.DataFrame({"label": ["a", "b"], "score": [1.0, 0.0]}),
            ("label",),
            "must be numeric",
        ),
        (
            pd.DataFrame({"x": [0.0, np.nan], "score": [1.0, 0.0]}),
            ("x",),
            "contain NaN or infinite values",
        ),
        (
            pd.DataFrame({"x": [1.0, 1.0], "score": [1.0, 0.0]}),
            ("x",),
            "at least one nonconstant numeric feature",
        ),
    ],
)
def test_diversity_matrix_validation(
    candidates: pd.DataFrame,
    features: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bo._candidate_diversity_matrix(
            candidates,
            score_column="score",
            feature_columns=features,
            standardize_features=True,
        )


def test_greedy_diversity_selection_is_stable_then_prefers_distance() -> None:
    candidates = pd.DataFrame({"score": [1.0, 1.0, 1.0], "x": [0.0, 0.1, 10.0]})
    positions, scores, distances = bo._greedy_diverse_positions(
        candidates,
        top_k=2,
        score_column="score",
        feature_columns=("x",),
        normalized_scores=np.ones(3),
        diversity_weight=1.0,
        min_distance=None,
        standardize_features=False,
    )
    assert positions.tolist() == [0, 2]
    np.testing.assert_allclose(scores, [1.0, 2.0])
    assert np.isnan(distances[0])
    np.testing.assert_allclose(distances[1], 10.0)


def test_greedy_diversity_minimum_distance_stops_short_batch() -> None:
    candidates = pd.DataFrame({"score": [1.0, 0.9, 0.8], "x": [0.0, 0.1, 0.2]})
    positions, scores, distances = bo._greedy_diverse_positions(
        candidates,
        top_k=3,
        score_column="score",
        feature_columns=("x",),
        normalized_scores=np.asarray([1.0, 0.5, 0.0]),
        diversity_weight=0.25,
        min_distance=1.0,
        standardize_features=False,
    )
    assert positions.tolist() == [0]
    np.testing.assert_allclose(scores, [1.0])
    assert np.isnan(distances[0])


def test_greedy_diversity_handles_zero_remaining_distance_scale() -> None:
    candidates = pd.DataFrame({"score": [1.0, 0.9, 0.8], "x": [0.0, 1.0, 0.0]})
    positions, scores, distances = bo._greedy_diverse_positions(
        candidates,
        top_k=3,
        score_column="score",
        feature_columns=("x",),
        normalized_scores=np.asarray([1.0, 0.5, 0.0]),
        diversity_weight=1.0,
        min_distance=None,
        standardize_features=False,
    )
    assert positions.tolist() == [0, 1, 2]
    np.testing.assert_allclose(scores, [1.0, 1.5, 0.0])
    assert np.isnan(distances[0])
    np.testing.assert_allclose(distances[1:], [1.0, 0.0])


def test_greedy_diversity_defensive_shape_and_feature_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function_globals = bo._greedy_diverse_positions.__globals__
    candidates = pd.DataFrame({"score": [1.0, 0.5], "x": [0.0, 1.0]})

    monkeypatch.setitem(
        function_globals,
        "_candidate_diversity_matrix",
        lambda *args, **kwargs: (np.ones((1, 1)), ("x",)),
    )
    with pytest.raises(ValueError, match="rows must match"):
        bo._greedy_diverse_positions(
            candidates,
            top_k=1,
            score_column="score",
            feature_columns=("x",),
            normalized_scores=np.ones(2),
            diversity_weight=1.0,
            min_distance=None,
            standardize_features=False,
        )

    monkeypatch.setitem(
        function_globals,
        "_candidate_diversity_matrix",
        lambda *args, **kwargs: (np.ones((2, 1)), ()),
    )
    with pytest.raises(ValueError, match="at least one diversity feature"):
        bo._greedy_diverse_positions(
            candidates,
            top_k=1,
            score_column="score",
            feature_columns=("x",),
            normalized_scores=np.ones(2),
            diversity_weight=1.0,
            min_distance=None,
            standardize_features=False,
        )


def test_greedy_diversity_stops_when_stable_argmax_has_no_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function_globals = bo._greedy_diverse_positions.__globals__
    monkeypatch.setitem(function_globals, "_argmax_stable", lambda values: None)
    positions, scores, distances = bo._greedy_diverse_positions(
        pd.DataFrame({"score": [1.0, 0.5], "x": [0.0, 1.0]}),
        top_k=2,
        score_column="score",
        feature_columns=("x",),
        normalized_scores=np.asarray([1.0, 0.0]),
        diversity_weight=1.0,
        min_distance=None,
        standardize_features=False,
    )
    assert positions.size == scores.size == distances.size == 0


def test_public_diverse_batch_supports_lower_scores_and_empty_frames() -> None:
    candidates = pd.DataFrame({"id": ["a", "b", "c"], "score": [2.0, 1.0, 3.0]})
    selected = bo.select_diverse_batch(
        candidates,
        top_k=2,
        score_column="score",
        diversity_weight=0.0,
        higher_is_better=False,
    )
    assert selected["id"].tolist() == ["b", "a"]
    assert selected["matgpr_batch_order"].tolist() == [1, 2]

    empty = bo.select_diverse_batch(
        pd.DataFrame({"score": pd.Series(dtype=float)}),
        top_k=1,
        score_column="score",
    )
    assert empty.empty
    assert {
        "matgpr_batch_selected",
        "matgpr_batch_order",
        "matgpr_batch_score",
        "matgpr_diversity_distance",
    }.issubset(empty.columns)


def test_public_diverse_batch_handles_duplicate_input_index_by_position() -> None:
    candidates = pd.DataFrame(
        {"id": ["first", "second", "third"], "score": [1.0, 0.9, 0.8]},
        index=[7, 7, 9],
    )
    selected = bo.select_diverse_batch(
        candidates,
        top_k=1,
        score_column="score",
        diversity_weight=0.0,
    )
    assert selected["id"].tolist() == ["first"]


def test_parent_batch_selection_helpers_are_exact_internal_aliases() -> None:
    policy = importlib.import_module("matgpr._bo_batch_selection")
    names = (
        "_OUTPUT_COLUMNS",
        "_argmax_stable",
        "_as_finite_numeric_vector",
        "_candidate_diversity_matrix",
        "_greedy_diverse_positions",
        "_normalize_vector",
        "_rank_descending",
        "_resolve_diversity_feature_columns",
        "_top_score_positions",
    )
    for name in names:
        assert getattr(bo, name) is getattr(policy, name)
